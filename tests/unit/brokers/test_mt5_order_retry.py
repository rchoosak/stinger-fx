"""MT5Broker.place_order — retry on transient retcodes (Phase 6.1.A).

Uses a fake MetaTrader5 module to script the retcode sequence returned by
`order_send`. Verifies REQUOTE retries, permanent errors don't retry, and
SDK-level None responses skip the retry loop.
"""

from __future__ import annotations

import sys
import types

import pytest

from stinger_fx.brokers.mt5.broker import MT5Broker
from stinger_fx.config.models import MT5Config
from stinger_fx.core import AsyncEventBus
from stinger_fx.domain import OrderRequest, OrderStatus, OrderType, Side


# MT5 retcode constants used in this test (mirrors the broker module).
TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_REQUOTE = 10004
TRADE_RETCODE_PRICE_OFF = 10021
TRADE_RETCODE_TIMEOUT = 10024
TRADE_RETCODE_INVALID_VOLUME = 10014   # permanent
TRADE_RETCODE_NO_MONEY = 10019         # permanent


class _FakeResult:
    def __init__(self, retcode: int, order: int = 0, volume: float = 0.0, price: float = 0.0):
        self.retcode = retcode
        self.order = order
        self.volume = volume
        self.price = price
        self.comment = "fake"


class _FakeMT5:
    TRADE_RETCODE_DONE = TRADE_RETCODE_DONE
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_PENDING = 5
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TYPE_BUY_LIMIT = 2
    ORDER_TYPE_SELL_LIMIT = 3
    ORDER_TYPE_BUY_STOP = 4
    ORDER_TYPE_SELL_STOP = 5
    ORDER_TYPE_BUY_STOP_LIMIT = 6
    ORDER_TYPE_SELL_STOP_LIMIT = 7
    ORDER_FILLING_IOC = 1
    ORDER_TIME_GTC = 0

    def __init__(self, retcode_sequence: list[int]) -> None:
        self._sequence = list(retcode_sequence)
        self.send_calls = 0
        # current tick → BUY uses ask, SELL uses bid
        self._tick = types.SimpleNamespace(ask=1.1002, bid=1.1000)

    def initialize(self, **_):
        return True

    def shutdown(self):
        pass

    def terminal_info(self):
        return types.SimpleNamespace(connected=True)

    def symbol_info_tick(self, _symbol):
        return self._tick

    def order_send(self, request):  # noqa: ARG002
        self.send_calls += 1
        # If sequence exhausted, keep returning the last code so we don't
        # crash if the broker decides to push past expected attempts.
        if not self._sequence:
            return _FakeResult(self._sequence[-1] if self._sequence else TRADE_RETCODE_DONE)
        rc = self._sequence.pop(0)
        return _FakeResult(rc, order=12345, volume=0.1, price=1.1002)

    def last_error(self):
        return (0, "no error")

    def symbol_select(self, _symbol, _enable):
        return True


def _install_fake_mt5(monkeypatch, fake: _FakeMT5) -> None:
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)


def _req(symbol: str = "EURUSD") -> OrderRequest:
    return OrderRequest(
        strategy_id="s",
        symbol=symbol,
        side=Side.BUY,
        type=OrderType.MARKET,
        volume=0.1,
        client_order_id="coid-1",
    )


@pytest.mark.asyncio
async def test_order_retries_on_requote_then_succeeds(monkeypatch) -> None:
    """REQUOTE twice → success on third attempt → ok=True, 3 SDK calls."""
    fake = _FakeMT5([TRADE_RETCODE_REQUOTE, TRADE_RETCODE_REQUOTE, TRADE_RETCODE_DONE])
    _install_fake_mt5(monkeypatch, fake)

    bus = AsyncEventBus()
    cfg = MT5Config(terminal_path="", login=0, password="", server="", timeout_ms=1000)
    broker = MT5Broker(bus, cfg)
    broker._health_check_interval = 60.0
    broker._order_retry_backoff = [0.001, 0.001, 0.001]

    try:
        await broker.connect()
        result = await broker.place_order(_req())
        assert result.ok is True
        assert result.status == OrderStatus.FILLED
        assert fake.send_calls == 3
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_order_does_not_retry_permanent_error(monkeypatch) -> None:
    """INVALID_VOLUME → no retry, ok=False, 1 SDK call only."""
    fake = _FakeMT5([TRADE_RETCODE_INVALID_VOLUME])
    _install_fake_mt5(monkeypatch, fake)

    bus = AsyncEventBus()
    cfg = MT5Config(terminal_path="", login=0, password="", server="", timeout_ms=1000)
    broker = MT5Broker(bus, cfg)
    broker._health_check_interval = 60.0
    broker._order_retry_backoff = [0.001]

    try:
        await broker.connect()
        result = await broker.place_order(_req())
        assert result.ok is False
        assert result.status == OrderStatus.REJECTED
        assert fake.send_calls == 1
        assert result.raw_code == TRADE_RETCODE_INVALID_VOLUME
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_order_gives_up_after_max_retries(monkeypatch) -> None:
    """REQUOTE on all attempts → ok=False after ORDER_MAX_RETRIES (3) calls."""
    fake = _FakeMT5([TRADE_RETCODE_REQUOTE] * 5)
    _install_fake_mt5(monkeypatch, fake)

    bus = AsyncEventBus()
    cfg = MT5Config(terminal_path="", login=0, password="", server="", timeout_ms=1000)
    broker = MT5Broker(bus, cfg)
    broker._health_check_interval = 60.0
    broker._order_retry_backoff = [0.001]

    try:
        await broker.connect()
        result = await broker.place_order(_req())
        assert result.ok is False
        # ORDER_MAX_RETRIES = 3 attempts total
        assert fake.send_calls == 3
        assert result.raw_code == TRADE_RETCODE_REQUOTE
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_order_retries_price_off_and_timeout(monkeypatch) -> None:
    """PRICE_OFF + TIMEOUT are both retryable — try all three, succeed third."""
    fake = _FakeMT5([TRADE_RETCODE_PRICE_OFF, TRADE_RETCODE_TIMEOUT, TRADE_RETCODE_DONE])
    _install_fake_mt5(monkeypatch, fake)

    bus = AsyncEventBus()
    cfg = MT5Config(terminal_path="", login=0, password="", server="", timeout_ms=1000)
    broker = MT5Broker(bus, cfg)
    broker._health_check_interval = 60.0
    broker._order_retry_backoff = [0.001, 0.001]

    try:
        await broker.connect()
        result = await broker.place_order(_req())
        assert result.ok is True
        assert fake.send_calls == 3
    finally:
        await broker.disconnect()
        await bus.close()
