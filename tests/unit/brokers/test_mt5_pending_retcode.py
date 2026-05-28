"""Regression tests for the MT5 pending-order retcode mapping hotfix.

Pre-fix bug: ``MT5Broker.place_order()`` accepted only ``TRADE_RETCODE_DONE``
(10009) as success and mapped every other retcode — including
``TRADE_RETCODE_PLACED`` (10008, "pending order queued at broker") and
``TRADE_RETCODE_DONE_PARTIAL`` (10010, "only part of the requested volume
filled") — to ``OrderStatus.REJECTED``. That meant live STOP / LIMIT
orders that MT5 actually accepted were reported as rejected, blocking
breakout / pullback strategies entirely.

These tests pin the retcode → OrderStatus map:

  | retcode               | ok    | status              |
  |-----------------------|-------|---------------------|
  | DONE (10009)          | True  | FILLED              |
  | PLACED (10008)        | True  | SUBMITTED           |
  | DONE_PARTIAL (10010)  | True  | PARTIALLY_FILLED    |
  | anything else         | False | REJECTED            |
"""

from __future__ import annotations

import sys
import types

import pytest

from stinger_fx.brokers.mt5.broker import MT5Broker
from stinger_fx.config.models import MT5Config
from stinger_fx.core import AsyncEventBus
from stinger_fx.domain import OrderRequest, OrderStatus, OrderType, Side


TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_PLACED = 10008
TRADE_RETCODE_DONE_PARTIAL = 10010
TRADE_RETCODE_REQUOTE = 10004
TRADE_RETCODE_INVALID_VOLUME = 10014


class _FakeResult:
    def __init__(
        self, retcode: int, order: int = 0,
        volume: float = 0.0, price: float = 0.0,
    ):
        self.retcode = retcode
        self.order = order
        self.volume = volume
        self.price = price
        self.comment = "fake"


class _FakeMT5:
    """Minimal MetaTrader5 stand-in with scriptable retcodes."""

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

    def __init__(self, retcode: int, volume: float = 0.0, price: float = 0.0):
        self._retcode = retcode
        self._volume = volume
        self._price = price
        self.send_calls = 0
        self._tick = types.SimpleNamespace(ask=1.1002, bid=1.1000)

    def initialize(self, **_):
        return True

    def shutdown(self):
        pass

    def terminal_info(self):
        return types.SimpleNamespace(connected=True)

    def symbol_info_tick(self, _symbol):
        return self._tick

    def order_send(self, _request):
        self.send_calls += 1
        return _FakeResult(
            self._retcode, order=42, volume=self._volume, price=self._price,
        )

    def last_error(self):
        return (0, "no error")

    def symbol_select(self, _s, _e):
        return True


def _install(monkeypatch, fake):
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)


def _market_req() -> OrderRequest:
    return OrderRequest(
        strategy_id="s", symbol="EURUSD", side=Side.BUY,
        type=OrderType.MARKET, volume=0.1, client_order_id="m1",
    )


def _stop_req() -> OrderRequest:
    return OrderRequest(
        strategy_id="s", symbol="EURUSD", side=Side.BUY,
        type=OrderType.STOP, volume=0.1, price=1.1050, client_order_id="p1",
    )


def _make_broker(bus, fake):
    cfg = MT5Config(terminal_path="", login=0, password="", server="", timeout_ms=1000)
    broker = MT5Broker(bus, cfg)
    broker._health_check_interval = 60.0
    broker._order_retry_backoff = [0.001]
    return broker


# --- DONE retcode → FILLED (regression: existing behaviour preserved) -------


@pytest.mark.asyncio
async def test_market_done_maps_to_filled(monkeypatch) -> None:
    fake = _FakeMT5(TRADE_RETCODE_DONE, volume=0.1, price=1.1001)
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus, fake)
    try:
        await broker.connect()
        result = await broker.place_order(_market_req())
        assert result.ok is True
        assert result.status == OrderStatus.FILLED
        assert result.order is not None
        assert result.order.status == OrderStatus.FILLED
        assert result.order.filled_volume == pytest.approx(0.1)
        assert result.order.fill_price == pytest.approx(1.1001)
        assert result.order.filled_at is not None
    finally:
        await broker.disconnect()
        await bus.close()


# --- PLACED retcode → SUBMITTED (THE BUG FIX) ------------------------------


@pytest.mark.asyncio
async def test_pending_placed_maps_to_submitted_not_rejected(monkeypatch) -> None:
    """Regression: pre-fix, PLACED (10008) was treated as REJECTED. After
    the fix, it must be SUBMITTED with ok=True so the strategy sees its
    pending order survive."""
    fake = _FakeMT5(TRADE_RETCODE_PLACED, volume=0.0, price=1.1050)
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus, fake)
    try:
        await broker.connect()
        result = await broker.place_order(_stop_req())
        assert result.ok is True, "PLACED must be success"
        assert result.status == OrderStatus.SUBMITTED
        assert result.order is not None
        assert result.order.status == OrderStatus.SUBMITTED
        # Pending PLACED has no fill yet — volume=0, fill_price=None, filled_at=None
        assert result.order.filled_volume == 0.0
        assert result.order.fill_price is None
        assert result.order.filled_at is None
        # ticket must still be set (the broker assigned one)
        assert result.ticket == 42
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_pending_placed_does_not_retry(monkeypatch) -> None:
    """Regression: PLACED must break out of the retry loop on the first
    call. Pre-fix, only DONE broke out — so PLACED would retry until
    exhaustion and then get rejected."""
    fake = _FakeMT5(TRADE_RETCODE_PLACED, volume=0.0, price=1.1050)
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus, fake)
    try:
        await broker.connect()
        await broker.place_order(_stop_req())
        # Exactly ONE send call — no retries on a successful PLACED
        assert fake.send_calls == 1
    finally:
        await broker.disconnect()
        await bus.close()


# --- DONE_PARTIAL retcode → PARTIALLY_FILLED -------------------------------


@pytest.mark.asyncio
async def test_done_partial_maps_to_partially_filled(monkeypatch) -> None:
    """Regression: partial fills (e.g. 0.05 of a 0.1 lot order at the
    venue) must surface as PARTIALLY_FILLED, not REJECTED."""
    fake = _FakeMT5(TRADE_RETCODE_DONE_PARTIAL, volume=0.05, price=1.1001)
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus, fake)
    try:
        await broker.connect()
        result = await broker.place_order(_market_req())
        assert result.ok is True
        assert result.status == OrderStatus.PARTIALLY_FILLED
        assert result.order is not None
        assert result.order.status == OrderStatus.PARTIALLY_FILLED
        # Partial fill: filled_volume reflects what actually filled
        assert result.order.filled_volume == pytest.approx(0.05)
        assert result.order.fill_price == pytest.approx(1.1001)
        assert result.order.filled_at is not None
    finally:
        await broker.disconnect()
        await bus.close()


# --- Permanent failures still reject ---------------------------------------


@pytest.mark.asyncio
async def test_invalid_volume_still_rejected(monkeypatch) -> None:
    """Sanity: permanent errors continue to map to REJECTED unchanged."""
    fake = _FakeMT5(TRADE_RETCODE_INVALID_VOLUME)
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus, fake)
    try:
        await broker.connect()
        result = await broker.place_order(_market_req())
        assert result.ok is False
        assert result.status == OrderStatus.REJECTED
        assert result.order is None
        assert result.raw_code == TRADE_RETCODE_INVALID_VOLUME
    finally:
        await broker.disconnect()
        await bus.close()
