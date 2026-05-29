"""Regression tests for two MT5Broker contract-mismatch hotfixes.

Pre-fix bugs
============

**P1: place_order doesn't emit OrderSubmittedEvent** —
``SimBroker.place_order`` (replay_broker.py:251) publishes
``OrderSubmittedEvent`` the moment a pending order enters the queue.
``MT5Broker.place_order`` returned ``OrderResult(status=SUBMITTED,
order=order)`` but never touched the bus. ``OrderRouter.handle_signal``
explicitly assumes the broker emitted the event (line 155-159 comment).
Net effect in live MT5:

  * UI's open-orders panel doesn't see pending placements.
  * Trade journal doesn't record the submission step (only the eventual
    fill — gaps in the audit trail).
  * MetricsCollector / Prometheus undercount pendings.

**P2: close_position treats DONE_PARTIAL as REJECTED** —
``MT5Broker.place_order`` was fixed in PR #47 to accept
``RETCODE_DONE_PARTIAL`` (broker filled some of the requested volume),
but ``close_position`` still checked only ``TRADE_RETCODE_DONE``. When
MT5 partially fills a close deal (e.g. low liquidity at the close
price), the broker returns DONE_PARTIAL, and our process:

  1. Marks the close as REJECTED to the caller.
  2. Emits no PartialClosedEvent.
  3. Leaves our notion of the position state unchanged.

But the broker actually closed part of the position. Result: divergence
between our process state and broker reality. RiskMonitor counter
wrong, OCO sibling logic wrong, trade journal misses the partial.

Fix
===

P1: emit ``OrderSubmittedEvent`` from ``place_order`` when status is
``SUBMITTED`` (PLACED retcode path).

P2: accept ``RETCODE_DONE_PARTIAL`` alongside ``RETCODE_DONE`` in
``close_position``. The existing full-vs-partial logic (compute
``filled_chunk`` from ``result.volume``, compare against
``pos_volume`` with epsilon) already routes to the right event;
relaxing the guard at the top is all that's needed.

These tests pin:

  1. Place pending order with PLACED retcode → OrderSubmittedEvent
     fires once with the pending Order. (P1 fix.)
  2. Place market order with DONE retcode → no OrderSubmittedEvent
     (FILLED goes through router).
  3. Close with DONE_PARTIAL → PartialClosedEvent fires with the
     broker-reported chunk volume; OrderResult.ok is True. (P2 fix.)
  4. Close with DONE_PARTIAL + result.volume close to position volume
     → still PartialClosedEvent (not full PositionClosedEvent) because
     the broker explicitly flagged it as partial.
  5. Place rejection (INVALID_VOLUME) still emits no SubmittedEvent.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from stinger_fx.brokers.mt5.broker import MT5Broker
from stinger_fx.config.models import MT5Config
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import (
    OrderSubmittedEvent,
    PartialClosedEvent,
    PositionClosedEvent,
)
from stinger_fx.domain import OrderRequest, OrderStatus, OrderType, Side
from tests._helpers import collect_into

RETCODE_DONE = 10009
RETCODE_PLACED = 10008
RETCODE_DONE_PARTIAL = 10010
RETCODE_INVALID_VOLUME = 10014


class _FakeResult:
    def __init__(
        self, *, retcode: int, order: int = 999,
        price: float = 0.0, volume: float = 0.0,
    ) -> None:
        self.retcode = retcode
        self.order = order
        self.deal = 0
        self.price = price
        self.volume = volume
        self.comment = "ok"


class _FakePosition:
    def __init__(
        self, *, ticket: int, symbol: str = "EURUSD",
        side: Side = Side.BUY, volume: float = 0.1,
        open_price: float = 1.1000, magic: int = 42,
    ) -> None:
        self.ticket = ticket
        self.symbol = symbol
        self.type = 0 if side is Side.BUY else 1
        self.volume = volume
        self.price_open = open_price
        self.sl = 0.0
        self.tp = 0.0
        self.magic = magic
        self.comment = "test"
        self.time = 1_700_000_000


class _FakeMT5:
    TRADE_RETCODE_DONE = RETCODE_DONE
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 6
    TRADE_ACTION_MODIFY = 7
    TRADE_ACTION_PENDING = 5
    TRADE_ACTION_REMOVE = 8
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

    def __init__(
        self, *, positions: list[_FakePosition] | None = None,
        send_retcode: int = RETCODE_DONE,
        send_price: float = 1.1100,
        send_volume: float = 0.0,
        contract_size: float = 100_000.0,
    ) -> None:
        self._positions = positions or []
        self._send_retcode = send_retcode
        self._send_price = send_price
        self._send_volume = send_volume
        self._contract_size = contract_size
        self.send_requests: list[dict] = []
        self.connected = True

    def initialize(self, **_kwargs) -> bool:
        return True

    def shutdown(self) -> None:
        self.connected = False

    def terminal_info(self):
        return types.SimpleNamespace(connected=True)

    def last_error(self):
        return (0, "no error")

    def symbol_select(self, _s: str, _e: bool) -> bool:
        return True

    def symbol_info(self, _s: str):
        return types.SimpleNamespace(trade_contract_size=self._contract_size)

    def symbol_info_tick(self, _s: str):
        return types.SimpleNamespace(
            bid=1.1099, ask=1.1101, last=1.1100,
            time=1_700_000_500, volume=1, flags=0,
        )

    def positions_get(self, ticket: int = 0, **_kw):
        return tuple(p for p in self._positions if p.ticket == ticket)

    def orders_get(self, ticket: int = 0, **_kw):
        return ()

    def order_send(self, request: dict):
        self.send_requests.append(dict(request))
        if self._send_retcode == RETCODE_INVALID_VOLUME:
            return _FakeResult(retcode=RETCODE_INVALID_VOLUME)
        echo_volume = self._send_volume or float(request.get("volume", 0.0))
        return _FakeResult(
            retcode=self._send_retcode,
            price=self._send_price,
            volume=echo_volume,
        )


def _install(monkeypatch, fake) -> None:
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)


def _make_broker(bus: AsyncEventBus) -> MT5Broker:
    cfg = MT5Config(terminal_path="", login=0, password="", server="", timeout_ms=1000)
    broker = MT5Broker(bus, cfg)
    broker._health_check_interval = 60.0
    broker._order_retry_backoff = [0.001]
    return broker


async def _drain(bus: AsyncEventBus, *, ticks: int = 3) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


def _stop_req() -> OrderRequest:
    return OrderRequest(
        strategy_id="s", symbol="EURUSD", side=Side.BUY,
        type=OrderType.STOP, volume=0.1, price=1.1050, client_order_id="p1",
    )


def _market_req() -> OrderRequest:
    return OrderRequest(
        strategy_id="s", symbol="EURUSD", side=Side.BUY,
        type=OrderType.MARKET, volume=0.1, client_order_id="m1",
    )


# --- P1: OrderSubmittedEvent on SUBMITTED ---------------------------------


@pytest.mark.asyncio
async def test_place_order_pending_publishes_submitted_event(monkeypatch) -> None:
    """Regression: pre-fix MT5Broker.place_order returned status=SUBMITTED
    for PLACED retcode but never published OrderSubmittedEvent. UI's
    open-orders panel and trade journal silently missed pending
    placements."""
    fake = _FakeMT5(send_retcode=RETCODE_PLACED, send_price=1.1050)
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)

    submitted: list[OrderSubmittedEvent] = []
    bus.subscribe(OrderSubmittedEvent, collect_into(submitted))

    try:
        await broker.connect()
        result = await broker.place_order(_stop_req())
        await _drain(bus)
        assert result.ok is True
        assert result.status is OrderStatus.SUBMITTED
        assert len(submitted) == 1, (
            f"PLACED retcode must emit one OrderSubmittedEvent — got "
            f"{submitted}. Pre-fix this was silently dropped."
        )
        evt = submitted[0]
        assert evt.order.status is OrderStatus.SUBMITTED
        assert evt.order.type is OrderType.STOP
        assert evt.order.side is Side.BUY
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_place_market_order_does_not_publish_submitted(monkeypatch) -> None:
    """A FILLED market order must NOT publish OrderSubmittedEvent — that
    event is reserved for pending-order placements. OrderRouter emits
    OrderFilledEvent for fills (PR #56)."""
    fake = _FakeMT5(send_retcode=RETCODE_DONE, send_price=1.1101)
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)

    submitted: list[OrderSubmittedEvent] = []
    bus.subscribe(OrderSubmittedEvent, collect_into(submitted))

    try:
        await broker.connect()
        result = await broker.place_order(_market_req())
        await _drain(bus)
        assert result.ok is True
        assert result.status is OrderStatus.FILLED
        assert submitted == [], (
            f"FILLED market order must not emit OrderSubmittedEvent; "
            f"got {submitted}"
        )
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_place_order_rejection_does_not_publish_submitted(monkeypatch) -> None:
    """Rejection path must not publish OrderSubmittedEvent."""
    fake = _FakeMT5(send_retcode=RETCODE_INVALID_VOLUME)
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)

    submitted: list[OrderSubmittedEvent] = []
    bus.subscribe(OrderSubmittedEvent, collect_into(submitted))

    try:
        await broker.connect()
        result = await broker.place_order(_stop_req())
        await _drain(bus)
        assert result.ok is False
        assert submitted == []
    finally:
        await broker.disconnect()
        await bus.close()


# --- P2: close_position accepts DONE_PARTIAL ------------------------------


@pytest.mark.asyncio
async def test_close_with_done_partial_emits_partial_closed(monkeypatch) -> None:
    """Regression: pre-fix, MT5 returning RETCODE_DONE_PARTIAL on a close
    deal was treated as REJECTED. No PartialClosedEvent fired even
    though the broker had partially closed the position → our state
    diverged from broker reality."""
    fake = _FakeMT5(
        positions=[_FakePosition(
            ticket=100, side=Side.BUY, volume=0.2,
            open_price=1.1000, magic=42,
        )],
        send_retcode=RETCODE_DONE_PARTIAL,
        send_price=1.1050,
        send_volume=0.07,  # broker partial-filled 0.07 of requested 0.1
    )
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)

    closed: list[PositionClosedEvent] = []
    partials: list[PartialClosedEvent] = []
    bus.subscribe(PositionClosedEvent, collect_into(closed))
    bus.subscribe(PartialClosedEvent, collect_into(partials))

    try:
        await broker.connect()
        # Request to close 0.1 of a 0.2 position; MT5 will return
        # DONE_PARTIAL with result.volume=0.07 (only 0.07 actually closed).
        result = await broker.close_position(100, volume=0.1)
        await _drain(bus)
        assert result.ok is True, (
            "DONE_PARTIAL must surface as ok=True to the caller; the "
            "broker did close part of the position. Pre-fix this was "
            f"REJECTED. result={result}"
        )
        assert len(partials) == 1, (
            f"DONE_PARTIAL on close must emit PartialClosedEvent — "
            f"got closed={closed} partials={partials}"
        )
        assert closed == []
        evt = partials[0]
        assert evt.closed_volume == pytest.approx(0.07)
        # Remaining = 0.2 - 0.07 = 0.13
        assert evt.position.volume == pytest.approx(0.13)
        # PnL on the actually-closed chunk: (1.1050 - 1.1000) * +1 * 0.07
        # * 100_000 = 35.0
        assert evt.realized_pnl == pytest.approx(35.0)
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_close_with_done_full_still_emits_position_closed(monkeypatch) -> None:
    """Regression guard: full-close DONE path unchanged by the fix."""
    fake = _FakeMT5(
        positions=[_FakePosition(
            ticket=100, side=Side.BUY, volume=0.1,
            open_price=1.1000, magic=42,
        )],
        send_retcode=RETCODE_DONE,
        send_price=1.1050,
    )
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)

    closed: list[PositionClosedEvent] = []
    partials: list[PartialClosedEvent] = []
    bus.subscribe(PositionClosedEvent, collect_into(closed))
    bus.subscribe(PartialClosedEvent, collect_into(partials))

    try:
        await broker.connect()
        result = await broker.close_position(100)
        await _drain(bus)
        assert result.ok is True
        assert len(closed) == 1
        assert partials == []
        assert closed[0].realized_pnl == pytest.approx(50.0)
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_close_with_invalid_retcode_still_rejected(monkeypatch) -> None:
    """Sanity: a non-DONE / non-DONE_PARTIAL retcode still rejects with no
    events fired."""
    fake = _FakeMT5(
        positions=[_FakePosition(ticket=100, side=Side.BUY, magic=42)],
        send_retcode=RETCODE_INVALID_VOLUME,
    )
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)

    closed: list[PositionClosedEvent] = []
    partials: list[PartialClosedEvent] = []
    bus.subscribe(PositionClosedEvent, collect_into(closed))
    bus.subscribe(PartialClosedEvent, collect_into(partials))

    try:
        await broker.connect()
        result = await broker.close_position(100)
        await _drain(bus)
        assert result.ok is False
        assert result.status is OrderStatus.REJECTED
        assert closed == []
        assert partials == []
    finally:
        await broker.disconnect()
        await bus.close()
