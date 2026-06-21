"""Regression tests for the MT5Broker event-emission hotfix.

Pre-fix bug
===========

``MT5Broker.close_position`` and ``MT5Broker.cancel_order`` returned
``OrderResult(ok=True)`` on broker confirmation but never called
``self.bus.publish(...)``. ``SimBroker`` emits the same events
(``PositionClosedEvent``, ``PartialClosedEvent``, ``OrderCancelledEvent``)
so the *entire* backtest test suite passed against SimBroker without
exercising the live-mode contract.

The router's comments at ``order_router.py`` lines 304-306 and 327-329
claim "the broker emits / will fall back here" but inspection shows the
router has no fallback publishes either. Net effect in live trading:

  * ``RiskMonitor._on_closed`` never fires → per-strategy open-position
    counters drift up forever and strategies hit cap silently.
  * ``OCOGroupManager`` never sees cancels/closes → sibling legs orphan.
  * ``MetricsCollector`` undercounts closes.
  * Trade journal misses position-closed rows.

Fix: emit the SimBroker contract directly from MT5Broker.close_position
and MT5Broker.cancel_order. The router stays simple. SimBroker is
untouched so backtest tests keep passing.

These tests pin that:

  1. Full close emits ``PositionClosedEvent`` once with a non-None
     position and a finite realized_pnl computed from the broker's
     reported fill price.
  2. Partial close emits ``PartialClosedEvent`` with the remaining-leg
     position and the chunk volume, not ``PositionClosedEvent``.
  3. cancel_order on a real pending emits ``OrderCancelledEvent`` with
     the order shape mypy-narrowable for downstream OCO subscribers.
  4. Failed broker actions do NOT emit (rejected close, unknown ticket).
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
    OrderCancelledEvent,
    PartialClosedEvent,
    PositionClosedEvent,
)
from stinger_fx.domain import OrderStatus, OrderType, Side
from tests._helpers import collect_into

TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_INVALID_VOLUME = 10014


class _FakeResult:
    def __init__(
        self, *, retcode: int = TRADE_RETCODE_DONE,
        price: float = 0.0, volume: float = 0.0, deal: int = 0,
    ) -> None:
        self.retcode = retcode
        self.order = 999
        self.deal = deal
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
        # MT5 raw type: 0 for BUY, 1 for SELL (matches _FakeMT5 constants)
        self.type = 0 if side is Side.BUY else 1
        self.volume = volume
        self.price_open = open_price
        self.sl = 0.0
        self.tp = 0.0
        self.magic = magic
        self.comment = "test"
        self.time = 1_700_000_000


class _FakePending:
    def __init__(
        self, *, ticket: int, symbol: str = "EURUSD",
        order_type: int = 4,  # BUY_STOP
        volume: float = 0.1, price: float = 1.1050,
        magic: int = 42,
    ) -> None:
        self.ticket = ticket
        self.symbol = symbol
        self.type = order_type
        self.volume_current = volume
        self.price_open = price
        self.magic = magic
        self.comment = "test"


class _FakeMT5:
    """Minimal MT5 stand-in driven by per-test scriptable state."""

    TRADE_RETCODE_DONE = TRADE_RETCODE_DONE
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
        pending_orders: list[_FakePending] | None = None,
        send_retcode: int = TRADE_RETCODE_DONE,
        send_price: float = 1.1100,
        send_volume: float = 0.0,  # 0 means "broker echoes request volume"
        contract_size: float = 100_000.0,
        deal: object | None = None,
    ) -> None:
        self._positions = positions or []
        self._pending = pending_orders or []
        self._send_retcode = send_retcode
        self._send_price = send_price
        self._send_volume = send_volume
        self._contract_size = contract_size
        self._deal = deal
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
        return tuple(o for o in self._pending if o.ticket == ticket)

    def order_send(self, request: dict):
        self.send_requests.append(dict(request))
        if self._send_retcode != TRADE_RETCODE_DONE:
            return _FakeResult(retcode=self._send_retcode)
        # On a CLOSE deal, echo the close volume if test didn't override.
        echo_volume = self._send_volume or float(request.get("volume", 0.0))
        return _FakeResult(
            retcode=TRADE_RETCODE_DONE,
            price=self._send_price,
            volume=echo_volume,
            deal=777 if self._deal is not None else 0,
        )

    def history_deals_get(self, *, ticket: int):
        if ticket == 777 and self._deal is not None:
            return (self._deal,)
        return ()


def _install(monkeypatch, fake) -> None:
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)


def _make_broker(bus: AsyncEventBus) -> MT5Broker:
    cfg = MT5Config(terminal_path="", login=0, password="", server="", timeout_ms=1000)
    broker = MT5Broker(bus, cfg)
    broker._health_check_interval = 60.0
    return broker


async def _drain(bus: AsyncEventBus, *, ticks: int = 3) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


# --- 1. Full close emits PositionClosedEvent --------------------------------


@pytest.mark.asyncio
async def test_close_position_full_publishes_position_closed_event(monkeypatch) -> None:
    """Regression: pre-fix, MT5Broker.close_position returned ok=True but
    never touched the bus. RiskMonitor / OCO / trade journal silently
    missed the close."""
    fake = _FakeMT5(
        positions=[_FakePosition(
            ticket=100, side=Side.BUY, volume=0.1, open_price=1.1000, magic=42,
        )],
        send_price=1.1050,  # close 50 pips above entry → profit
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
        assert result.status == OrderStatus.FILLED
        assert len(closed) == 1, (
            f"expected one PositionClosedEvent, got closed={closed} partials={partials}"
        )
        assert len(partials) == 0
        evt = closed[0]
        assert evt.position.ticket == 100
        assert evt.position.symbol == "EURUSD"
        assert evt.position.side is Side.BUY
        # PnL: (1.1050 - 1.1000) * +1 * 0.1 * 100_000 = 50.0
        assert evt.realized_pnl == pytest.approx(50.0)
    finally:
        await broker.disconnect()
        await bus.close()


# --- 2. Partial close emits PartialClosedEvent (not PositionClosed) --------


@pytest.mark.asyncio
async def test_close_position_partial_publishes_partial_closed_event(monkeypatch) -> None:
    """Partial close (volume < pos.volume) must emit PartialClosedEvent
    with the *remaining* leg, not PositionClosedEvent."""
    fake = _FakeMT5(
        positions=[_FakePosition(
            ticket=100, side=Side.BUY, volume=0.3, open_price=1.1000, magic=42,
        )],
        send_price=1.1020,
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
        # Close 0.1 out of 0.3 → 0.2 remaining
        result = await broker.close_position(100, volume=0.1)
        await _drain(bus)
        assert result.ok is True
        assert len(partials) == 1, (
            f"expected PartialClosedEvent, got closed={closed} partials={partials}"
        )
        assert len(closed) == 0, "PositionClosedEvent must NOT fire on partial close"
        evt = partials[0]
        assert evt.closed_volume == pytest.approx(0.1)
        assert evt.position.volume == pytest.approx(0.2)  # remaining
        # PnL on the closed chunk only: (1.1020 - 1.1000) * +1 * 0.1 * 100_000 = 20.0
        assert evt.realized_pnl == pytest.approx(20.0)
    finally:
        await broker.disconnect()
        await bus.close()


# --- 3. SELL-side P&L direction is correct ---------------------------------


@pytest.mark.asyncio
async def test_close_short_position_pnl_uses_correct_sign(monkeypatch) -> None:
    """SELL closes: pnl = (open - close) * volume * contract_size. Pre-fix
    no event was emitted at all; with the fix we must also have the sign
    correct."""
    fake = _FakeMT5(
        positions=[_FakePosition(
            ticket=200, side=Side.SELL, volume=0.1, open_price=1.1100, magic=42,
        )],
        send_price=1.1050,  # close 50 pips BELOW entry → profit on SELL
    )
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)

    closed: list[PositionClosedEvent] = []
    bus.subscribe(PositionClosedEvent, collect_into(closed))

    try:
        await broker.connect()
        await broker.close_position(200)
        await _drain(bus)
        assert len(closed) == 1
        # (1.1050 - 1.1100) * -1 (SELL) * 0.1 * 100_000 = 50.0
        assert closed[0].realized_pnl == pytest.approx(50.0)
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_close_uses_net_account_currency_pnl_from_deal(monkeypatch) -> None:
    deal = types.SimpleNamespace(
        profit=50.0,
        commission=-3.0,
        swap=-2.0,
        fee=-1.0,
    )
    fake = _FakeMT5(
        positions=[
            _FakePosition(
                ticket=100,
                side=Side.BUY,
                volume=0.1,
                open_price=1.1000,
                magic=42,
            )
        ],
        send_price=1.1050,
        deal=deal,
    )
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)
    closed: list[PositionClosedEvent] = []
    bus.subscribe(PositionClosedEvent, collect_into(closed))

    try:
        await broker.connect()
        await broker.close_position(100)
        await _drain(bus)
        assert closed[0].realized_pnl == pytest.approx(44.0)
    finally:
        await broker.disconnect()
        await bus.close()


# --- 4. cancel_order emits OrderCancelledEvent -----------------------------


@pytest.mark.asyncio
async def test_cancel_order_publishes_order_cancelled_event(monkeypatch) -> None:
    """Regression: pre-fix, cancel_order returned ok=True but never
    published. OCOGroupManager couldn't see cancels of sibling legs →
    orphaned orders at the broker."""
    fake = _FakeMT5(
        pending_orders=[_FakePending(
            ticket=42, order_type=4,  # BUY_STOP
            volume=0.1, price=1.1050, magic=42,
        )],
    )
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)

    cancelled: list[OrderCancelledEvent] = []
    bus.subscribe(OrderCancelledEvent, collect_into(cancelled))

    try:
        await broker.connect()
        result = await broker.cancel_order(42)
        await _drain(bus)
        assert result.ok is True
        assert result.status == OrderStatus.CANCELLED
        assert len(cancelled) == 1, (
            f"expected one OrderCancelledEvent, got {cancelled}"
        )
        evt = cancelled[0]
        assert evt.order.ticket == 42
        assert evt.order.status is OrderStatus.CANCELLED
        assert evt.order.type is OrderType.STOP
        assert evt.order.side is Side.BUY
        assert evt.order.magic == 42
    finally:
        await broker.disconnect()
        await bus.close()


# --- 5. Failure paths must NOT emit ---------------------------------------


@pytest.mark.asyncio
async def test_close_position_failure_does_not_publish(monkeypatch) -> None:
    """If MT5 rejects the close (e.g. invalid volume), no event must fire."""
    fake = _FakeMT5(
        positions=[_FakePosition(ticket=100, side=Side.BUY, magic=42)],
        send_retcode=TRADE_RETCODE_INVALID_VOLUME,
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
        assert result.status == OrderStatus.REJECTED
        assert closed == []
        assert partials == []
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_close_position_unknown_ticket_does_not_publish(monkeypatch) -> None:
    """Unknown ticket → return REJECTED before any send. No bus traffic."""
    fake = _FakeMT5()  # no positions
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)

    closed: list[PositionClosedEvent] = []
    bus.subscribe(PositionClosedEvent, collect_into(closed))

    try:
        await broker.connect()
        result = await broker.close_position(9999)
        await _drain(bus)
        assert result.ok is False
        assert closed == []
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_cancel_order_failure_does_not_publish(monkeypatch) -> None:
    fake = _FakeMT5(
        pending_orders=[_FakePending(ticket=42, magic=42)],
        send_retcode=TRADE_RETCODE_INVALID_VOLUME,
    )
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)

    cancelled: list[OrderCancelledEvent] = []
    bus.subscribe(OrderCancelledEvent, collect_into(cancelled))

    try:
        await broker.connect()
        result = await broker.cancel_order(42)
        await _drain(bus)
        assert result.ok is False
        assert cancelled == []
    finally:
        await broker.disconnect()
        await bus.close()


# --- 6. Contract uniformity with SimBroker --------------------------------


@pytest.mark.asyncio
async def test_close_position_contract_matches_sim_broker(monkeypatch) -> None:
    """MT5Broker and SimBroker must publish the same event types so
    subscribers (RiskMonitor, OCOGroupManager, trade journal) get a
    uniform feed regardless of which broker is live.

    This test asserts MT5Broker emits exactly the events SimBroker does
    for an analogous full close."""
    fake = _FakeMT5(
        positions=[_FakePosition(ticket=100, side=Side.BUY, magic=42)],
        send_price=1.1050,
    )
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)

    # Subscribe the same way RiskMonitor does to verify wire-up.
    seen_event_types: list[str] = []

    async def capture_closed(evt: PositionClosedEvent) -> None:
        seen_event_types.append("position_closed")

    async def capture_partial(evt: PartialClosedEvent) -> None:
        seen_event_types.append("partial_closed")

    bus.subscribe(PositionClosedEvent, capture_closed)
    bus.subscribe(PartialClosedEvent, capture_partial)

    try:
        await broker.connect()
        await broker.close_position(100)
        await _drain(bus)
        assert seen_event_types == ["position_closed"], (
            f"MT5Broker close_position must emit exactly one PositionClosedEvent; "
            f"got {seen_event_types}"
        )
    finally:
        await broker.disconnect()
        await bus.close()
