"""Regression tests for the OrderRouter partial-fill emission hotfix.

Pre-fix bug
===========

``OrderRouter.handle_signal`` at line 152 (pre-fix) only emitted
``OrderFilledEvent`` when ``result.status == OrderStatus.FILLED``. The
three-way ladder was:

    if result.ok and ... and status == FILLED:    emit OrderFilledEvent
    elif result.ok and status == SUBMITTED:       no-op (pending queued)
    elif not result.ok:                           emit OrderRejectedEvent

After PR #47 (MT5 pending retcode hotfix), live ``MT5Broker.place_order``
returns ``ok=True`` + ``status=PARTIALLY_FILLED`` on MT5 retcode
``DONE_PARTIAL`` (broker accepted some but not all of the requested
volume). ``PARTIALLY_FILLED`` matches none of the three branches above,
so the partial fill **disappears silently**:

  * ``strategy.on_order_filled`` never fires → strategy doesn't know it
    has a (smaller) position open.
  * ``RiskMonitor._on_filled`` never increments → per-strategy counter
    drifts under-by-one for every partial fill.
  * ``MetricsCollector.orders_filled_total`` undercounts.
  * Trade journal misses the order row.
  * UI's open-positions panel out of sync with broker reality.

This is a *live-only* gap: ``SimBroker`` doesn't do partial fills on
market orders (everything's all-or-nothing in the simulator), so the
entire backtest suite passes against SimBroker without ever exercising
the live path that has the bug.

Fix: combine ``FILLED`` and ``PARTIALLY_FILLED`` into one branch — both
mean "a fill happened, emit OrderFilledEvent". Subscribers can read
``order.status`` and ``order.filled_volume`` to distinguish.

These tests pin:

  1. ``PARTIALLY_FILLED`` result now emits ``OrderFilledEvent`` with
     the partial-volume order. (THE bug fix.)
  2. ``FILLED`` continues to emit (regression preserved).
  3. ``SUBMITTED`` (pending placed) still doesn't emit fill (broker
     emitted OrderSubmittedEvent itself).
  4. Rejection still emits ``OrderRejectedEvent``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from stinger_fx.backtest.order_router import OrderRouter
from stinger_fx.brokers.base import BaseBroker
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import OrderFilledEvent, OrderRejectedEvent
from stinger_fx.domain import (
    AccountInfo,
    AccountSnapshot,
    Order,
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    Position,
    Side,
    Signal,
    SignalStrength,
    SymbolInfo,
)
from tests._helpers import collect_into


class _ScriptedBroker(BaseBroker):
    """Returns a pre-scripted OrderResult so tests can drive each branch
    of handle_signal exactly."""

    name = "scripted"

    def __init__(self, bus: AsyncEventBus, result: OrderResult) -> None:
        super().__init__(bus)
        self._result = result
        self.calls: list[OrderRequest] = []

    async def connect(self): ...
    async def disconnect(self): ...
    async def is_connected(self): return True
    async def get_account_info(self) -> AccountInfo:
        return AccountInfo(
            account_id="t", broker="t", server="t", currency="USD", leverage=100,
        )
    async def get_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            account_id="t", time=datetime.now(UTC),
            balance=10_000, equity=10_000, margin=0, free_margin=10_000,
        )
    async def get_symbol_info(self, symbol):
        return SymbolInfo(
            symbol="EURUSD", digits=5, point=0.00001, contract_size=100_000,
            volume_min=0.01, volume_max=100, volume_step=0.01,
            currency_base="EUR", currency_profit="USD", currency_margin="USD",
        )
    async def list_symbols(self): return ["EURUSD"]
    async def subscribe_ticks(self, symbol): ...
    async def subscribe_bars(self, symbol, tf): ...
    async def unsubscribe(self, symbol, tf=None): ...
    async def get_history_bars(self, *a, **kw):
        from stinger_fx.data.parquet_store import BAR_SCHEMA
        return BAR_SCHEMA.empty_table()
    async def get_history_ticks(self, *a, **kw):
        from stinger_fx.data.parquet_store import TICK_SCHEMA
        return TICK_SCHEMA.empty_table()
    async def place_order(self, req: OrderRequest) -> OrderResult:
        self.calls.append(req)
        return self._result
    async def modify_order(self, ticket, **kw): raise NotImplementedError
    async def close_position(self, ticket, volume=None): raise NotImplementedError
    async def cancel_order(self, ticket): raise NotImplementedError
    async def get_positions(self) -> list[Position]: return []
    async def get_open_orders(self) -> list[Order]: return []


def _signal() -> Signal:
    return Signal(
        strategy_id="s1", time=datetime.now(UTC),
        symbol="EURUSD", side=Side.BUY,
        strength=SignalStrength.NORMAL, suggested_volume=0.1,
        order_type=OrderType.MARKET,
    )


def _filled_order(*, filled_volume: float, status: OrderStatus) -> Order:
    return Order(
        ticket=42, strategy_id="s1", symbol="EURUSD",
        side=Side.BUY, type=OrderType.MARKET,
        volume=0.1, filled_volume=filled_volume,
        status=status,
    )


# --- 1. THE BUG FIX: PARTIALLY_FILLED emits OrderFilledEvent --------------


@pytest.mark.asyncio
async def test_partially_filled_emits_order_filled_event() -> None:
    """Regression: pre-fix this dropped the event silently.

    Live MT5 returns ``ok=True`` + ``status=PARTIALLY_FILLED`` when the
    broker fills only part of the requested volume (DONE_PARTIAL retcode).
    The router *must* emit OrderFilledEvent so RiskMonitor, strategy
    hooks, metrics, and the trade journal all see the partial execution.
    """
    bus = AsyncEventBus()
    # Scripted: filled 0.06 of 0.1 requested.
    partial_order = _filled_order(filled_volume=0.06, status=OrderStatus.PARTIALLY_FILLED)
    broker = _ScriptedBroker(bus, OrderResult(
        ok=True, ticket=42, status=OrderStatus.PARTIALLY_FILLED,
        order=partial_order,
    ))
    router = OrderRouter(bus, broker, strategy_magic={"s1": 1})
    await router.attach()

    fills: list[OrderFilledEvent] = []
    rejects: list[OrderRejectedEvent] = []
    bus.subscribe(OrderFilledEvent, collect_into(fills))
    bus.subscribe(OrderRejectedEvent, collect_into(rejects))

    try:
        await router.handle_signal(_signal())
        # Drain
        import asyncio
        for _ in range(3):
            await asyncio.sleep(0)

        assert len(fills) == 1, (
            f"PARTIALLY_FILLED must emit OrderFilledEvent — pre-fix dropped "
            f"it silently. Got fills={fills} rejects={rejects}"
        )
        assert fills[0].order.status is OrderStatus.PARTIALLY_FILLED
        assert fills[0].order.filled_volume == pytest.approx(0.06)
        assert fills[0].order.volume == pytest.approx(0.1)
        # And no false rejection.
        assert rejects == []
    finally:
        await router.detach()
        await bus.close()


# --- 2. FILLED continues to work (regression guard) ----------------------


@pytest.mark.asyncio
async def test_filled_still_emits_order_filled_event() -> None:
    """Pre-existing behaviour preserved by the fix."""
    bus = AsyncEventBus()
    full_order = _filled_order(filled_volume=0.1, status=OrderStatus.FILLED)
    broker = _ScriptedBroker(bus, OrderResult(
        ok=True, ticket=42, status=OrderStatus.FILLED, order=full_order,
    ))
    router = OrderRouter(bus, broker, strategy_magic={"s1": 1})
    await router.attach()

    fills: list[OrderFilledEvent] = []
    bus.subscribe(OrderFilledEvent, collect_into(fills))

    try:
        await router.handle_signal(_signal())
        import asyncio
        for _ in range(3):
            await asyncio.sleep(0)

        assert len(fills) == 1
        assert fills[0].order.status is OrderStatus.FILLED
        assert fills[0].order.filled_volume == pytest.approx(0.1)
    finally:
        await router.detach()
        await bus.close()


# --- 3. SUBMITTED (pending placed) still does NOT emit fill --------------


@pytest.mark.asyncio
async def test_submitted_pending_does_not_emit_fill() -> None:
    """SUBMITTED means a pending order was queued at the broker. The
    broker emitted OrderSubmittedEvent itself; OrderFilledEvent fires
    later when the pending triggers. Router must not emit on SUBMITTED."""
    bus = AsyncEventBus()
    pending = Order(
        ticket=42, strategy_id="s1", symbol="EURUSD",
        side=Side.BUY, type=OrderType.STOP,
        volume=0.1, price=1.1050,
        status=OrderStatus.SUBMITTED,
    )
    broker = _ScriptedBroker(bus, OrderResult(
        ok=True, ticket=42, status=OrderStatus.SUBMITTED, order=pending,
    ))
    router = OrderRouter(bus, broker, strategy_magic={"s1": 1})
    await router.attach()

    fills: list[OrderFilledEvent] = []
    bus.subscribe(OrderFilledEvent, collect_into(fills))

    try:
        # Use a STOP signal so suggested_price isn't required for this path.
        sig = Signal(
            strategy_id="s1", time=datetime.now(UTC),
            symbol="EURUSD", side=Side.BUY,
            strength=SignalStrength.NORMAL, suggested_volume=0.1,
            order_type=OrderType.STOP, suggested_price=1.1050,
        )
        await router.handle_signal(sig)
        import asyncio
        for _ in range(3):
            await asyncio.sleep(0)

        assert fills == [], (
            f"SUBMITTED is a pending placement — must not emit OrderFilledEvent. "
            f"Got {fills}"
        )
    finally:
        await router.detach()
        await bus.close()


# --- 4. REJECTED still emits OrderRejectedEvent --------------------------


@pytest.mark.asyncio
async def test_rejected_still_emits_order_rejected_event() -> None:
    """Pre-existing rejection-emission path preserved by the fix."""
    bus = AsyncEventBus()
    broker = _ScriptedBroker(bus, OrderResult(
        ok=False, ticket=0, status=OrderStatus.REJECTED,
        message="not enough money",
    ))
    router = OrderRouter(bus, broker, strategy_magic={"s1": 1})
    await router.attach()

    fills: list[OrderFilledEvent] = []
    rejects: list[OrderRejectedEvent] = []
    bus.subscribe(OrderFilledEvent, collect_into(fills))
    bus.subscribe(OrderRejectedEvent, collect_into(rejects))

    try:
        await router.handle_signal(_signal())
        import asyncio
        for _ in range(3):
            await asyncio.sleep(0)

        assert fills == []
        assert len(rejects) == 1
        assert rejects[0].reason == "not enough money"
    finally:
        await router.detach()
        await bus.close()
