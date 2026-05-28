"""OrderRouter — pending signals route to SimBroker without emitting
OrderFilledEvent (the broker emits OrderSubmittedEvent itself).

Phase 6.2.B verifies the router branches correctly:
  * MARKET signal → OrderFilledEvent (existing path)
  * Pending signal → OrderSubmittedEvent (from broker), no OrderFilledEvent
                     until later check_pending() triggers it
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from stinger_fx.backtest.order_router import OrderRouter
from stinger_fx.backtest.replay_broker import SimBroker
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import (
    OrderFilledEvent,
    OrderRejectedEvent,
    OrderSubmittedEvent,
    SignalEvent,
)
from stinger_fx.domain import OrderType, Side
from stinger_fx.domain.signals import Signal, SignalStrength
from tests._helpers import collect_into


async def _drain(bus: AsyncEventBus, *, ticks: int = 5) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_market_signal_emits_filled_event() -> None:
    """MARKET signal goes straight through → OrderFilledEvent."""
    bus = AsyncEventBus()
    broker = SimBroker(bus, initial_balance=10_000)
    broker.set_market_tick("EURUSD", 1.10, 1.1002)

    fills: list[OrderFilledEvent] = []
    submits: list[OrderSubmittedEvent] = []
    sub_f = bus.subscribe(OrderFilledEvent, collect_into(fills))
    sub_s = bus.subscribe(OrderSubmittedEvent, collect_into(submits))

    router = OrderRouter(bus, broker, strategy_magic={"s1": 1})
    await router.attach()

    try:
        await bus.publish(SignalEvent(signal=Signal(
            strategy_id="s1",
            time=datetime(2024, 1, 1, tzinfo=UTC),
            symbol="EURUSD",
            side=Side.BUY,
            strength=SignalStrength.NORMAL,
            suggested_volume=0.1,
            order_type=OrderType.MARKET,
        )))
        await _drain(bus)

        assert len(fills) == 1
        assert submits == []
    finally:
        await sub_f.unsubscribe()
        await sub_s.unsubscribe()
        await router.detach()
        await bus.close()


@pytest.mark.asyncio
async def test_pending_signal_emits_only_submitted() -> None:
    """Pending signal → broker parks it, emits OrderSubmittedEvent.
    Router must NOT emit OrderFilledEvent (the order isn't filled yet)."""
    bus = AsyncEventBus()
    broker = SimBroker(bus, initial_balance=10_000)
    broker.set_market_tick("EURUSD", 1.10, 1.1002)

    fills: list[OrderFilledEvent] = []
    submits: list[OrderSubmittedEvent] = []
    sub_f = bus.subscribe(OrderFilledEvent, collect_into(fills))
    sub_s = bus.subscribe(OrderSubmittedEvent, collect_into(submits))

    router = OrderRouter(bus, broker, strategy_magic={"s1": 1})
    await router.attach()

    try:
        await bus.publish(SignalEvent(signal=Signal(
            strategy_id="s1",
            time=datetime(2024, 1, 1, tzinfo=UTC),
            symbol="EURUSD",
            side=Side.BUY,
            strength=SignalStrength.NORMAL,
            suggested_volume=0.1,
            order_type=OrderType.STOP,
            suggested_price=1.10,  # buy stop at current ask — triggers below
        )))
        await _drain(bus)

        # Submitted, NOT filled
        assert len(submits) == 1
        # No fill yet — pending sits until tick crosses
        assert fills == []
        # Pending exists in broker
        opens = await broker.get_open_orders()
        assert len(opens) == 1
        assert opens[0].type == OrderType.STOP
    finally:
        await sub_f.unsubscribe()
        await sub_s.unsubscribe()
        await router.detach()
        await bus.close()


@pytest.mark.asyncio
async def test_pending_signal_triggers_filled_on_tick_cross() -> None:
    """After the router submits a pending, check_pending should fire it
    when the price crosses — and that's when OrderFilledEvent reaches consumers."""
    bus = AsyncEventBus()
    broker = SimBroker(bus, initial_balance=10_000)
    broker.set_market_tick("EURUSD", 1.0998, 1.0999)

    fills: list[OrderFilledEvent] = []
    sub = bus.subscribe(OrderFilledEvent, collect_into(fills))

    router = OrderRouter(bus, broker, strategy_magic={"s1": 1})
    await router.attach()

    try:
        # Submit BUY_STOP at 1.10
        await bus.publish(SignalEvent(signal=Signal(
            strategy_id="s1",
            time=datetime(2024, 1, 1, tzinfo=UTC),
            symbol="EURUSD",
            side=Side.BUY,
            strength=SignalStrength.NORMAL,
            suggested_volume=0.1,
            order_type=OrderType.STOP,
            suggested_price=1.10,
        )))
        await _drain(bus)
        assert fills == []  # not yet

        # Price crosses — broker triggers
        await broker.check_pending("EURUSD", 1.0999, 1.10)
        await _drain(bus)
        assert len(fills) == 1
        # And the position now exists
        positions = await broker.get_positions()
        assert len(positions) == 1
    finally:
        await sub.unsubscribe()
        await router.detach()
        await bus.close()


@pytest.mark.asyncio
async def test_pending_without_price_rejected_at_broker() -> None:
    """Pending signal with no suggested_price → broker rejects."""
    bus = AsyncEventBus()
    broker = SimBroker(bus, initial_balance=10_000)
    broker.set_market_tick("EURUSD", 1.10, 1.1002)

    rejects: list[OrderRejectedEvent] = []
    sub = bus.subscribe(OrderRejectedEvent, collect_into(rejects))

    router = OrderRouter(bus, broker, strategy_magic={"s1": 1})
    await router.attach()

    try:
        await bus.publish(SignalEvent(signal=Signal(
            strategy_id="s1",
            time=datetime(2024, 1, 1, tzinfo=UTC),
            symbol="EURUSD",
            side=Side.BUY,
            strength=SignalStrength.NORMAL,
            suggested_volume=0.1,
            order_type=OrderType.STOP,
            suggested_price=None,  # missing!
        )))
        await _drain(bus)

        assert len(rejects) == 1
        assert "requires a price" in rejects[0].reason
    finally:
        await sub.unsubscribe()
        await router.detach()
        await bus.close()
