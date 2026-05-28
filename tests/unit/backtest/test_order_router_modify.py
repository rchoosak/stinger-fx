"""OrderRouter — modify_order / partial_close routing + ownership check."""

from __future__ import annotations

import asyncio

import pytest

from stinger_fx.backtest import SimBroker
from stinger_fx.backtest.order_router import OrderRouter
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import (
    ModifyOrderRequestEvent,
    OrderModifiedEvent,
    PartialClosedEvent,
    PartialCloseRequestEvent,
)
from stinger_fx.domain import OrderRequest, OrderType, Side
from tests._helpers import collect_into


async def _open_position(
    broker: SimBroker, *, strategy_id: str, magic: int, volume: float = 0.10
) -> int:
    """Place a market BUY and return the fill ticket."""
    broker.set_market("EURUSD", 1.10)
    res = await broker.place_order(
        OrderRequest(
            strategy_id=strategy_id,
            symbol="EURUSD",
            side=Side.BUY,
            type=OrderType.MARKET,
            volume=volume,
            sl=1.09,
            tp=1.11,
            magic=magic,
            client_order_id="t1",
        )
    )
    assert res.ok and res.ticket is not None
    return res.ticket


async def _drain(bus: AsyncEventBus, *, ticks: int = 5) -> None:
    """Yield enough times for queued events to dispatch."""
    for _ in range(ticks):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_modify_routes_to_broker_and_publishes_event() -> None:
    bus = AsyncEventBus()
    broker = SimBroker(bus, initial_balance=10_000.0)
    router = OrderRouter(bus, broker, strategy_magic={"strat_a": 111})
    await router.attach()

    modified: list[OrderModifiedEvent] = []
    sub = bus.subscribe(OrderModifiedEvent, collect_into(modified))

    try:
        ticket = await _open_position(broker, strategy_id="strat_a", magic=111)
        # Move SL up to 1.095
        await bus.publish(
            ModifyOrderRequestEvent(
                strategy_id="strat_a",
                ticket=ticket,
                sl=1.095,
                reason="trail",
            )
        )
        await _drain(bus)
        assert len(modified) == 1, f"expected 1 OrderModifiedEvent, got {len(modified)}"
        # Broker state reflects the new SL
        positions = await broker.get_positions()
        assert positions[0].sl == pytest.approx(1.095)
        # And TP is preserved
        assert positions[0].tp == pytest.approx(1.11)
    finally:
        await sub.unsubscribe()
        await router.detach()
        await bus.close()


@pytest.mark.asyncio
async def test_modify_refuses_cross_strategy_ticket() -> None:
    """strat_b mustn't be able to move SL on strat_a's position."""
    bus = AsyncEventBus()
    broker = SimBroker(bus, initial_balance=10_000.0)
    router = OrderRouter(
        bus, broker,
        strategy_magic={"strat_a": 111, "strat_b": 222},
    )
    await router.attach()

    modified: list[OrderModifiedEvent] = []
    sub = bus.subscribe(OrderModifiedEvent, collect_into(modified))

    try:
        ticket = await _open_position(broker, strategy_id="strat_a", magic=111)
        # strat_b tries to modify strat_a's position
        await bus.publish(
            ModifyOrderRequestEvent(
                strategy_id="strat_b",
                ticket=ticket,
                sl=1.095,
            )
        )
        await _drain(bus)
        assert modified == [], "cross-strategy modify must be refused"
        # SL unchanged
        positions = await broker.get_positions()
        assert positions[0].sl == pytest.approx(1.09)
    finally:
        await sub.unsubscribe()
        await router.detach()
        await bus.close()


@pytest.mark.asyncio
async def test_partial_close_shrinks_volume_and_emits_event() -> None:
    bus = AsyncEventBus()
    broker = SimBroker(bus, initial_balance=10_000.0)
    router = OrderRouter(bus, broker, strategy_magic={"strat_a": 111})
    await router.attach()

    partial_events: list[PartialClosedEvent] = []
    sub = bus.subscribe(
        PartialClosedEvent,
        collect_into(partial_events),
    )

    try:
        ticket = await _open_position(broker, strategy_id="strat_a", magic=111, volume=0.10)
        # Move the market up before the partial close so the closed chunk
        # books a profit (proves the P&L math runs).
        broker.set_market("EURUSD", 1.105)
        await bus.publish(
            PartialCloseRequestEvent(
                strategy_id="strat_a",
                ticket=ticket,
                volume=0.04,
                reason="take_some",
            )
        )
        await _drain(bus)

        positions = await broker.get_positions()
        assert len(positions) == 1
        assert positions[0].volume == pytest.approx(0.06)

        assert len(partial_events) == 1
        evt = partial_events[0]
        assert evt.closed_volume == pytest.approx(0.04)
        assert evt.position.volume == pytest.approx(0.06)
        # Closed chunk was 0.04 lot × (1.105 - 1.10) × 100_000 = $20
        assert evt.realized_pnl == pytest.approx(20.0, abs=0.5)
    finally:
        await sub.unsubscribe()
        await router.detach()
        await bus.close()


@pytest.mark.asyncio
async def test_partial_close_refuses_oversize_request() -> None:
    """Asking to close more than the position holds is a no-op so callers
    don't accidentally close everything via this path."""
    bus = AsyncEventBus()
    broker = SimBroker(bus, initial_balance=10_000.0)
    router = OrderRouter(bus, broker, strategy_magic={"strat_a": 111})
    await router.attach()

    try:
        ticket = await _open_position(broker, strategy_id="strat_a", magic=111, volume=0.10)
        await bus.publish(
            PartialCloseRequestEvent(
                strategy_id="strat_a",
                ticket=ticket,
                volume=0.20,  # > 0.10 — refuse
            )
        )
        await _drain(bus)
        # Position remains intact at its original volume
        positions = await broker.get_positions()
        assert len(positions) == 1
        assert positions[0].volume == pytest.approx(0.10)
    finally:
        await router.detach()
        await bus.close()


@pytest.mark.asyncio
async def test_partial_close_refuses_cross_strategy_ticket() -> None:
    bus = AsyncEventBus()
    broker = SimBroker(bus, initial_balance=10_000.0)
    router = OrderRouter(
        bus, broker,
        strategy_magic={"strat_a": 111, "strat_b": 222},
    )
    await router.attach()

    try:
        ticket = await _open_position(broker, strategy_id="strat_a", magic=111, volume=0.10)
        await bus.publish(
            PartialCloseRequestEvent(
                strategy_id="strat_b",
                ticket=ticket,
                volume=0.04,
            )
        )
        await _drain(bus)
        positions = await broker.get_positions()
        assert len(positions) == 1
        assert positions[0].volume == pytest.approx(0.10)
    finally:
        await router.detach()
        await bus.close()
