"""ctx.close() — publishes ClosePositionRequestEvent; router routes to broker."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import pytest

from stinger_fx.backtest.order_router import OrderRouter
from stinger_fx.backtest.replay_broker import SimBroker
from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.core.events import ClosePositionRequestEvent, PositionClosedEvent
from stinger_fx.domain import Position, Side, Tick, Timeframe
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.parameters import StrategyParams
from stinger_fx.strategies.runner import derive_magic


def _make_ctx(bus: AsyncEventBus) -> StrategyContext:
    return StrategyContext(
        strategy_id="test_strat",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        params=StrategyParams(),
        clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)),
        logger=logging.getLogger("test"),
        magic=derive_magic("test_strat"),
        signal_sink=lambda s: asyncio.sleep(0),
        bus=bus,
    )


def _make_position(ticket: int, magic: int) -> Position:
    return Position(
        ticket=ticket,
        symbol="EURUSD",
        side=Side.BUY,
        volume=0.1,
        open_price=1.10,
        open_time=datetime(2024, 1, 1, tzinfo=UTC),
        sl=1.0980,
        magic=magic,
    )


async def _drain(bus: AsyncEventBus, *, ticks: int = 3) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_ctx_close_publishes_event() -> None:
    """ctx.close(ticket) must publish a ClosePositionRequestEvent on the bus."""
    bus = AsyncEventBus()
    seen: list[ClosePositionRequestEvent] = []

    async def collect(evt: ClosePositionRequestEvent) -> None:
        seen.append(evt)

    sub = bus.subscribe(ClosePositionRequestEvent, collect, name="t.close")
    ctx = _make_ctx(bus)

    try:
        await ctx.close(42, reason="test_reason")
        await _drain(bus)
        assert len(seen) == 1
        assert seen[0].ticket == 42
        assert seen[0].strategy_id == "test_strat"
        assert seen[0].reason == "test_reason"
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_ctx_close_raises_without_bus() -> None:
    """ctx.close() must raise RuntimeError when no bus was provided."""
    ctx = StrategyContext(
        strategy_id="test_strat",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        params=StrategyParams(),
        clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)),
        logger=logging.getLogger("test"),
        magic=12345,
        signal_sink=lambda s: asyncio.sleep(0),
        bus=None,
    )
    with pytest.raises(RuntimeError, match="bus"):
        await ctx.close(1)


@pytest.mark.asyncio
async def test_router_handle_close_triggers_position_closed() -> None:
    """OrderRouter.handle_close must close the position and the broker
    emits PositionClosedEvent."""
    bus = AsyncEventBus()
    magic = derive_magic("test_strat")
    broker = SimBroker(bus, initial_balance=10_000.0)
    broker._last_price["EURUSD"] = 1.1010
    # Inject a position directly into SimBroker
    pos = _make_position(ticket=7, magic=magic)
    broker._positions[7] = pos

    closed_events: list[PositionClosedEvent] = []

    async def collect_closed(evt: PositionClosedEvent) -> None:
        closed_events.append(evt)

    sub = bus.subscribe(PositionClosedEvent, collect_closed, name="t.closed")

    router = OrderRouter(
        bus,
        broker=broker,
        strategy_magic={"test_strat": magic},
    )
    await router.attach()

    try:
        evt = ClosePositionRequestEvent(strategy_id="test_strat", ticket=7, reason="test")
        await bus.publish(evt)
        await _drain(bus, ticks=5)

        assert len(closed_events) == 1
        assert closed_events[0].position.ticket == 7
        # Position should be gone from broker
        assert 7 not in broker._positions
    finally:
        await router.detach()
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_router_rejects_close_cross_strategy() -> None:
    """Close request for a ticket owned by a different strategy must be ignored."""
    bus = AsyncEventBus()
    magic_a = derive_magic("strat_a")
    magic_b = derive_magic("strat_b")
    broker = SimBroker(bus, initial_balance=10_000.0)
    broker._last_price["EURUSD"] = 1.1010
    # Ticket belongs to strat_a (magic_a)
    pos = _make_position(ticket=99, magic=magic_a)
    broker._positions[99] = pos

    closed_events: list[PositionClosedEvent] = []

    async def collect_closed(evt: PositionClosedEvent) -> None:
        closed_events.append(evt)

    sub = bus.subscribe(PositionClosedEvent, collect_closed, name="t.closed")
    router = OrderRouter(
        bus,
        broker=broker,
        strategy_magic={"strat_a": magic_a, "strat_b": magic_b},
    )
    await router.attach()

    try:
        # strat_b tries to close strat_a's ticket — should be refused
        evt = ClosePositionRequestEvent(strategy_id="strat_b", ticket=99, reason="test")
        await bus.publish(evt)
        await _drain(bus, ticks=5)

        assert closed_events == [], "cross-strategy close must be rejected"
        assert 99 in broker._positions, "position should still exist"
    finally:
        await router.detach()
        await sub.unsubscribe()
        await bus.close()
