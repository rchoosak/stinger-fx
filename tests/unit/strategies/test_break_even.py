"""BreakEvenMover — locks SL at (or just past) open price after trigger profit."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import pytest

from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.core.events import ModifyOrderRequestEvent
from stinger_fx.domain import Position, Side, Tick, Timeframe
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.managers.break_even import BreakEvenMover
from stinger_fx.strategies.parameters import StrategyParams


def _make_ctx(bus: AsyncEventBus) -> StrategyContext:
    return StrategyContext(
        strategy_id="test_strat",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        params=StrategyParams(),
        clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)),
        logger=logging.getLogger("test"),
        magic=12345,
        signal_sink=lambda s: asyncio.sleep(0),
        bus=bus,
    )


def _make_position(
    side: Side, *, open_price: float, sl: float, ticket: int = 1, symbol: str = "EURUSD"
) -> Position:
    return Position(
        ticket=ticket,
        symbol=symbol,
        side=side,
        volume=0.1,
        open_price=open_price,
        open_time=datetime(2024, 1, 1, tzinfo=UTC),
        sl=sl,
        magic=12345,
    )


async def _drain(bus: AsyncEventBus, *, ticks: int = 3) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_break_even_fires_once_on_trigger() -> None:
    """Once profit ≥ trigger_pips, SL jumps to open_price + lock_pips and the
    manager stays silent afterwards."""
    bus = AsyncEventBus()
    seen: list[ModifyOrderRequestEvent] = []

    async def collect(evt: ModifyOrderRequestEvent) -> None:
        seen.append(evt)

    sub = bus.subscribe(ModifyOrderRequestEvent, collect, name="t.modify")
    ctx = _make_ctx(bus)
    ctx.position.update([_make_position(Side.BUY, open_price=1.10, sl=1.0980)])
    mover = BreakEvenMover(ctx, trigger_pips=15, lock_pips=1)

    try:
        # +10 pips — below trigger
        await mover.on_tick(
            ctx, Tick(symbol="EURUSD", time=datetime(2024, 1, 1, tzinfo=UTC), bid=1.1010, ask=1.10102)
        )
        await _drain(bus)
        assert seen == []

        # +16 pips — clearly past trigger → fire
        await mover.on_tick(
            ctx, Tick(symbol="EURUSD", time=datetime(2024, 1, 1, tzinfo=UTC), bid=1.1016, ask=1.10162)
        )
        await _drain(bus)
        assert len(seen) == 1
        # SL = open + lock = 1.10 + 1pip = 1.1001
        assert seen[0].sl == pytest.approx(1.1001)
        assert seen[0].reason == "break_even"
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_break_even_idempotent_after_trigger() -> None:
    """More ticks past the trigger must NOT re-fire the move."""
    bus = AsyncEventBus()
    seen: list[ModifyOrderRequestEvent] = []

    async def collect(evt: ModifyOrderRequestEvent) -> None:
        seen.append(evt)

    sub = bus.subscribe(ModifyOrderRequestEvent, collect, name="t.modify")
    ctx = _make_ctx(bus)
    ctx.position.update([_make_position(Side.BUY, open_price=1.10, sl=1.0980)])
    mover = BreakEvenMover(ctx, trigger_pips=10, lock_pips=0)

    try:
        # Fire past +10pip
        await mover.on_tick(
            ctx, Tick(symbol="EURUSD", time=datetime(2024, 1, 1, tzinfo=UTC), bid=1.1012, ask=1.10122)
        )
        await _drain(bus)
        assert len(seen) == 1

        # Continue past trigger — must not re-fire
        for bid in (1.1015, 1.1020, 1.1025):
            await mover.on_tick(
                ctx, Tick(symbol="EURUSD", time=datetime(2024, 1, 1, tzinfo=UTC), bid=bid, ask=bid + 0.00002)
            )
        await _drain(bus)
        assert len(seen) == 1, f"break-even re-fired: {seen}"
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_break_even_symbol_filter() -> None:
    """A manager configured for EURUSD must ignore ticks on GBPUSD."""
    bus = AsyncEventBus()
    seen: list[ModifyOrderRequestEvent] = []

    async def collect(evt: ModifyOrderRequestEvent) -> None:
        seen.append(evt)

    sub = bus.subscribe(ModifyOrderRequestEvent, collect, name="t.modify")
    ctx = _make_ctx(bus)
    # Two positions — only the EURUSD one should be touched
    ctx.position.update(
        [
            _make_position(Side.BUY, open_price=1.10, sl=1.0980, ticket=1, symbol="EURUSD"),
            _make_position(Side.BUY, open_price=1.27, sl=1.265, ticket=2, symbol="GBPUSD"),
        ]
    )
    mover = BreakEvenMover(ctx, trigger_pips=10, symbol="EURUSD")

    try:
        # GBPUSD tick — manager ignores it
        await mover.on_tick(
            ctx, Tick(symbol="GBPUSD", time=datetime(2024, 1, 1, tzinfo=UTC), bid=1.281, ask=1.28102)
        )
        await _drain(bus)
        assert seen == []

        # EURUSD tick clearly past trigger — fires
        await mover.on_tick(
            ctx, Tick(symbol="EURUSD", time=datetime(2024, 1, 1, tzinfo=UTC), bid=1.1012, ask=1.10122)
        )
        await _drain(bus)
        assert len(seen) == 1
        assert seen[0].ticket == 1
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_break_even_sell_side() -> None:
    """Mirror behaviour for a SELL position."""
    bus = AsyncEventBus()
    seen: list[ModifyOrderRequestEvent] = []

    async def collect(evt: ModifyOrderRequestEvent) -> None:
        seen.append(evt)

    sub = bus.subscribe(ModifyOrderRequestEvent, collect, name="t.modify")
    ctx = _make_ctx(bus)
    ctx.position.update([_make_position(Side.SELL, open_price=1.10, sl=1.1020)])
    mover = BreakEvenMover(ctx, trigger_pips=15, lock_pips=1)

    try:
        # Ask falls to 1.0985 = +15 pips for a SELL @ 1.10
        await mover.on_tick(
            ctx, Tick(symbol="EURUSD", time=datetime(2024, 1, 1, tzinfo=UTC), bid=1.09848, ask=1.0985)
        )
        await _drain(bus)
        assert len(seen) == 1
        # SL = open - lock = 1.10 - 1pip = 1.0999
        assert seen[0].sl == pytest.approx(1.0999)
    finally:
        await sub.unsubscribe()
        await bus.close()
