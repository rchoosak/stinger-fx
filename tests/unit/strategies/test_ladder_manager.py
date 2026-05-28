"""LadderManager — pyramids into a position as price advances."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import pytest

from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.core.events import SignalEvent
from stinger_fx.domain import Position, Side, Tick, Timeframe
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.managers.ladder import LadderManager
from stinger_fx.strategies.parameters import StrategyParams


def _make_ctx(bus: AsyncEventBus) -> StrategyContext:

    async def sink(signal):
        await bus.publish(SignalEvent(signal=signal))

    return StrategyContext(
        strategy_id="test_strat",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        params=StrategyParams(),
        clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)),
        logger=logging.getLogger("test"),
        magic=12345,
        signal_sink=sink,
        bus=bus,
    )


def _make_buy_position(ticket: int = 1, open_price: float = 1.10) -> Position:
    return Position(
        ticket=ticket,
        symbol="EURUSD",
        side=Side.BUY,
        volume=0.1,
        open_price=open_price,
        open_time=datetime(2024, 1, 1, tzinfo=UTC),
        sl=1.098,
        magic=12345,
    )


def _make_sell_position(ticket: int = 2, open_price: float = 1.10) -> Position:
    return Position(
        ticket=ticket,
        symbol="EURUSD",
        side=Side.SELL,
        volume=0.1,
        open_price=open_price,
        open_time=datetime(2024, 1, 1, tzinfo=UTC),
        sl=1.102,
        magic=12345,
    )


async def _drain(bus: AsyncEventBus, *, ticks: int = 3) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_ladder_buy_fires_on_step() -> None:
    """BUY: each time bid rises by step_pips from last entry, a new BUY signal fires."""
    bus = AsyncEventBus()
    signals: list[SignalEvent] = []

    async def collect(evt: SignalEvent) -> None:
        signals.append(evt)

    sub = bus.subscribe(SignalEvent, collect, name="t.signal")
    ctx = _make_ctx(bus)
    # Seed one open BUY at 1.10
    ctx.position.update([_make_buy_position(ticket=1, open_price=1.10)])
    manager = LadderManager(ctx, step_pips=10, max_levels=2, level_volume=0.05)

    try:
        t = datetime(2024, 1, 1, tzinfo=UTC)

        # +8 pips — below step → no signal
        await manager.on_tick(ctx, Tick(symbol="EURUSD", time=t, bid=1.1008, ask=1.10082))
        await _drain(bus)
        assert signals == []

        # +10 pips exactly → first level fires
        await manager.on_tick(ctx, Tick(symbol="EURUSD", time=t, bid=1.1010, ask=1.10102))
        await _drain(bus)
        assert len(signals) == 1
        assert signals[0].signal.suggested_volume == pytest.approx(0.05)

        # +10 more pips from new reference (1.1010) → second level
        await manager.on_tick(ctx, Tick(symbol="EURUSD", time=t, bid=1.1020, ask=1.10202))
        await _drain(bus)
        assert len(signals) == 2

        # Third attempt — max_levels=2 already reached → no more signals
        await manager.on_tick(ctx, Tick(symbol="EURUSD", time=t, bid=1.1030, ask=1.10302))
        await _drain(bus)
        assert len(signals) == 2
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_ladder_sell_fires_on_step() -> None:
    """SELL: each time ask falls by step_pips from last entry, a new SELL fires."""
    bus = AsyncEventBus()
    signals: list[SignalEvent] = []

    async def collect(evt: SignalEvent) -> None:
        signals.append(evt)

    sub = bus.subscribe(SignalEvent, collect, name="t.signal")
    ctx = _make_ctx(bus)
    ctx.position.update([_make_sell_position(ticket=2, open_price=1.10)])
    manager = LadderManager(ctx, step_pips=10, max_levels=1, level_volume=0.1)

    try:
        t = datetime(2024, 1, 1, tzinfo=UTC)

        # ask falls to 1.0990 — exactly -10 pips → fires
        await manager.on_tick(ctx, Tick(symbol="EURUSD", time=t, bid=1.09898, ask=1.0990))
        await _drain(bus)
        assert len(signals) == 1
        assert signals[0].signal.side == Side.SELL

        # max_levels=1 — no more
        await manager.on_tick(ctx, Tick(symbol="EURUSD", time=t, bid=1.09798, ask=1.0980))
        await _drain(bus)
        assert len(signals) == 1
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_ladder_ignores_wrong_symbol() -> None:
    """Manager configured for EURUSD must ignore GBPUSD ticks."""
    bus = AsyncEventBus()
    signals: list[SignalEvent] = []

    async def collect(evt: SignalEvent) -> None:
        signals.append(evt)

    sub = bus.subscribe(SignalEvent, collect, name="t.signal")
    ctx = _make_ctx(bus)
    ctx.position.update([_make_buy_position(ticket=1, open_price=1.10)])
    manager = LadderManager(ctx, step_pips=5, max_levels=3, level_volume=0.1, symbol="EURUSD")

    try:
        t = datetime(2024, 1, 1, tzinfo=UTC)
        # GBPUSD tick 100 pips up — ignored
        await manager.on_tick(ctx, Tick(symbol="GBPUSD", time=t, bid=1.3100, ask=1.31002))
        await _drain(bus)
        assert signals == []
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_ladder_idle_with_no_positions() -> None:
    """When no position is open the manager must remain idle (no signal emitted)."""
    bus = AsyncEventBus()
    signals: list[SignalEvent] = []

    async def collect(evt: SignalEvent) -> None:
        signals.append(evt)

    sub = bus.subscribe(SignalEvent, collect, name="t.signal")
    ctx = _make_ctx(bus)
    # No position registered
    manager = LadderManager(ctx, step_pips=10, max_levels=3, level_volume=0.1)

    try:
        t = datetime(2024, 1, 1, tzinfo=UTC)
        for bid in (1.10, 1.1010, 1.1020, 1.1050):
            await manager.on_tick(ctx, Tick(symbol="EURUSD", time=t, bid=bid, ask=bid + 0.00002))
        await _drain(bus)
        assert signals == []
    finally:
        await sub.unsubscribe()
        await bus.close()
