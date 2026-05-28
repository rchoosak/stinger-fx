"""TimeExitManager — closes positions that exceed a time / bar limit."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
import structlog

from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.core.events import ClosePositionRequestEvent
from stinger_fx.domain import Bar, Position, Side, Tick, Timeframe
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.managers.time_exit import TimeExitManager
from stinger_fx.strategies.parameters import StrategyParams


def _make_ctx(bus: AsyncEventBus) -> StrategyContext:
    return StrategyContext(
        strategy_id="test_strat",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        params=StrategyParams(),
        clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)),
        logger=structlog.get_logger("test"),
        magic=12345,
        signal_sink=lambda s: asyncio.sleep(0),
        bus=bus,
    )


def _make_position(
    ticket: int = 1,
    *,
    open_time: datetime,
    symbol: str = "EURUSD",
) -> Position:
    return Position(
        ticket=ticket,
        symbol=symbol,
        side=Side.BUY,
        volume=0.1,
        open_price=1.10,
        open_time=open_time,
        sl=1.098,
        magic=12345,
    )


def _make_bar(symbol: str, time: datetime) -> Bar:
    return Bar(
        symbol=symbol,
        timeframe=Timeframe.M1,
        time=time,
        open=1.10,
        high=1.101,
        low=1.099,
        close=1.1005,
        tick_volume=100,
        is_closed=True,
    )


async def _drain(bus: AsyncEventBus, *, ticks: int = 3) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


# --- max_seconds mode ---------------------------------------------------------


@pytest.mark.asyncio
async def test_time_exit_seconds_fires_after_limit() -> None:
    """After max_seconds elapse, a ClosePositionRequestEvent must be published."""
    bus = AsyncEventBus()
    seen: list[ClosePositionRequestEvent] = []

    async def collect(evt: ClosePositionRequestEvent) -> None:
        seen.append(evt)

    sub = bus.subscribe(ClosePositionRequestEvent, collect, name="t.close")
    ctx = _make_ctx(bus)
    open_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    ctx.position.update([_make_position(ticket=1, open_time=open_time)])
    manager = TimeExitManager(ctx, max_seconds=3600)  # 1 hour

    try:
        # After 59 min — below limit → no close
        t59 = open_time + timedelta(minutes=59)
        await manager.on_tick(ctx, Tick(symbol="EURUSD", time=t59, bid=1.1005, ask=1.10052))
        await _drain(bus)
        assert seen == []

        # After 60 min exactly — at limit → close fires
        t60 = open_time + timedelta(hours=1)
        await manager.on_tick(ctx, Tick(symbol="EURUSD", time=t60, bid=1.1005, ask=1.10052))
        await _drain(bus)
        assert len(seen) == 1
        assert seen[0].ticket == 1
        assert seen[0].reason == "time_exit"
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_time_exit_seconds_idempotent() -> None:
    """Further ticks past the limit must NOT re-fire the close request."""
    bus = AsyncEventBus()
    seen: list[ClosePositionRequestEvent] = []

    async def collect(evt: ClosePositionRequestEvent) -> None:
        seen.append(evt)

    sub = bus.subscribe(ClosePositionRequestEvent, collect, name="t.close")
    ctx = _make_ctx(bus)
    open_time = datetime(2024, 1, 1, tzinfo=UTC)
    ctx.position.update([_make_position(ticket=1, open_time=open_time)])
    manager = TimeExitManager(ctx, max_seconds=60)

    try:
        for extra_min in (1, 2, 3, 5):
            t = open_time + timedelta(minutes=extra_min)
            await manager.on_tick(ctx, Tick(symbol="EURUSD", time=t, bid=1.1005, ask=1.10052))
        await _drain(bus)
        assert len(seen) == 1, f"expected exactly one close, got {len(seen)}"
    finally:
        await sub.unsubscribe()
        await bus.close()


# --- max_bars mode ------------------------------------------------------------


@pytest.mark.asyncio
async def test_time_exit_bars_fires_after_limit() -> None:
    """After max_bars closed bars, a ClosePositionRequestEvent must be published."""
    bus = AsyncEventBus()
    seen: list[ClosePositionRequestEvent] = []

    async def collect(evt: ClosePositionRequestEvent) -> None:
        seen.append(evt)

    sub = bus.subscribe(ClosePositionRequestEvent, collect, name="t.close")
    ctx = _make_ctx(bus)
    open_time = datetime(2024, 1, 1, tzinfo=UTC)
    ctx.position.update([_make_position(ticket=1, open_time=open_time)])
    manager = TimeExitManager(ctx, max_bars=3)

    try:
        # 2 bars — below limit
        for i in range(1, 3):
            await manager.on_bar(ctx, _make_bar("EURUSD", open_time + timedelta(minutes=i)))
        await _drain(bus)
        assert seen == []

        # 3rd bar — reaches limit → close
        await manager.on_bar(ctx, _make_bar("EURUSD", open_time + timedelta(minutes=3)))
        await _drain(bus)
        assert len(seen) == 1
        assert seen[0].ticket == 1
        assert seen[0].reason == "time_exit_bars"
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_time_exit_symbol_filter() -> None:
    """Manager configured for EURUSD must ignore GBPUSD ticks."""
    bus = AsyncEventBus()
    seen: list[ClosePositionRequestEvent] = []

    async def collect(evt: ClosePositionRequestEvent) -> None:
        seen.append(evt)

    sub = bus.subscribe(ClosePositionRequestEvent, collect, name="t.close")
    ctx = _make_ctx(bus)
    open_time = datetime(2024, 1, 1, tzinfo=UTC)
    ctx.position.update([_make_position(ticket=1, open_time=open_time, symbol="EURUSD")])
    manager = TimeExitManager(ctx, max_seconds=10, symbol="EURUSD")

    try:
        # GBPUSD tick after 60 s — wrong symbol, must be ignored
        t = open_time + timedelta(seconds=60)
        await manager.on_tick(ctx, Tick(symbol="GBPUSD", time=t, bid=1.27, ask=1.27002))
        await _drain(bus)
        assert seen == []
    finally:
        await sub.unsubscribe()
        await bus.close()
