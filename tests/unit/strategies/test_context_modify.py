"""StrategyContext.move_stop / partial_close emit the right events.

These tests exercise the strategy-facing API in isolation — they don't go
through the OrderRouter or any broker. They verify:

  • `ctx.move_stop()` publishes a `ModifyOrderRequestEvent` with the
    requested fields.
  • `ctx.partial_close()` publishes a `PartialCloseRequestEvent`.
  • Both methods refuse to publish (or raise) when handed nonsense
    (empty modify, non-positive partial-close volume, no bus wired).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
import structlog

from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.core.events import (
    ModifyOrderRequestEvent,
    PartialCloseRequestEvent,
)
from stinger_fx.domain import Timeframe
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.parameters import StrategyParams


def _make_ctx(bus: AsyncEventBus | None = None) -> StrategyContext:
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


@pytest.mark.asyncio
async def test_move_stop_publishes_modify_request() -> None:
    bus = AsyncEventBus()
    seen: list[ModifyOrderRequestEvent] = []

    async def collect(evt: ModifyOrderRequestEvent) -> None:
        seen.append(evt)

    sub = bus.subscribe(ModifyOrderRequestEvent, collect, name="t.modify")
    ctx = _make_ctx(bus=bus)
    try:
        await ctx.move_stop(ticket=42, sl=1.09950, reason="trail")
        # Let the bus deliver
        for _ in range(3):
            await asyncio.sleep(0)
        assert len(seen) == 1
        evt = seen[0]
        assert evt.strategy_id == "test_strat"
        assert evt.ticket == 42
        assert evt.sl == 1.09950
        assert evt.tp is None
        assert evt.reason == "trail"
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_move_stop_noop_when_both_none() -> None:
    """Empty modify is a no-op — keeps the bus quiet."""
    bus = AsyncEventBus()
    seen: list[ModifyOrderRequestEvent] = []

    async def collect(evt: ModifyOrderRequestEvent) -> None:
        seen.append(evt)

    sub = bus.subscribe(ModifyOrderRequestEvent, collect, name="t.modify")
    ctx = _make_ctx(bus=bus)
    try:
        await ctx.move_stop(ticket=42)  # sl=None, tp=None
        for _ in range(3):
            await asyncio.sleep(0)
        assert seen == []
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_partial_close_publishes_request() -> None:
    bus = AsyncEventBus()
    seen: list[PartialCloseRequestEvent] = []

    async def collect(evt: PartialCloseRequestEvent) -> None:
        seen.append(evt)

    sub = bus.subscribe(PartialCloseRequestEvent, collect, name="t.partial")
    ctx = _make_ctx(bus=bus)
    try:
        await ctx.partial_close(ticket=7, volume=0.05, reason="take_half")
        for _ in range(3):
            await asyncio.sleep(0)
        assert len(seen) == 1
        evt = seen[0]
        assert evt.strategy_id == "test_strat"
        assert evt.ticket == 7
        assert evt.volume == pytest.approx(0.05)
        assert evt.reason == "take_half"
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_partial_close_rejects_non_positive_volume() -> None:
    ctx = _make_ctx(bus=AsyncEventBus())
    with pytest.raises(ValueError, match="volume must be > 0"):
        await ctx.partial_close(ticket=7, volume=0.0)
    with pytest.raises(ValueError, match="volume must be > 0"):
        await ctx.partial_close(ticket=7, volume=-0.1)


@pytest.mark.asyncio
async def test_modify_primitives_require_bus() -> None:
    """A context built without a bus= refuses modify/partial_close — better
    to fail loudly than silently drop the request."""
    ctx = _make_ctx(bus=None)
    with pytest.raises(RuntimeError, match="without a bus"):
        await ctx.move_stop(ticket=1, sl=1.0)
    with pytest.raises(RuntimeError, match="without a bus"):
        await ctx.partial_close(ticket=1, volume=0.01)
