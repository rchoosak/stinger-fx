"""ctx.move_pending — publishes ModifyOrderRequestEvent with the right fields."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
import structlog

from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.core.events import ModifyOrderRequestEvent
from stinger_fx.domain import Timeframe
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.parameters import StrategyParams
from tests._helpers import collect_into


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


async def _drain(bus: AsyncEventBus, *, ticks: int = 3) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_move_pending_publishes_event_with_price() -> None:
    bus = AsyncEventBus()
    seen: list[ModifyOrderRequestEvent] = []

    async def cap(evt: ModifyOrderRequestEvent) -> None:
        seen.append(evt)

    sub = bus.subscribe(ModifyOrderRequestEvent, cap, name="t.mod")
    ctx = _make_ctx(bus)

    try:
        await ctx.move_pending(42, price=1.1050, reason="trail_resistance")
        await _drain(bus)
        assert len(seen) == 1
        evt = seen[0]
        assert evt.ticket == 42
        assert evt.strategy_id == "test_strat"
        assert evt.price == pytest.approx(1.1050)
        assert evt.volume is None
        assert evt.sl is None
        assert evt.tp is None
        assert evt.reason == "trail_resistance"
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_move_pending_supports_volume_and_sl() -> None:
    bus = AsyncEventBus()
    seen: list[ModifyOrderRequestEvent] = []
    sub = bus.subscribe(
        ModifyOrderRequestEvent, collect_into(seen), name="t.mod"
    )
    ctx = _make_ctx(bus)

    try:
        await ctx.move_pending(7, volume=0.2, sl=1.0990)
        await _drain(bus)
        assert len(seen) == 1
        evt = seen[0]
        assert evt.volume == pytest.approx(0.2)
        assert evt.sl == pytest.approx(1.0990)
        assert evt.price is None
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_move_pending_with_no_args_raises() -> None:
    bus = AsyncEventBus()
    ctx = _make_ctx(bus)
    try:
        with pytest.raises(ValueError, match="at least one of"):
            await ctx.move_pending(1)
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_move_pending_without_bus_raises() -> None:
    ctx = StrategyContext(
        strategy_id="s",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        params=StrategyParams(),
        clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)),
        logger=structlog.get_logger("test"),
        magic=1,
        signal_sink=lambda s: asyncio.sleep(0),
        bus=None,
    )
    with pytest.raises(RuntimeError, match="bus"):
        await ctx.move_pending(1, price=1.10)
