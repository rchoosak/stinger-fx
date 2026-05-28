"""TrailingStopManager — ratchets SL toward the market on favourable moves."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
import structlog

from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.core.events import ModifyOrderRequestEvent
from stinger_fx.domain import Position, Side, Tick, Timeframe
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.managers.trailing import TrailingStopManager
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


def _make_position(side: Side, *, open_price: float, sl: float, ticket: int = 1) -> Position:
    return Position(
        ticket=ticket,
        symbol="EURUSD",
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
async def test_trailing_buy_ratchets_sl_up() -> None:
    bus = AsyncEventBus()
    seen: list[ModifyOrderRequestEvent] = []

    async def collect(evt: ModifyOrderRequestEvent) -> None:
        seen.append(evt)

    sub = bus.subscribe(ModifyOrderRequestEvent, collect, name="t.modify")
    ctx = _make_ctx(bus)
    ctx.position.update([_make_position(Side.BUY, open_price=1.10, sl=1.0980)])
    manager = TrailingStopManager(ctx, distance_pips=15)  # 15 pip = 0.0015

    try:
        # Bid rises to 1.1020 — desired SL = 1.1020 - 0.0015 = 1.1005 (advance!)
        await manager.on_tick(
            ctx, Tick(symbol="EURUSD", time=datetime(2024, 1, 1, tzinfo=UTC), bid=1.1020, ask=1.10202)
        )
        await _drain(bus)
        assert len(seen) == 1
        assert seen[0].sl == pytest.approx(1.1005)
        assert seen[0].reason == "trailing"

        # Bid rises further to 1.1030 — desired SL = 1.1015 (advance)
        await manager.on_tick(
            ctx, Tick(symbol="EURUSD", time=datetime(2024, 1, 1, tzinfo=UTC), bid=1.1030, ask=1.10302)
        )
        await _drain(bus)
        assert len(seen) == 2
        assert seen[1].sl == pytest.approx(1.1015)
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_trailing_buy_does_not_retreat() -> None:
    """A dip in price after the SL advanced must NOT pull SL back down."""
    bus = AsyncEventBus()
    seen: list[ModifyOrderRequestEvent] = []

    async def collect(evt: ModifyOrderRequestEvent) -> None:
        seen.append(evt)

    sub = bus.subscribe(ModifyOrderRequestEvent, collect, name="t.modify")
    ctx = _make_ctx(bus)
    ctx.position.update([_make_position(Side.BUY, open_price=1.10, sl=1.0980)])
    manager = TrailingStopManager(ctx, distance_pips=15)

    try:
        # Advance to SL = 1.1015
        await manager.on_tick(
            ctx, Tick(symbol="EURUSD", time=datetime(2024, 1, 1, tzinfo=UTC), bid=1.1030, ask=1.10302)
        )
        await _drain(bus)
        assert len(seen) == 1
        # Refresh ctx.position with the new SL so subsequent ticks compare
        # against the post-advance state (in real life this happens via
        # OrderModifiedEvent → runner._track_open)
        ctx.position.update([_make_position(Side.BUY, open_price=1.10, sl=1.1015)])

        # Bid dips to 1.1025 — desired SL = 1.1010 (retreat — REFUSE)
        await manager.on_tick(
            ctx, Tick(symbol="EURUSD", time=datetime(2024, 1, 1, tzinfo=UTC), bid=1.1025, ask=1.10252)
        )
        await _drain(bus)
        # No second event — trailing didn't retreat
        assert len(seen) == 1
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_trailing_sell_ratchets_sl_down() -> None:
    bus = AsyncEventBus()
    seen: list[ModifyOrderRequestEvent] = []

    async def collect(evt: ModifyOrderRequestEvent) -> None:
        seen.append(evt)

    sub = bus.subscribe(ModifyOrderRequestEvent, collect, name="t.modify")
    ctx = _make_ctx(bus)
    ctx.position.update([_make_position(Side.SELL, open_price=1.10, sl=1.1020)])
    manager = TrailingStopManager(ctx, distance_pips=15)

    try:
        # Ask falls to 1.0980 — desired SL = 1.0980 + 0.0015 = 1.0995 (advance)
        await manager.on_tick(
            ctx, Tick(symbol="EURUSD", time=datetime(2024, 1, 1, tzinfo=UTC), bid=1.09798, ask=1.0980)
        )
        await _drain(bus)
        assert len(seen) == 1
        assert seen[0].sl == pytest.approx(1.0995)
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_trailing_respects_activate_after_pips() -> None:
    """With activate_after_pips=20 the manager stays silent until the BUY
    is +20 pips in profit."""
    bus = AsyncEventBus()
    seen: list[ModifyOrderRequestEvent] = []

    async def collect(evt: ModifyOrderRequestEvent) -> None:
        seen.append(evt)

    sub = bus.subscribe(ModifyOrderRequestEvent, collect, name="t.modify")
    ctx = _make_ctx(bus)
    ctx.position.update([_make_position(Side.BUY, open_price=1.10, sl=1.0980)])
    manager = TrailingStopManager(ctx, distance_pips=10, activate_after_pips=20)

    try:
        # +10 pips — below activate threshold → no event
        await manager.on_tick(
            ctx, Tick(symbol="EURUSD", time=datetime(2024, 1, 1, tzinfo=UTC), bid=1.1010, ask=1.10102)
        )
        await _drain(bus)
        assert seen == []

        # +20 pips — exactly at threshold → fire
        await manager.on_tick(
            ctx, Tick(symbol="EURUSD", time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=1),
                      bid=1.1020, ask=1.10202)
        )
        await _drain(bus)
        assert len(seen) == 1
        # SL = 1.1020 - 10pip = 1.1010
        assert seen[0].sl == pytest.approx(1.1010)
    finally:
        await sub.unsubscribe()
        await bus.close()
