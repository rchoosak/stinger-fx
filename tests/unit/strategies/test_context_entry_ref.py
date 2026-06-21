"""ctx.buy/sell stamp `entry_ref_price` from the primary feed's latest price,
so the OrderRouter can size by risk (stop distance) for market entries."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import structlog

from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.domain import Bar, Tick, Timeframe
from stinger_fx.domain.signals import Signal
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.parameters import StrategyParams


def _make_ctx() -> tuple[StrategyContext, list[Signal]]:
    captured: list[Signal] = []

    async def sink(sig: Signal) -> None:
        captured.append(sig)

    ctx = StrategyContext(
        strategy_id="s",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        params=StrategyParams(),
        clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)),
        logger=structlog.get_logger("test"),
        magic=1,
        signal_sink=sink,
        bus=AsyncEventBus(),
    )
    return ctx, captured


@pytest.mark.asyncio
async def test_buy_stamps_entry_ref_from_last_tick_mid() -> None:
    ctx, captured = _make_ctx()
    ctx.history.update_tick(
        Tick(symbol="EURUSD", time=datetime(2024, 1, 1, tzinfo=UTC), bid=1.0990, ask=1.1010)
    )
    await ctx.buy(0.1, sl=1.095)
    assert captured[0].entry_ref_price == pytest.approx(1.1000)  # mid


@pytest.mark.asyncio
async def test_sell_stamps_entry_ref_from_last_bar_close_when_no_tick() -> None:
    ctx, captured = _make_ctx()
    ctx.history.append_bar(
        Bar(
            symbol="EURUSD", timeframe=Timeframe.M1,
            time=datetime(2024, 1, 1, tzinfo=UTC),
            open=1.10, high=1.11, low=1.09, close=1.105, volume=1, is_closed=True,
        )
    )
    await ctx.sell(0.1, sl=1.11)
    assert captured[0].entry_ref_price == pytest.approx(1.105)


@pytest.mark.asyncio
async def test_entry_ref_none_when_no_history() -> None:
    ctx, captured = _make_ctx()
    await ctx.buy(0.1, sl=1.095)
    assert captured[0].entry_ref_price is None


@pytest.mark.asyncio
async def test_pending_helper_uses_trigger_price_as_entry_ref() -> None:
    ctx, captured = _make_ctx()
    await ctx.buy_stop(1.1015, 0.1, sl=1.10)
    assert captured[0].entry_ref_price == pytest.approx(1.1015)
