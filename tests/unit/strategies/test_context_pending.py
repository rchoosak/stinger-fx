"""ctx.buy_stop / sell_stop / buy_limit / sell_limit (Phase 6.2.B)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.domain import OrderType, Side, Timeframe
from stinger_fx.domain.signals import Signal
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.parameters import StrategyParams


def _make_ctx() -> tuple[StrategyContext, list[Signal]]:
    """Build a StrategyContext that captures emitted signals in a list."""
    captured: list[Signal] = []

    async def sink(sig: Signal) -> None:
        captured.append(sig)

    ctx = StrategyContext(
        strategy_id="test_strat",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        params=StrategyParams(),
        clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)),
        logger=logging.getLogger("test"),
        magic=12345,
        signal_sink=sink,
        bus=AsyncEventBus(),
    )
    return ctx, captured


@pytest.mark.asyncio
async def test_buy_stop_emits_correct_signal() -> None:
    ctx, captured = _make_ctx()
    await ctx.buy_stop(1.1015, 0.1, sl=1.10, tp=1.105, comment="breakout")
    assert len(captured) == 1
    sig = captured[0]
    assert sig.side == Side.BUY
    assert sig.order_type == OrderType.STOP
    assert sig.suggested_price == pytest.approx(1.1015)
    assert sig.suggested_volume == pytest.approx(0.1)
    assert sig.suggested_sl == pytest.approx(1.10)
    assert sig.suggested_tp == pytest.approx(1.105)
    assert sig.comment == "breakout"


@pytest.mark.asyncio
async def test_sell_stop_emits_correct_signal() -> None:
    ctx, captured = _make_ctx()
    await ctx.sell_stop(1.0985, 0.1)
    assert len(captured) == 1
    sig = captured[0]
    assert sig.side == Side.SELL
    assert sig.order_type == OrderType.STOP
    assert sig.suggested_price == pytest.approx(1.0985)


@pytest.mark.asyncio
async def test_buy_limit_emits_correct_signal() -> None:
    ctx, captured = _make_ctx()
    await ctx.buy_limit(1.0980, 0.1)
    assert len(captured) == 1
    sig = captured[0]
    assert sig.side == Side.BUY
    assert sig.order_type == OrderType.LIMIT
    assert sig.suggested_price == pytest.approx(1.0980)


@pytest.mark.asyncio
async def test_sell_limit_emits_correct_signal() -> None:
    ctx, captured = _make_ctx()
    await ctx.sell_limit(1.1020, 0.1)
    assert len(captured) == 1
    sig = captured[0]
    assert sig.side == Side.SELL
    assert sig.order_type == OrderType.LIMIT
    assert sig.suggested_price == pytest.approx(1.1020)


@pytest.mark.asyncio
async def test_market_buy_still_emits_market_type() -> None:
    """Backward compat — ctx.buy() still emits MARKET signals."""
    ctx, captured = _make_ctx()
    await ctx.buy(0.1)
    assert len(captured) == 1
    sig = captured[0]
    assert sig.order_type == OrderType.MARKET
    assert sig.suggested_price is None
