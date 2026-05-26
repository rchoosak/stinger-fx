"""End-to-end OCO: runner dispatches on_position_closed → manager cancels siblings.

This verifies the wiring beyond the unit test: PositionClosedEvent on the
bus → runner._on_position_closed → manager.on_position_closed → ctx.close()
→ ClosePositionRequestEvent → router → broker.close_position.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import pytest

from stinger_fx.backtest.order_router import OrderRouter
from stinger_fx.backtest.replay_broker import SimBroker
from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.core.events import (
    PositionClosedEvent,
)
from stinger_fx.domain import Position, Side, Subscription, Timeframe
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.managers.oco import OCOGroupManager
from stinger_fx.strategies.parameters import StrategyParams
from stinger_fx.strategies.runner import StrategyRunner, derive_magic


class _OcoParams(StrategyParams):
    pass


class _OcoStrategy(BaseStrategy):
    name = "oco_test_strategy"
    Params = _OcoParams

    @classmethod
    def subscriptions(cls, params: _OcoParams) -> list[Subscription]:
        return [Subscription(symbol="EURUSD", timeframe=Timeframe.M1)]

    async def on_start(self, ctx) -> None:
        self._oco = OCOGroupManager(ctx)
        ctx.attach_manager(self._oco)


async def _drain(bus: AsyncEventBus, *, ticks: int = 6) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_oco_via_runner_cascades_close() -> None:
    """When ticket A closes (e.g. via SL), the runner's on_position_closed
    dispatch must drive the OCO manager to call ctx.close(B), which goes
    through the router and produces a real broker close on the sibling.
    """
    bus = AsyncEventBus()
    strategy_id = "oco_test_strategy"
    magic = derive_magic(strategy_id)
    broker = SimBroker(bus, initial_balance=10_000.0)
    broker.set_market_tick("EURUSD", 1.1000, 1.1002)

    # Seed two open positions owned by this strategy
    pos_a = Position(
        ticket=10,
        symbol="EURUSD",
        side=Side.BUY,
        volume=0.1,
        open_price=1.10,
        open_time=datetime(2024, 1, 1, tzinfo=UTC),
        magic=magic,
    )
    pos_b = Position(
        ticket=11,
        symbol="EURUSD",
        side=Side.SELL,
        volume=0.1,
        open_price=1.10,
        open_time=datetime(2024, 1, 1, tzinfo=UTC),
        magic=magic,
    )
    broker._positions[10] = pos_a
    broker._positions[11] = pos_b

    router = OrderRouter(bus, broker, strategy_magic={strategy_id: magic})
    await router.attach()

    strategy = _OcoStrategy()
    runner = StrategyRunner(
        strategy_id=strategy_id,
        strategy=strategy,
        params=_OcoParams(),
        bus=bus,
        clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)),
        reload_lock=asyncio.Lock(),
        signal_sink=lambda s: asyncio.sleep(0),
    )
    await runner.start()
    # Both tickets are now in ctx.position from the runner's POV? No — those
    # tickets weren't observed via OrderFilledEvent. So we need to register
    # them manually for the OCO test.
    strategy._oco.add(10, group_id="bracket")
    strategy._oco.add(11, group_id="bracket")

    closed_events: list[PositionClosedEvent] = []

    async def collect_closed(evt: PositionClosedEvent) -> None:
        closed_events.append(evt)

    sub = bus.subscribe(PositionClosedEvent, collect_closed, name="t.closed")

    try:
        # Simulate the broker emitting PositionClosedEvent for ticket 10
        # (e.g. SL fired). The runner should pick this up, dispatch to the
        # OCO manager, and the manager will call ctx.close(11).
        await bus.publish(PositionClosedEvent(position=pos_a, realized_pnl=-5.0))
        await _drain(bus)

        # We should see TWO PositionClosedEvent total — the original one for
        # ticket 10, plus the cascade-close for ticket 11 emitted by the
        # broker after the router handled the ctx.close(11) request.
        closed_tickets = sorted(e.position.ticket for e in closed_events)
        assert closed_tickets == [10, 11], (
            f"expected both tickets closed via OCO cascade, got: {closed_tickets}"
        )
        # Broker should have removed ticket 11 from its open-positions dict
        assert 11 not in broker._positions
    finally:
        await sub.unsubscribe()
        await runner.stop()
        await router.detach()
        await bus.close()
