"""Multiple StrategyRunners on one bus dispatch independently and stay isolated.

Regression guard for multi-strategy / portfolio operation. A single `BarEvent`
must reach *every* subscribed runner; each runner attributes its own fills by
magic (no cross-talk between strategies sharing an account); and per-strategy
state — the `HistoryView` and the streaming indicators added for the backtest
perf work — is independent per runner.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from stinger_fx.backtest.order_router import OrderRouter
from stinger_fx.backtest.replay_broker import SimBroker
from stinger_fx.config.models import RiskConfig
from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.core.events import (
    AccountSnapshotEvent,
    BarEvent,
    DecisionEvent,
    SignalEvent,
)
from stinger_fx.domain import AccountSnapshot, Bar, Side, Subscription, Timeframe
from stinger_fx.risk import RiskMonitor
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.parameters import StrategyParams
from stinger_fx.strategies.runner import StrategyRunner, derive_magic

_T0 = datetime(2024, 1, 1, tzinfo=UTC)


class _OneShotParams(StrategyParams):
    side: Side = Side.BUY


class _OneShotStrategy(BaseStrategy):
    """Reads a streaming indicator via the runner-owned HistoryView, then places
    exactly one order (direction from params) on the first warm bar."""

    name = "oneshot"
    Params = _OneShotParams

    @classmethod
    def subscriptions(cls, params: StrategyParams) -> list[Subscription]:
        return [Subscription(symbol="EURUSD", timeframe=Timeframe.M1)]

    def __init__(self) -> None:
        super().__init__()
        self.bars_seen = 0
        self.rsi_seen: float | None = None
        self.placed = False

    async def on_bar(self, ctx, bar: Bar) -> None:  # type: ignore[no-untyped-def]
        self.bars_seen += 1
        r = ctx.history.rsi(14)  # streaming accessor, exercised through the runner
        if r is None or self.placed:
            return
        self.rsi_seen = r
        self.placed = True
        if ctx.params.side is Side.BUY:
            await ctx.buy(volume=0.1, sl=bar.close - 0.01)
        else:
            await ctx.sell(volume=0.1, sl=bar.close + 0.01)


async def _drain(bus: AsyncEventBus, *, ticks: int = 8) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_two_runners_one_bus_dispatch_attribution_isolation() -> None:
    bus = AsyncEventBus()
    broker = SimBroker(bus, initial_balance=10_000.0)
    broker.set_market_tick("EURUSD", 1.0999, 1.1001)

    buyer_id, seller_id = "buyer", "seller"
    magic_buyer, magic_seller = derive_magic(buyer_id), derive_magic(seller_id)
    router = OrderRouter(
        bus, broker,
        strategy_magic={buyer_id: magic_buyer, seller_id: magic_seller},
    )
    await router.attach()

    async def sink(sig) -> None:  # type: ignore[no-untyped-def]
        await bus.publish(SignalEvent(signal=sig))

    buyer, seller = _OneShotStrategy(), _OneShotStrategy()
    runner_b = StrategyRunner(
        strategy_id=buyer_id, strategy=buyer, params=_OneShotParams(side=Side.BUY),
        bus=bus, clock=SimClock(_T0), reload_lock=asyncio.Lock(), signal_sink=sink,
    )
    runner_s = StrategyRunner(
        strategy_id=seller_id, strategy=seller, params=_OneShotParams(side=Side.SELL),
        bus=bus, clock=SimClock(_T0), reload_lock=asyncio.Lock(), signal_sink=sink,
    )
    await runner_b.start()
    await runner_s.start()

    try:
        # One BarEvent stream; both runners are subscribed to the same feed.
        px = 1.10
        for i in range(20):
            px += 0.0002 if i % 2 == 0 else -0.0001  # gains + losses → real RSI
            await bus.publish(BarEvent(bar=Bar(
                symbol="EURUSD", timeframe=Timeframe.M1,
                time=_T0 + timedelta(minutes=i), open=px, high=px + 0.0005,
                low=px - 0.0005, close=px, tick_volume=1, is_closed=True,
            )))
            await _drain(bus)

        # 1) Every bar reached BOTH runners.
        assert buyer.bars_seen == 20
        assert seller.bars_seen == 20

        # 2) Both placed; the streaming accessor worked through the runner and
        #    gave the same value (same feed) — consistent but independently kept.
        assert buyer.placed and seller.placed
        assert buyer.rsi_seen is not None
        assert buyer.rsi_seen == seller.rsi_seen

        # 3) Two positions on the shared broker, each with the right magic + side.
        positions = list(broker._positions.values())
        assert len(positions) == 2
        by_magic = {p.magic: p for p in positions}
        assert by_magic[magic_buyer].side is Side.BUY
        assert by_magic[magic_seller].side is Side.SELL

        # 4) Per-strategy attribution: each runner's PositionView sees ONLY its
        #    own position (magic), never the sibling's — no cross-talk.
        assert [p.magic for p in runner_b._ctx.position.all()] == [magic_buyer]
        assert [p.magic for p in runner_s._ctx.position.all()] == [magic_seller]

        # 5) Isolation: the runners own distinct HistoryViews + distinct streaming
        #    indicator state for the same feed (no shared/leaked Wilder state).
        vb = runner_b._ctx.history_for("EURUSD", Timeframe.M1)
        vs = runner_s._ctx.history_for("EURUSD", Timeframe.M1)
        assert vb is not None and vs is not None
        assert vb is not vs
        assert vb._inc is not vs._inc
    finally:
        await runner_b.stop()
        await runner_s.stop()
        await router.detach()
        await bus.close()


@pytest.mark.asyncio
async def test_account_kill_switch_blocks_every_runner() -> None:
    """An account-level gate is *shared* across every runner on the bus: once the
    kill switch trips, neither strategy's order reaches the broker."""
    bus = AsyncEventBus()
    broker = SimBroker(bus, initial_balance=10_000.0)
    broker.set_market_tick("EURUSD", 1.0999, 1.1001)

    a_id, b_id = "alpha", "beta"
    magic_a, magic_b = derive_magic(a_id), derive_magic(b_id)

    rm = RiskMonitor(bus, RiskConfig(kill_switch_drawdown_pct=20.0))
    await rm.start()

    def _snap(equity: float) -> AccountSnapshotEvent:
        return AccountSnapshotEvent(snapshot=AccountSnapshot(
            account_id="x", time=_T0, balance=10_000.0, equity=equity,
            margin=0.0, free_margin=equity))

    # Peak 12_000 then 9_000 → 25% drawdown → account-wide kill switch tripped.
    await rm._on_snapshot(_snap(12_000.0))
    await rm._on_snapshot(_snap(9_000.0))
    assert rm.snapshot()["kill_switch_tripped"] is True

    router = OrderRouter(
        bus, broker,
        strategy_magic={a_id: magic_a, b_id: magic_b},
        risk=rm,
    )
    await router.attach()

    async def sink(sig) -> None:  # type: ignore[no-untyped-def]
        await bus.publish(SignalEvent(signal=sig))

    a, b = _OneShotStrategy(), _OneShotStrategy()
    runner_a = StrategyRunner(
        strategy_id=a_id, strategy=a, params=_OneShotParams(side=Side.BUY),
        bus=bus, clock=SimClock(_T0), reload_lock=asyncio.Lock(), signal_sink=sink,
    )
    runner_b = StrategyRunner(
        strategy_id=b_id, strategy=b, params=_OneShotParams(side=Side.BUY),
        bus=bus, clock=SimClock(_T0), reload_lock=asyncio.Lock(), signal_sink=sink,
    )
    await runner_a.start()
    await runner_b.start()

    # Capture pre-trade rejections so we can assert the *reason*, not just the
    # absence of positions (which any rejection would produce).
    rejected: list[DecisionEvent] = []
    rej_sub = bus.subscribe(
        DecisionEvent,
        lambda e: rejected.append(e) if e.decision.action == "rejected" else None,
        name="t.reject",
    )

    try:
        px = 1.10
        for i in range(20):
            px += 0.0002 if i % 2 == 0 else -0.0001
            await bus.publish(BarEvent(bar=Bar(
                symbol="EURUSD", timeframe=Timeframe.M1,
                time=_T0 + timedelta(minutes=i), open=px, high=px + 0.0005,
                low=px - 0.0005, close=px, tick_volume=1, is_closed=True,
            )))
            await _drain(bus)

        # Both strategies tried to enter, but the shared kill switch rejected
        # every order — the broker opened nothing...
        assert a.placed and b.placed
        assert broker._positions == {}
        # ...and the rejection is specifically the kill switch, for BOTH runners
        # (not some unrelated gate that merely happened to block them).
        assert {e.decision.signal.strategy_id for e in rejected} == {a_id, b_id}
        assert all(e.decision.reason == "kill_switch_tripped" for e in rejected)
    finally:
        await rej_sub.unsubscribe()
        await runner_a.stop()
        await runner_b.stop()
        await router.detach()
        await bus.close()
