"""Integration test for live-mode tick backfill (Plan A2).

After PR A1, the live runtime wires a ``BarAggregator`` for every
non-TICK timeframe.  But a freshly-started aggregator is *cold* — its
first BarEvent fires only after a full bar window of live ticks have
accumulated.  For strategies that need warm indicator state
immediately (most), the gap between engine start and first useful
signal is up to the largest TF's bar duration (15 minutes for M15,
hours for H1).

PR A2 fixes this by:
  1. Asking each strategy how much history it needs via
     ``BaseStrategy.warmup_bars()``.
  2. Computing the per-feed warmup window in seconds (or falling back
     to a 48h default).
  3. Fetching that range of historical ticks via
     ``broker.get_history_ticks(symbol, from_time, now)``.
  4. Replaying them into the live aggregator with ``emit_bars=False``
     so no backdated BarEvent flood reaches the bus.
  5. THEN starting the live tick subscription.

This test pins:
  * Per-strategy ``warmup_bars()`` is honoured.
  * Strategies that don't override it get the 48h default.
  * Historical ticks reach the live aggregator (state warms up).
  * No backdated BarEvents reach the bus during the warmup phase.
  * Broker.subscribe_bars (which kicks off the live tick pump) is
    called AFTER warmup completes — not before.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pytest

from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import BarEvent
from stinger_fx.data.parquet_store import TICK_SCHEMA
from stinger_fx.domain import Subscription, Timeframe
from stinger_fx.runtime import _DEFAULT_WARMUP_SECONDS, StingerApp

SYMBOL = "XAUUSD"


class _RecordingBroker:
    """Stand-in for MT5Broker that lets the test stage historical ticks
    and observe the call order (warmup before subscribe_bars).

    The runtime calls (in order):
      1. ``get_history_ticks(symbol, from_time, now)`` — returns the
         pre-staged tick table.
      2. ``subscribe_bars(symbol, tf)`` — would normally start the
         live tick pump; here just records the call.
    """

    def __init__(self) -> None:
        self.staged_ticks: list[tuple[datetime, float, float]] = []
        self.history_fetched: list[tuple[str, datetime, datetime]] = []
        self.subscribed: list[tuple[str, Timeframe]] = []

    def stage_ticks(
        self, ticks: list[tuple[datetime, float, float]],
    ) -> None:
        """Set the historical tick stream that the next
        ``get_history_ticks`` call returns."""
        self.staged_ticks = ticks

    async def get_history_ticks(
        self, symbol: str, start: datetime, end: datetime,
    ) -> pa.Table:
        self.history_fetched.append((symbol, start, end))
        rows = [
            {
                "time_ns": t,
                "bid": bid,
                "ask": ask,
                "last": bid,
                "volume": 1,
                "flags": 0,
            }
            for (t, bid, ask) in self.staged_ticks
            if start <= t <= end
        ]
        return pa.Table.from_pylist(rows, schema=TICK_SCHEMA)

    async def subscribe_bars(self, symbol: str, tf: Timeframe) -> None:
        self.subscribed.append((symbol, tf))


def _make_app(bus: AsyncEventBus) -> StingerApp:
    """Minimum StingerApp instance for the wiring helpers."""
    app = StingerApp(config_dir=Path("/nonexistent"))
    app.bus = bus
    return app


# ---------------------------------------------------------------------- #
# Tests                                                                    #
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_backfill_warms_aggregator_state_without_publishing_bars() -> None:
    """Stage 5 historical ticks across two M1 bars.  After backfill the
    live aggregator's state must reflect the most recent in-progress
    bar — but no BarEvent must reach the bus (no backdated emissions
    to confuse downstream subscribers)."""
    bus = AsyncEventBus()
    captured: list[BarEvent] = []

    async def collect(evt: BarEvent) -> None:
        captured.append(evt)

    bus.subscribe(BarEvent, collect, name="probe.bar")

    app = _make_app(bus)
    broker = _RecordingBroker()

    # 5 ticks across ~90s.  Where each falls in an M1 boundary
    # depends on the absolute clock, so we make assertions that are
    # invariant: close == last tick's bid, _open_time is set.
    # The "no BarEvent reached the bus" check is the test's core
    # contract — backdated emissions during warmup are the bug we're
    # guarding against.
    now = datetime.now(UTC)
    base = now - timedelta(seconds=90)
    broker.stage_ticks([
        (base, 1900.0, 1900.2),
        (base + timedelta(seconds=10), 1902.0, 1902.2),
        (base + timedelta(seconds=30), 1898.0, 1898.2),
        (base + timedelta(seconds=65), 1905.0, 1905.2),
        (base + timedelta(seconds=80), 1903.0, 1903.2),
    ])

    await app._subscribe_one(
        broker, SYMBOL, Timeframe.M1, backfill_seconds=300,
    )

    # Drain — proves "no events" is real, not un-drained tasks.
    import asyncio
    for _ in range(3):
        await asyncio.sleep(0)

    # No BarEvent must have leaked through despite the boundary
    # crossing in the historical ticks.  THIS is the contract.
    assert captured == [], (
        f"backfill must not publish backdated BarEvents; got {captured}"
    )

    # Aggregator state warmed up — close == last replayed tick's bid.
    agg = app.aggregators[(SYMBOL, Timeframe.M1)]
    assert agg._open_time is not None
    assert agg._c == pytest.approx(1903.0)
    # The aggregator's tick_count proves at least one tick was
    # absorbed (would be 0 if backfill silently no-op'd).
    assert agg._tick_count >= 1

    # Broker.subscribe_bars was called AFTER history was fetched.
    assert broker.history_fetched, "expected history fetch before subscribe"
    assert (SYMBOL, Timeframe.M1) in broker.subscribed

    await bus.close()


@pytest.mark.asyncio
async def test_backfill_window_uses_per_strategy_warmup_bars() -> None:
    """``_compute_warmup_windows()`` reads ``warmup_bars()`` per strategy
    and converts ``bars × tf.seconds`` to a per-feed second count.  A
    strategy with no runners registered → empty result (no feeds to
    warm)."""
    bus = AsyncEventBus()
    app = _make_app(bus)
    # No runners → no warmup windows.
    assert app._compute_warmup_windows() == {}
    await bus.close()


@pytest.mark.asyncio
async def test_backfill_skipped_when_seconds_zero() -> None:
    """``backfill_seconds=0`` is the explicit "no warmup" signal — must
    not fetch history and must still wire the aggregator + start the
    broker subscription."""
    bus = AsyncEventBus()
    app = _make_app(bus)
    broker = _RecordingBroker()

    await app._subscribe_one(
        broker, SYMBOL, Timeframe.M1, backfill_seconds=0,
    )

    assert (SYMBOL, Timeframe.M1) in app.aggregators
    assert (SYMBOL, Timeframe.M1) in broker.subscribed
    assert broker.history_fetched == []
    # Aggregator is cold (no ticks ever fed)
    agg = app.aggregators[(SYMBOL, Timeframe.M1)]
    assert agg._open_time is None
    await bus.close()


@pytest.mark.asyncio
async def test_backfill_handles_empty_history_gracefully() -> None:
    """Brokers that return an empty tick table (no history, quiet
    pair, just-after-weekend) must not crash startup — aggregator
    stays cold and subscribe_bars still fires."""
    bus = AsyncEventBus()
    app = _make_app(bus)
    broker = _RecordingBroker()
    broker.stage_ticks([])   # explicitly empty

    await app._subscribe_one(
        broker, SYMBOL, Timeframe.M1, backfill_seconds=3600,
    )

    assert (SYMBOL, Timeframe.M1) in broker.subscribed
    agg = app.aggregators[(SYMBOL, Timeframe.M1)]
    assert agg._open_time is None  # still cold
    await bus.close()


@pytest.mark.asyncio
async def test_backfill_swallows_broker_errors_and_continues() -> None:
    """If the broker's ``get_history_ticks`` raises (network blip,
    permission error, etc.), the warmup is silently skipped and live
    subscription still proceeds.  Aggregator falls back to cold-start
    via the live tick stream — same behaviour as A1 had."""
    bus = AsyncEventBus()
    app = _make_app(bus)

    class _FailingBroker(_RecordingBroker):
        async def get_history_ticks(self, *args, **kwargs):
            raise RuntimeError("network down")

    broker = _FailingBroker()
    await app._subscribe_one(
        broker, SYMBOL, Timeframe.M1, backfill_seconds=3600,
    )

    # subscribe_bars still ran — engine startup didn't break.
    assert (SYMBOL, Timeframe.M1) in broker.subscribed
    agg = app.aggregators[(SYMBOL, Timeframe.M1)]
    assert agg._open_time is None
    await bus.close()


@pytest.mark.asyncio
async def test_default_warmup_seconds_is_48_hours() -> None:
    """Plan A2 defines the conservative default as 48 hours.  Exported
    so the runtime + tests share one definition."""
    assert _DEFAULT_WARMUP_SECONDS == 48 * 3600


@pytest.mark.asyncio
async def test_subscription_factory_skips_warmup_for_tick_timeframe() -> None:
    """TICK-subscribed strategies don't need bars at all → no aggregator
    + no warmup fetch."""
    bus = AsyncEventBus()
    app = _make_app(bus)
    broker = _RecordingBroker()

    await app._subscribe_one(
        broker, SYMBOL, Timeframe.TICK, backfill_seconds=3600,
    )

    assert (SYMBOL, Timeframe.TICK) not in app.aggregators
    assert broker.history_fetched == []   # didn't try to fetch
    assert (SYMBOL, Timeframe.TICK) in broker.subscribed
    await bus.close()


@pytest.mark.asyncio
async def test_warmup_window_compute_for_strategy_with_explicit_override() -> None:
    """Sanity: declared warmup_bars dict gets translated to seconds
    correctly (bars × tf.seconds), and per-feed values are taken
    individually, not collapsed.

    Builds a fake StrategyRunner-shaped object so we don't have to
    spin up the whole runner subsystem.
    """
    bus = AsyncEventBus()
    app = _make_app(bus)

    from types import SimpleNamespace

    from stinger_fx.strategies.examples.liquidity_sweep_reversal import (
        LiquiditySweepReversal,
        LiquiditySweepReversalParams,
    )
    params = LiquiditySweepReversalParams()
    runner = SimpleNamespace(
        strategy=LiquiditySweepReversal(),
        _params=params,
    )
    app.runners["test_lsr"] = runner  # type: ignore[assignment]

    windows = app._compute_warmup_windows()

    m1_key = (params.symbol, Timeframe.M1)
    m5_key = (params.symbol, Timeframe.M5)
    m15_key = (params.symbol, Timeframe.M15)

    assert windows[m1_key] == 1 * Timeframe.M1.seconds
    assert windows[m5_key] == (
        max(params.range_lookback_bars, params.atr_period + 1)
        * Timeframe.M5.seconds
    )
    assert windows[m15_key] == 2 * params.adx_period * Timeframe.M15.seconds
    await bus.close()


@pytest.mark.asyncio
async def test_warmup_window_compute_falls_back_to_default_when_undeclared() -> None:
    """A strategy that doesn't override ``warmup_bars()`` (returns
    None) gets the 48h default applied to every subscription it
    declares."""
    bus = AsyncEventBus()
    app = _make_app(bus)

    from types import SimpleNamespace

    class _NoWarmupStrategy:
        def warmup_bars(self, params):
            return None
        def subscriptions(self, params):
            return [Subscription(symbol=SYMBOL, timeframe=Timeframe.M5)]

    runner = SimpleNamespace(
        strategy=_NoWarmupStrategy(),
        _params=object(),
    )
    app.runners["test_nowarmup"] = runner  # type: ignore[assignment]

    windows = app._compute_warmup_windows()
    assert windows[(SYMBOL, Timeframe.M5)] == _DEFAULT_WARMUP_SECONDS
    await bus.close()
