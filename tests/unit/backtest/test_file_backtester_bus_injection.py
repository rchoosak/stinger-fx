"""FileBacktester `bus` injection + BacktestEquitySampleEvent emission.

These tests pin the two contract changes that the live-backtest feature
relies on:

  * Passing ``bus=...`` to ``FileBacktester(...)`` makes it publish into
    THAT bus instead of an isolated private one. Without this, the web
    UI's SSE handlers would never see the events.
  * ``BacktestEquitySampleEvent`` lands on the bus at every equity sample
    (per bar in bar mode, per UTC minute in tick mode). The chart's
    equity line is driven entirely off these events.

The legacy CLI path (bus=None) still creates its own bus — that's the
default-arg behavior we don't want to regress.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stinger_fx.backtest import FileBacktester
from stinger_fx.config.models import BacktestRunConfig, StrategyEntry
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import (
    BacktestEquitySampleEvent,
    BarEvent,
    TickEvent,
)
from stinger_fx.data import in_memory_store
from stinger_fx.data.parquet_store import ParquetStore
from stinger_fx.domain import Tick, Timeframe


@pytest.fixture
def trending_tick_root(tmp_path: Path) -> Path:
    """Reuse the trending-tick fixture pattern from the existing tick
    integration test — small enough to run fast, long enough to produce
    multiple equity samples."""
    root = tmp_path / "parquet"
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rise = [1.1000 + 0.0001 * i for i in range(120)]
    fall = [rise[-1] - 0.0001 * (i + 1) for i in range(120)]
    store = ParquetStore(root)
    ticks = [
        Tick(
            symbol="EURUSD",
            time=base + timedelta(seconds=i),
            bid=b,
            ask=b + 2e-5,
        )
        for i, b in enumerate(rise + fall)
    ]
    store.append_ticks("EURUSD", ticks)
    return root


def _entry() -> StrategyEntry:
    return StrategyEntry(
        id="ma_tick",
        class_path="stinger_fx.strategies.examples.ma_crossover:MACrossover",
        enabled=True,
        params={
            "symbol": "EURUSD", "timeframe": "M1",
            "fast": 2, "slow": 5, "volume": 0.1,
        },
    )


def _cfg(parquet_root: Path) -> BacktestRunConfig:
    return BacktestRunConfig(
        id="bus_test",
        mode="file",
        strategy_id="ma_tick",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=10),
        initial_balance=10_000.0,
        granularity="tick",
        data_source=parquet_root,
    )


@pytest.mark.asyncio
async def test_default_creates_own_isolated_bus(
    trending_tick_root: Path, tmp_path: Path
) -> None:
    """No ``bus`` passed → backtester creates its own. The external bus
    we set up here MUST stay empty — we're asserting isolation."""
    external = AsyncEventBus()
    seen_external: list[TickEvent] = []
    external.subscribe(TickEvent, lambda e: seen_external.append(e), name="probe")

    bt = FileBacktester(
        strategy=_entry(),
        parquet_root=trending_tick_root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "reports",
        # bus omitted on purpose
    )
    await bt.run(_cfg(trending_tick_root))

    assert seen_external == [], (
        "external bus must not receive events when not injected; "
        f"got {len(seen_external)} TickEvents"
    )
    await external.close()


@pytest.mark.asyncio
async def test_external_bus_receives_tick_and_bar_events(
    trending_tick_root: Path, tmp_path: Path
) -> None:
    """When ``bus=external`` is passed, every TickEvent / BarEvent the
    backtester publishes must land on that bus — that's the wire the SSE
    handler subscribes to."""
    external = AsyncEventBus()
    ticks: list[TickEvent] = []
    bars: list[BarEvent] = []

    async def on_tick(e: TickEvent) -> None:
        ticks.append(e)

    async def on_bar(e: BarEvent) -> None:
        bars.append(e)

    external.subscribe(TickEvent, on_tick, name="probe.tick")
    external.subscribe(BarEvent, on_bar, name="probe.bar")

    bt = FileBacktester(
        strategy=_entry(),
        parquet_root=trending_tick_root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "reports",
        bus=external,
    )
    await bt.run(_cfg(trending_tick_root))

    assert len(ticks) > 0, "external bus must receive TickEvents from replay"
    # ≥1 closed M1 bar (240 sec of ticks → 4 closed M1 bars expected)
    closed_bars = [b for b in bars if b.bar.is_closed]
    assert len(closed_bars) >= 1, "external bus must receive closed BarEvents"
    await external.close()


@pytest.mark.asyncio
async def test_equity_sample_events_published_per_minute_in_tick_mode(
    trending_tick_root: Path, tmp_path: Path
) -> None:
    """Tick mode samples equity once per UTC-minute boundary. Each sample
    must also publish ``BacktestEquitySampleEvent`` with consistent
    time/balance/equity values."""
    external = AsyncEventBus()
    samples: list[BacktestEquitySampleEvent] = []

    async def on_sample(e: BacktestEquitySampleEvent) -> None:
        samples.append(e)

    external.subscribe(BacktestEquitySampleEvent, on_sample, name="probe.equity")

    bt = FileBacktester(
        strategy=_entry(),
        parquet_root=trending_tick_root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "reports",
        bus=external,
    )
    report = await bt.run(_cfg(trending_tick_root))

    # 240s of ticks → ≥3 minute boundaries crossed → ≥3 samples
    assert len(samples) >= 3, (
        f"expected ≥3 per-minute equity samples; got {len(samples)}"
    )
    # Samples must mirror the equity_curve the report exposes (same time
    # values, same equity numbers). Allow for the report's "final close"
    # extra point at the very end — so we compare prefix.
    report_pairs = [(t.isoformat(), e) for (t, e) in report.equity_curve]
    sample_pairs = [(s.time.isoformat(), s.equity) for s in samples]
    n = min(len(report_pairs), len(sample_pairs))
    assert report_pairs[:n] == sample_pairs[:n], (
        "equity event stream must match the equity_curve recorded in the report"
    )
    # balance + mtm == equity invariant (in tick mode mtm rolls into equity)
    for s in samples:
        assert s.equity >= s.balance - 1e-6  # mtm may be slightly negative
    await external.close()


@pytest.mark.asyncio
async def test_external_bus_is_not_closed_by_backtester(
    trending_tick_root: Path, tmp_path: Path
) -> None:
    """The backtester must NOT close a bus it didn't create — otherwise
    the web UI's other SSE subscribers would be torn down after every run."""
    external = AsyncEventBus()
    bt = FileBacktester(
        strategy=_entry(),
        parquet_root=trending_tick_root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "reports",
        bus=external,
    )
    await bt.run(_cfg(trending_tick_root))

    # If the bus had been closed, publish would raise or be a no-op. Sanity
    # check by subscribing AFTER the run and publishing — handler must fire.
    received: list[int] = []

    async def handler(_evt: TickEvent) -> None:
        received.append(1)

    external.subscribe(TickEvent, handler, name="post-run.probe")
    tick = Tick(
        symbol="EURUSD",
        time=datetime(2024, 1, 1, tzinfo=UTC),
        bid=1.1, ask=1.10002,
    )
    await external.publish(TickEvent(tick=tick))
    # let the bus deliver
    import asyncio
    for _ in range(3):
        await asyncio.sleep(0)
    assert received == [1], "external bus must still be alive after backtester run"
    await external.close()
