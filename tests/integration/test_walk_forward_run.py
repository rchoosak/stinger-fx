"""End-to-end walk-forward: seed bars → run WalkForward → assert fold-by-fold metrics.

The seeded trend lets the MA crossover strategy actually find profitable
parameters in-sample, then we verify both folds produce non-None metrics
and that the consistency score is a well-defined float in [-1, 1].
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stinger_fx.backtest.walk_forward import WalkForward
from stinger_fx.config.models import StrategyEntry, WalkForwardConfig
from stinger_fx.data import in_memory_store
from stinger_fx.data.parquet_store import ParquetStore
from stinger_fx.domain import Bar, Timeframe


@pytest.fixture
def long_trend_bars(tmp_path: Path) -> Path:
    """240 M15 bars covering 60 hours of clear up/down trend."""
    root = tmp_path / "parquet"
    store = ParquetStore(root)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    # Slow oscillation: up 60 bars, down 60, up 60, down 60
    n = 240
    prices = []
    for i in range(n):
        cycle = i // 60
        local = i % 60
        if cycle % 2 == 0:
            p = 1.10 + 0.001 * local
        else:
            p = 1.16 - 0.001 * local
        prices.append(p)
    store.append_bars(
        "EURUSD",
        Timeframe.M15,
        [
            Bar(
                symbol="EURUSD",
                timeframe=Timeframe.M15,
                time=base + timedelta(minutes=15 * i),
                open=p,
                high=p + 0.0002,
                low=p - 0.0002,
                close=p,
                tick_volume=100,
                is_closed=True,
            )
            for i, p in enumerate(prices)
        ],
    )
    return root


@pytest.mark.asyncio
async def test_walk_forward_runs_with_grid_search(long_trend_bars: Path, tmp_path: Path) -> None:
    """3-fold expanding walk-forward across 60h of bars. Verifies the runner
    completes all folds, each fold records both IS and OOS metrics, and the
    summary file is persisted."""
    entry = StrategyEntry(
        id="ma_wf",
        class_path="stinger_fx.strategies.examples.ma_crossover:MACrossover",
        enabled=True,
        params={"symbol": "EURUSD", "timeframe": "M15", "fast": 5, "slow": 20, "volume": 0.1},
    )
    base = datetime(2024, 1, 1, tzinfo=UTC)
    cfg = WalkForwardConfig(
        id="wf_smoke",
        strategy_id="ma_wf",
        symbol="EURUSD",
        timeframe=Timeframe.M15,
        start=base,
        end=base + timedelta(minutes=15 * 240),
        initial_balance=10_000.0,
        data_source=long_trend_bars,
        n_folds=3,
        in_sample_pct=0.7,
        scheme="expanding",
        parameter_grid={"fast": [5, 10], "slow": [20, 30]},
        rank_by="net_pnl",
        algo="grid",
    )
    wf = WalkForward(
        base_strategy=entry,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "wf_reports",
    )
    report = await wf.run(cfg)

    # Every fold should have completed
    assert len(report.folds) == 3
    for fold in report.folds:
        assert "net_pnl" in fold.in_sample_metrics
        assert "net_pnl" in fold.oos_metrics
        # Best params are one of the 2x2 = 4 grid cells
        assert fold.best_params["fast"] in (5, 10)
        assert fold.best_params["slow"] in (20, 30)

    # Consistency score is a real number in [-1, 1]
    score = report.consistency_score
    assert -1.0 <= score <= 1.0

    # avg_oos_metric is set
    assert report.avg_oos_metric is not None

    # Persisted file exists
    summary_path = tmp_path / "wf_reports" / "wf_smoke_summary.json"
    assert summary_path.exists()


@pytest.mark.asyncio
async def test_walk_forward_rolling_scheme_in_sample_is_window_not_origin(
    long_trend_bars: Path, tmp_path: Path
) -> None:
    """In rolling scheme, fold N's in-sample starts at fold N's window —
    NOT at the global start."""
    entry = StrategyEntry(
        id="ma_wf_r",
        class_path="stinger_fx.strategies.examples.ma_crossover:MACrossover",
        enabled=True,
        params={"symbol": "EURUSD", "timeframe": "M15", "fast": 5, "slow": 20, "volume": 0.1},
    )
    base = datetime(2024, 1, 1, tzinfo=UTC)
    cfg = WalkForwardConfig(
        id="wf_rolling",
        strategy_id="ma_wf_r",
        symbol="EURUSD",
        timeframe=Timeframe.M15,
        start=base,
        end=base + timedelta(minutes=15 * 240),
        initial_balance=10_000.0,
        data_source=long_trend_bars,
        n_folds=3,
        in_sample_pct=0.5,
        scheme="rolling",
        parameter_grid={"fast": [5], "slow": [20]},  # single cell — fast
        rank_by="net_pnl",
        algo="grid",
    )
    wf = WalkForward(
        base_strategy=entry,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "wf_rolling_reports",
    )
    report = await wf.run(cfg)
    assert len(report.folds) == 3

    # In rolling scheme, each fold's IS window is its own slice (not from base)
    # Fold 0 IS starts at base; fold 1 and fold 2 should NOT start at base
    assert report.folds[0].in_sample_start == base
    assert report.folds[1].in_sample_start > base
    assert report.folds[2].in_sample_start > report.folds[1].in_sample_start
