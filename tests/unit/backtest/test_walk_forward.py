"""Walk-forward optimisation: slicing logic + end-to-end fold execution."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stinger_fx.backtest.walk_forward import WalkForward, slice_folds
from stinger_fx.config.models import (
    StrategyEntry,
    SweepRunConfig,
    WalkForwardConfig,
)
from stinger_fx.data import in_memory_store
from stinger_fx.data.parquet_store import ParquetStore
from stinger_fx.domain import Bar, Timeframe


# --- slice_folds ------------------------------------------------------------


def test_slice_folds_expanding_partition_is_anchored() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 12, 31, tzinfo=UTC)
    out = slice_folds(start, end, folds=4, scheme="expanding")
    assert len(out) == 4
    # in-sample always starts at `start` for expanding
    for in_start, in_end, oos_start, oos_end in out:
        assert in_start == start
        # OOS immediately follows in-sample
        assert oos_start == in_end
        # In-sample is non-empty
        assert in_end > in_start
        # OOS is non-empty
        assert oos_end > oos_start
    # Each successive in_end is later than the previous
    in_ends = [w[1] for w in out]
    assert in_ends == sorted(in_ends)
    # Last OOS reaches `end`
    assert out[-1][3] == end


def test_slice_folds_rolling_window_is_fixed_length() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 5, 1, tzinfo=UTC)
    out = slice_folds(start, end, folds=3, scheme="rolling", in_sample_ratio=0.6)
    assert len(out) == 3
    in_lens = [w[1] - w[0] for w in out]
    # All in-sample windows are the same length (within microsecond tolerance)
    assert max(in_lens) - min(in_lens) < timedelta(microseconds=10)
    # OOS windows are equal length too
    oos_lens = [w[3] - w[2] for w in out]
    assert max(oos_lens) - min(oos_lens) < timedelta(microseconds=10)
    # Each fold's OOS immediately follows its in-sample
    for in_start, in_end, oos_start, oos_end in out:
        assert oos_start == in_end


def test_slice_folds_rejects_invalid_inputs() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    t1 = datetime(2024, 2, 1, tzinfo=UTC)
    with pytest.raises(ValueError):
        slice_folds(t0, t1, folds=1)
    with pytest.raises(ValueError):
        slice_folds(t1, t0, folds=4)
    with pytest.raises(ValueError):
        slice_folds(t0, t1, folds=4, scheme="weird")


# --- End-to-end ------------------------------------------------------------


@pytest.fixture
def seeded_parquet(tmp_path: Path) -> Path:
    """200 bars of M15 EURUSD with up-down-up shape across the whole range."""
    root = tmp_path / "parquet"
    store = ParquetStore(root)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    prices = (
        [1.10 + 0.001 * i for i in range(70)]
        + [1.17 - 0.001 * i for i in range(70)]
        + [1.10 + 0.001 * i for i in range(60)]
    )
    bars = [
        Bar(
            symbol="EURUSD",
            timeframe=Timeframe.M15,
            time=base + timedelta(minutes=15 * i),
            open=p, high=p + 0.0002, low=p - 0.0002, close=p,
            tick_volume=100, is_closed=True,
        )
        for i, p in enumerate(prices)
    ]
    store.append_bars("EURUSD", Timeframe.M15, bars)
    return root


@pytest.mark.asyncio
async def test_walk_forward_produces_one_row_per_fold(
    seeded_parquet: Path, tmp_path: Path
) -> None:
    entry = StrategyEntry(
        id="ma_wf_target",
        class_path="stinger_fx.strategies.examples.ma_crossover:MACrossover",
        enabled=True,
        params={"symbol": "EURUSD", "timeframe": "M15", "volume": 0.1},
    )
    sweep_cfg = SweepRunConfig(
        id="wf_sweep",
        strategy_id="ma_wf_target",
        symbol="EURUSD",
        timeframe=Timeframe.M15,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=15 * 200),
        initial_balance=10_000.0,
        slippage_pips=0.5,
        data_source=seeded_parquet,
        parameter_grid={"fast": [3, 5], "slow": [10, 20]},
        rank_by="net_pnl",
        top_n=4,
    )
    wf_cfg = WalkForwardConfig(
        id="wf_smoke",
        sweep_id="wf_sweep",
        folds=3,
        scheme="expanding",
    )
    sqlite = in_memory_store()
    wf = WalkForward(
        base_strategy=entry,
        sqlite_store=sqlite,
        report_dir=tmp_path / "walk_forwards",
    )
    report = await wf.run(wf_cfg, sweep_cfg)

    assert len(report.folds) == 3
    for f in report.folds:
        assert f.out_of_sample_start == f.in_sample_end  # expanding contract
        assert isinstance(f.best_params, dict)
        # Out-of-sample metrics always include the rank-by key
        assert "net_pnl" in f.out_of_sample_metrics

    # Summary JSON written
    summary = tmp_path / "walk_forwards" / "wf_smoke_summary.json"
    assert summary.exists()
    body = json.loads(summary.read_text())
    assert body["wf_id"] == "wf_smoke"
    assert len(body["folds"]) == 3
    assert body["scheme"] == "expanding"


@pytest.mark.asyncio
async def test_walk_forward_inherits_rank_by_when_unset(
    seeded_parquet: Path, tmp_path: Path
) -> None:
    """rank_by=None on the WF config should pick up the sweep's rank_by."""
    entry = StrategyEntry(
        id="ma_wf_target2",
        class_path="stinger_fx.strategies.examples.ma_crossover:MACrossover",
        enabled=True,
        params={"symbol": "EURUSD", "timeframe": "M15", "volume": 0.1},
    )
    sweep_cfg = SweepRunConfig(
        id="wf_sweep2",
        strategy_id="ma_wf_target2",
        symbol="EURUSD",
        timeframe=Timeframe.M15,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=15 * 200),
        initial_balance=10_000.0,
        slippage_pips=0.5,
        data_source=seeded_parquet,
        parameter_grid={"fast": [3, 5], "slow": [10]},
        rank_by="sharpe",
        top_n=2,
    )
    wf_cfg = WalkForwardConfig(
        id="wf_inherit",
        sweep_id="wf_sweep2",
        folds=2,
        scheme="expanding",
        rank_by=None,
    )
    wf = WalkForward(
        base_strategy=entry,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "wf",
    )
    report = await wf.run(wf_cfg, sweep_cfg)
    assert report.rank_by == "sharpe"
