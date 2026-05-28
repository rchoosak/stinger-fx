"""Parameter sweep — cartesian enumeration, ranking, persistence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stinger_fx.backtest.sweep import (
    ParameterSweep,
    SweepCellResult,
    SweepReport,
    enumerate_grid,
)
from stinger_fx.config.models import StrategyEntry, SweepRunConfig
from stinger_fx.data import SweepRepo, in_memory_store
from stinger_fx.data.parquet_store import ParquetStore
from stinger_fx.domain import Bar, Timeframe

# --- Cartesian enumeration ---------------------------------------------------


def test_enumerate_grid_empty_returns_empty() -> None:
    assert enumerate_grid({}) == []


def test_enumerate_grid_single_param() -> None:
    out = enumerate_grid({"fast": [5, 10, 15]})
    assert out == [{"fast": 5}, {"fast": 10}, {"fast": 15}]


def test_enumerate_grid_cartesian_product_size() -> None:
    out = enumerate_grid({"fast": [5, 10, 15], "slow": [20, 30]})
    assert len(out) == 6
    # Verify each combo is unique and well-formed
    combos = {(c["fast"], c["slow"]) for c in out}
    assert combos == {(5, 20), (5, 30), (10, 20), (10, 30), (15, 20), (15, 30)}


# --- Ranking -----------------------------------------------------------------


def _cell(net_pnl: float, max_dd: float = 0.0, **extra) -> SweepCellResult:
    metrics = {"net_pnl": net_pnl, "max_drawdown": max_dd, **extra}
    return SweepCellResult(params={"x": net_pnl}, metrics=metrics)


def _report(rank_by: str, results: list[SweepCellResult]) -> SweepReport:
    now = datetime.now(UTC)
    return SweepReport(
        sweep_id="s",
        strategy_id="strat",
        started_at=now,
        finished_at=now,
        rank_by=rank_by,
        total_combos=len(results),
        results=results,
    )


def test_rank_by_net_pnl_bigger_is_better() -> None:
    r = _report("net_pnl", [_cell(50), _cell(200), _cell(-10)])
    ranked = [c.metrics["net_pnl"] for c in r.ranked]
    assert ranked == [200, 50, -10]
    best = r.best()
    assert best is not None
    assert best.metrics["net_pnl"] == 200


def test_rank_by_max_drawdown_smaller_is_better() -> None:
    r = _report("max_drawdown", [_cell(0, max_dd=200), _cell(0, max_dd=50), _cell(0, max_dd=500)])
    ranked = [c.metrics["max_drawdown"] for c in r.ranked]
    assert ranked == [50, 200, 500]
    best = r.best()
    assert best is not None
    assert best.metrics["max_drawdown"] == 50


def test_top_n_truncates() -> None:
    r = _report("net_pnl", [_cell(i) for i in range(10)])
    assert len(r.top(3)) == 3
    assert [c.metrics["net_pnl"] for c in r.top(3)] == [9, 8, 7]


# --- End-to-end --------------------------------------------------------------


@pytest.fixture
def seeded_parquet(tmp_path: Path) -> Path:
    """100 bars with a clear up-then-down pattern so MA crossover fires."""
    root = tmp_path / "parquet"
    store = ParquetStore(root)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    prices = (
        [1.10 + 0.001 * i for i in range(50)]
        + [1.15 - 0.001 * i for i in range(50)]
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
async def test_sweep_runs_all_combos_and_persists(
    seeded_parquet: Path, tmp_path: Path
) -> None:
    entry = StrategyEntry(
        id="ma_sweep_target",
        class_path="stinger_fx.strategies.examples.ma_crossover:MACrossover",
        enabled=True,
        params={"symbol": "EURUSD", "timeframe": "M15", "volume": 0.1},
    )
    cfg = SweepRunConfig(
        id="sweep_smoke",
        strategy_id="ma_sweep_target",
        symbol="EURUSD",
        timeframe=Timeframe.M15,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=15 * 200),
        initial_balance=10_000.0,
        slippage_pips=0.5,
        data_source=seeded_parquet,
        parameter_grid={"fast": [3, 5], "slow": [10, 20]},
        rank_by="net_pnl",
        top_n=5,
    )
    sqlite = in_memory_store()
    sweep = ParameterSweep(
        base_strategy=entry,
        sqlite_store=sqlite,
        report_dir=tmp_path / "sweeps",
    )
    report = await sweep.run(cfg)

    assert report.total_combos == 4
    assert len(report.results) == 4
    # Every result should have a numeric net_pnl
    for cell in report.results:
        assert isinstance(cell.metrics.get("net_pnl"), float)
    # Persistence
    summary = tmp_path / "sweeps" / "sweep_smoke_summary.json"
    assert summary.exists()
    # SQLite
    sweeps = SweepRepo(sqlite).list_sweeps()
    assert len(sweeps) == 1
    assert sweeps[0].sweep_id == "sweep_smoke"
    assert sweeps[0].total_combos == 4
    top_cells = SweepRepo(sqlite).top_cells("sweep_smoke", n=2)
    assert len(top_cells) == 2
    assert top_cells[0].rank == 1
    assert top_cells[1].rank == 2


@pytest.mark.asyncio
async def test_sweep_empty_grid_raises_at_config_validation() -> None:
    """SweepRunConfig validator must reject an empty parameter_grid."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SweepRunConfig(
            id="bad",
            strategy_id="x",
            symbol="EURUSD",
            timeframe=Timeframe.M15,
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 2, tzinfo=UTC),
            data_source=Path("/tmp"),
            parameter_grid={},
        )
    # Also reject an empty value list inside a key
    with pytest.raises(ValidationError):
        SweepRunConfig(
            id="bad2",
            strategy_id="x",
            symbol="EURUSD",
            timeframe=Timeframe.M15,
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 2, tzinfo=UTC),
            data_source=Path("/tmp"),
            parameter_grid={"fast": []},
        )
