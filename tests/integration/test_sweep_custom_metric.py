"""End-to-end: sweep ranks by a custom DSL metric defined in YAML.

Defines a custom metric ``risk_adjusted`` as
``sharpe - 0.5 * max_drawdown / 10``, then asks the sweep to rank by it.
The chosen "best" cell should match the cell whose computed
risk_adjusted score is highest — and the value must appear in the
metrics dict for every cell.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stinger_fx.backtest.sweep import ParameterSweep
from stinger_fx.config.models import StrategyEntry, SweepRunConfig
from stinger_fx.data import in_memory_store
from stinger_fx.data.parquet_store import ParquetStore
from stinger_fx.domain import Bar, Timeframe


@pytest.fixture
def trending_bars_root(tmp_path: Path) -> Path:
    """Seed enough bars for an MA crossover sweep."""
    root = tmp_path / "parquet"
    store = ParquetStore(root)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    prices = (
        [1.10 + 0.001 * i for i in range(50)]
        + [1.15 - 0.001 * i for i in range(50)]
        + [1.10 + 0.001 * i for i in range(20)]
    )
    store.append_bars(
        "EURUSD",
        Timeframe.M15,
        [
            Bar(
                symbol="EURUSD", timeframe=Timeframe.M15,
                time=base + timedelta(minutes=15 * i),
                open=p, high=p + 0.0002, low=p - 0.0002, close=p,
                tick_volume=100, is_closed=True,
            )
            for i, p in enumerate(prices)
        ],
    )
    return root


@pytest.mark.asyncio
async def test_sweep_ranks_by_custom_metric(trending_bars_root: Path, tmp_path: Path) -> None:
    """A 2×2 grid where rank_by points at a custom metric — every result
    cell must contain the custom value, and ranking must use it."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    entry = StrategyEntry(
        id="ma_custom",
        class_path="stinger_fx.strategies.examples.ma_crossover:MACrossover",
        enabled=True,
        params={
            "symbol": "EURUSD", "timeframe": "M15",
            "fast": 5, "slow": 20, "volume": 0.1,
        },
    )
    cfg = SweepRunConfig(
        id="ma_custom_sweep",
        strategy_id="ma_custom",
        symbol="EURUSD", timeframe=Timeframe.M15,
        start=base, end=base + timedelta(minutes=15 * 120),
        initial_balance=10_000.0,
        data_source=trending_bars_root,
        parameter_grid={"fast": [5, 10], "slow": [20, 30]},
        rank_by="risk_adjusted",
        algo="grid",
        custom_metrics={
            "risk_adjusted": "sharpe - 0.05 * max_drawdown",
        },
    )
    sweep = ParameterSweep(
        base_strategy=entry,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "sweep_out",
    )
    report = await sweep.run(cfg)

    # 4 cells (2×2 grid)
    assert len(report.results) == 4
    # Every cell has the custom metric computed
    for r in report.results:
        assert "risk_adjusted" in r.metrics
        assert isinstance(r.metrics["risk_adjusted"], (int, float))
        # And it equals sharpe - 0.05 * max_drawdown for that cell
        expected = r.metrics["sharpe"] - 0.05 * r.metrics["max_drawdown"]
        assert r.metrics["risk_adjusted"] == pytest.approx(expected, abs=1e-6)
    # Best cell is the one with highest risk_adjusted
    best = report.best()
    assert best is not None
    max_score = max(r.metrics["risk_adjusted"] for r in report.results)
    assert best.metrics["risk_adjusted"] == pytest.approx(max_score)


@pytest.mark.asyncio
async def test_sweep_with_invalid_custom_metric_fails_fast(
    trending_bars_root: Path, tmp_path: Path
) -> None:
    """A syntax error in a custom metric expression should raise
    MetricDSLError BEFORE any backtest runs — no wasted CPU."""
    from stinger_fx.backtest.metric_dsl import MetricDSLError

    base = datetime(2024, 1, 1, tzinfo=UTC)
    entry = StrategyEntry(
        id="ma_bad",
        class_path="stinger_fx.strategies.examples.ma_crossover:MACrossover",
        enabled=True,
        params={"symbol": "EURUSD", "timeframe": "M15",
                "fast": 5, "slow": 20, "volume": 0.1},
    )
    cfg = SweepRunConfig(
        id="ma_bad_sweep",
        strategy_id="ma_bad",
        symbol="EURUSD", timeframe=Timeframe.M15,
        start=base, end=base + timedelta(minutes=15 * 120),
        initial_balance=10_000.0,
        data_source=trending_bars_root,
        parameter_grid={"fast": [5], "slow": [20]},
        rank_by="net_pnl",
        algo="grid",
        custom_metrics={
            "broken": "sharpe +",  # syntax error
        },
    )
    sweep = ParameterSweep(
        base_strategy=entry,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "sweep_out",
    )
    with pytest.raises(MetricDSLError, match="syntax error"):
        await sweep.run(cfg)
