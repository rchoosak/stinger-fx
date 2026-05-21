"""End-to-end file-backtest: seed Parquet → run → assert metrics + persistence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stinger_fx.backtest import FileBacktester
from stinger_fx.config.models import BacktestRunConfig, StrategyEntry
from stinger_fx.data import in_memory_store
from stinger_fx.data.parquet_store import ParquetStore
from stinger_fx.domain import Bar, Timeframe


@pytest.fixture
def trending_bars_root(tmp_path: Path) -> Path:
    root = tmp_path / "parquet"
    store = ParquetStore(root)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    # 120 bars: rise, fall, rise again — should produce at least two crossovers.
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
async def test_file_backtest_runs_ma_crossover(trending_bars_root: Path, tmp_path: Path) -> None:
    entry = StrategyEntry(
        id="ma_test",
        class_path="stinger_fx.strategies.examples.ma_crossover:MACrossover",
        enabled=True,
        params={"symbol": "EURUSD", "timeframe": "M15", "fast": 5, "slow": 20, "volume": 0.1},
    )
    cfg = BacktestRunConfig(
        id="smoke",
        mode="file",
        strategy_id="ma_test",
        symbol="EURUSD",
        timeframe=Timeframe.M15,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=15 * 200),
        initial_balance=10_000.0,
        slippage_pips=0.5,
    )
    bt = FileBacktester(
        strategy=entry,
        parquet_root=trending_bars_root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "reports",
    )
    report = await bt.run(cfg)
    assert len(report.trades) >= 1
    metrics = report.to_metrics_dict()
    assert metrics["trades"] >= 1
    assert (tmp_path / "reports" / "smoke_metrics.json").exists()
    assert (tmp_path / "reports" / "smoke_equity.parquet").exists()


@pytest.mark.asyncio
async def test_file_backtest_is_deterministic(trending_bars_root: Path, tmp_path: Path) -> None:
    """Same inputs must produce the same trades and net P&L on every run."""
    entry = StrategyEntry(
        id="ma_test",
        class_path="stinger_fx.strategies.examples.ma_crossover:MACrossover",
        enabled=True,
        params={"symbol": "EURUSD", "timeframe": "M15", "fast": 5, "slow": 20, "volume": 0.1},
    )
    cfg = BacktestRunConfig(
        id="det",
        mode="file",
        strategy_id="ma_test",
        symbol="EURUSD",
        timeframe=Timeframe.M15,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=15 * 200),
        initial_balance=10_000.0,
    )
    bt1 = FileBacktester(strategy=entry, parquet_root=trending_bars_root,
                        sqlite_store=in_memory_store(), report_dir=tmp_path / "r1")
    bt2 = FileBacktester(strategy=entry, parquet_root=trending_bars_root,
                        sqlite_store=in_memory_store(), report_dir=tmp_path / "r2")
    r1 = await bt1.run(cfg)
    r2 = await bt2.run(cfg)
    assert len(r1.trades) == len(r2.trades)
    assert r1.to_metrics_dict()["net_pnl"] == r2.to_metrics_dict()["net_pnl"]
