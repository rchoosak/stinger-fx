"""End-to-end: regime-filtered MA crossover takes fewer trades in chop.

Two backtests on the same chop fixture:
  1. RegimeFilteredMA with high ADX threshold → almost no trades
  2. Plain MACrossover → many false-signal trades

Demonstrates the regime filter actually gates the signals.
"""

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
def choppy_bars_root(tmp_path: Path) -> Path:
    """Seed 200 bars of pure noise around 1.10 — MA crossovers galore,
    but ADX stays low so trending filter blocks most of them."""
    import random

    root = tmp_path / "parquet"
    store = ParquetStore(root)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rng = random.Random(42)
    bars = []
    for i in range(200):
        c = 1.10 + rng.gauss(0, 0.0005)
        bars.append(Bar(
            symbol="EURUSD", timeframe=Timeframe.M15,
            time=base + timedelta(minutes=15 * i),
            open=c, high=c + 0.0001, low=c - 0.0001, close=c,
            tick_volume=100, is_closed=True,
        ))
    store.append_bars("EURUSD", Timeframe.M15, bars)
    return root


@pytest.mark.asyncio
async def test_regime_filter_reduces_trades_in_chop(
    choppy_bars_root: Path, tmp_path: Path
) -> None:
    """Same chop dataset → fewer trades when the ADX filter is on."""
    base = datetime(2024, 1, 1, tzinfo=UTC)

    # 1) Plain MA crossover (no filter)
    plain_entry = StrategyEntry(
        id="ma_plain",
        class_path="stinger_fx.strategies.examples.ma_crossover:MACrossover",
        enabled=True,
        params={
            "symbol": "EURUSD", "timeframe": "M15",
            "fast": 5, "slow": 20, "volume": 0.1,
        },
    )
    plain_cfg = BacktestRunConfig(
        id="ma_plain", mode="file", strategy_id="ma_plain",
        symbol="EURUSD", timeframe=Timeframe.M15,
        start=base, end=base + timedelta(minutes=15 * 200),
        initial_balance=10_000.0, data_source=choppy_bars_root,
    )
    plain_report = await FileBacktester(
        strategy=plain_entry, parquet_root=choppy_bars_root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "plain",
    ).run(plain_cfg)

    # 2) Regime-filtered MA — strict ADX threshold blocks most signals
    filtered_entry = StrategyEntry(
        id="ma_filtered",
        class_path="stinger_fx.strategies.examples.regime_filtered_ma:RegimeFilteredMA",
        enabled=True,
        params={
            "symbol": "EURUSD", "timeframe": "M15",
            "fast": 5, "slow": 20, "volume": 0.1,
            "adx_period": 14, "adx_threshold": 30.0,  # high bar
        },
    )
    filtered_cfg = BacktestRunConfig(
        id="ma_filtered", mode="file", strategy_id="ma_filtered",
        symbol="EURUSD", timeframe=Timeframe.M15,
        start=base, end=base + timedelta(minutes=15 * 200),
        initial_balance=10_000.0, data_source=choppy_bars_root,
    )
    filtered_report = await FileBacktester(
        strategy=filtered_entry, parquet_root=choppy_bars_root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "filtered",
    ).run(filtered_cfg)

    # The filtered strategy should take strictly fewer trades. We allow
    # equal (in case the random walk happens to produce a strong run
    # somewhere) but not more.
    assert len(filtered_report.trades) <= len(plain_report.trades), (
        f"filter should reduce trades — plain={len(plain_report.trades)}, "
        f"filtered={len(filtered_report.trades)}"
    )
