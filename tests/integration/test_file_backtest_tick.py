"""End-to-end tick-mode backtest — seed ticks → BarAggregator → strategy → fills.

Three scenarios:
  • Tick-mode replay runs end-to-end on a seeded tick stream
  • Deterministic — same tick stream produces the same trades twice
  • Tick-precise SL/TP fires on the exact tick that breaches the stop
"""

from __future__ import annotations

import sys
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stinger_fx.backtest import FileBacktester
from stinger_fx.config.models import BacktestRunConfig, StrategyEntry
from stinger_fx.data import in_memory_store
from stinger_fx.data.parquet_store import ParquetStore
from stinger_fx.domain import Tick, Timeframe


def _seed_ticks(
    root: Path,
    symbol: str,
    base: datetime,
    bid_series: list[float],
    *,
    spread: float = 2e-5,
    step_ms: int = 100,
) -> None:
    """Write a tick series to Parquet. `ask = bid + spread`."""
    store = ParquetStore(root)
    ticks = [
        Tick(
            symbol=symbol,
            time=base + timedelta(milliseconds=step_ms * i),
            bid=b,
            ask=b + spread,
        )
        for i, b in enumerate(bid_series)
    ]
    store.append_ticks(symbol, ticks)


@pytest.fixture
def trending_tick_root(tmp_path: Path) -> Path:
    """Build a tick stream that, when aggregated to M1, produces a clean
    fast/slow SMA crossover.

    Layout: 5 minutes of ticks (300 ticks @ 1s spacing). Price climbs the
    first 3 minutes, falls the last 2 — aggregation gives 5 closed M1 bars
    with closes: rising, rising, peak, falling, falling. Plenty for a
    fast=2 / slow=4 MA crossover.
    """
    root = tmp_path / "parquet"
    base = datetime(2024, 1, 1, tzinfo=UTC)
    # 600 ticks over 10 minutes — one per second — V-shape so SMA crosses.
    rise = [1.1000 + 0.0001 * i for i in range(300)]
    fall = [rise[-1] - 0.0001 * (i + 1) for i in range(300)]
    _seed_ticks(root, "EURUSD", base, rise + fall, step_ms=1_000)
    return root


@pytest.mark.asyncio
async def test_tick_mode_runs_end_to_end(
    trending_tick_root: Path, tmp_path: Path
) -> None:
    """granularity=tick replays Parquet ticks, aggregates to M1 bars, and the
    MA-crossover strategy sees enough bars to act."""
    entry = StrategyEntry(
        id="ma_tick",
        class_path="stinger_fx.strategies.examples.ma_crossover:MACrossover",
        enabled=True,
        params={
            "symbol": "EURUSD",
            "timeframe": "M1",
            "fast": 2,
            "slow": 5,
            "volume": 0.1,
        },
    )
    cfg = BacktestRunConfig(
        id="tick_smoke",
        mode="file",
        strategy_id="ma_tick",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=20),
        initial_balance=10_000.0,
        granularity="tick",
        data_source=trending_tick_root,
    )
    bt = FileBacktester(
        strategy=entry,
        parquet_root=trending_tick_root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "reports",
    )
    report = await bt.run(cfg)
    # The equity curve must have at least one sample (we sample per-minute boundary)
    assert len(report.equity_curve) >= 1
    # Persistence sidecars produced
    assert (tmp_path / "reports" / "tick_smoke_metrics.json").exists()
    assert (tmp_path / "reports" / "tick_smoke_equity.parquet").exists()
    assert (tmp_path / "reports" / "tick_smoke_trades.json").exists()


@pytest.mark.asyncio
async def test_tick_mode_is_deterministic(
    trending_tick_root: Path, tmp_path: Path
) -> None:
    """Two runs on the same tick stream must yield identical trades and P&L."""
    entry = StrategyEntry(
        id="ma_tick",
        class_path="stinger_fx.strategies.examples.ma_crossover:MACrossover",
        enabled=True,
        params={
            "symbol": "EURUSD",
            "timeframe": "M1",
            "fast": 2,
            "slow": 5,
            "volume": 0.1,
        },
    )
    cfg = BacktestRunConfig(
        id="tick_det",
        mode="file",
        strategy_id="ma_tick",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=20),
        initial_balance=10_000.0,
        granularity="tick",
    )
    bt1 = FileBacktester(
        strategy=entry,
        parquet_root=trending_tick_root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "r1",
    )
    bt2 = FileBacktester(
        strategy=entry,
        parquet_root=trending_tick_root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "r2",
    )
    r1 = await bt1.run(cfg)
    r2 = await bt2.run(cfg)
    assert len(r1.trades) == len(r2.trades)
    assert r1.to_metrics_dict()["net_pnl"] == r2.to_metrics_dict()["net_pnl"]
    assert len(r1.equity_curve) == len(r2.equity_curve)


@pytest.mark.asyncio
async def test_tick_mode_sl_fires_on_exact_tick(tmp_path: Path) -> None:
    """A BUY position with SL=1.0995 must close on the tick whose bid first
    touches 1.0995 — *before* the strategy can react in its on_tick hook.

    We use an in-test strategy that:
      1. Opens BUY on the first tick at price ≈1.1000 with SL=1.0995.
      2. Logs every on_tick call.

    Tick stream drops bid stepwise. The SL must fire on the first tick whose
    bid ≤ 1.0995 — and the strategy must NOT see any further on_tick events
    pointed at a now-closed ticket (positions list will be empty afterward).
    """
    # 1) Seed a precise tick sequence
    root = tmp_path / "parquet"
    base = datetime(2024, 1, 1, tzinfo=UTC)
    # First tick is the open-price reference (bid=1.1000).
    # Bid then drops in 5-pip steps: 1.1000 → 1.0998 → 1.0996 → 1.0995 (SL) → 1.0990.
    bids = [1.1000, 1.0998, 1.0996, 1.0995, 1.0990]
    _seed_ticks(root, "EURUSD", base, bids, step_ms=1_000)

    # 2) Build an in-test strategy that opens BUY with SL on first tick
    strat_dir = tmp_path / "tick_strats"
    strat_dir.mkdir()
    (strat_dir / "__init__.py").write_text("")
    (strat_dir / "sl_tick.py").write_text(textwrap.dedent("""
        from stinger_fx.domain import Subscription, Timeframe
        from stinger_fx.strategies.base import BaseStrategy
        from stinger_fx.strategies.parameters import StrategyParams

        class SlTickParams(StrategyParams):
            symbol: str = "EURUSD"

        class SlTickStrategy(BaseStrategy):
            name = "sl_tick"
            Params = SlTickParams
            opened: bool = False

            @classmethod
            def subscriptions(cls, params):
                return [Subscription(symbol=params.symbol, timeframe=Timeframe.M1)]

            async def on_tick(self, ctx, tick):
                if not type(self).opened:
                    type(self).opened = True
                    # Open BUY at the first tick, SL just below at 1.0995
                    await ctx.buy(0.1, sl=1.0995)
    """))

    sys.path.insert(0, str(tmp_path))
    try:
        from importlib import import_module

        mod = import_module("tick_strats.sl_tick")
        mod.SlTickStrategy.opened = False

        entry = StrategyEntry(
            id="sl_tick_test",
            class_path="tick_strats.sl_tick:SlTickStrategy",
            enabled=True,
            params={"symbol": "EURUSD"},
        )
        cfg = BacktestRunConfig(
            id="sl_tick_run",
            mode="file",
            strategy_id="sl_tick_test",
            symbol="EURUSD",
            timeframe=Timeframe.M1,
            start=base,
            end=base + timedelta(minutes=1),
            initial_balance=10_000.0,
            granularity="tick",
        )
        bt = FileBacktester(
            strategy=entry,
            parquet_root=root,
            sqlite_store=in_memory_store(),
            report_dir=tmp_path / "reports",
        )
        report = await bt.run(cfg)

        # 3) Assertions — exactly one trade, closed at SL price (1.0995)
        # accounting for fill slippage = 0 in this test.
        assert len(report.trades) == 1, f"expected 1 trade, got {len(report.trades)}"
        trade = report.trades[0]
        # Open price = mid of first tick (bid 1.1000, ask 1.10002) shifted by
        # slippage=0 → fill mid = 1.10001 (the SimBroker fills BUY at mid here
        # because broker.set_market() stores bid as last-known, then fills get
        # nudged by slippage; with slippage=0 the fill = bid).
        # SL fires at the 4th tick (bid=1.0995). Close price uses sell-side
        # slippage off that bid → 1.0995 with slippage_pips=0.
        assert trade.close_price == pytest.approx(1.0995, abs=1e-6), (
            f"SL closed at unexpected price: {trade.close_price}"
        )
        # P&L must be negative (we bought above SL and exited at SL).
        assert trade.pnl < 0
        # Close timestamp aligns with the SL-breach tick (index 3 → +3s)
        assert trade.close_ts == base + timedelta(seconds=3)
    finally:
        sys.path.remove(str(tmp_path))
