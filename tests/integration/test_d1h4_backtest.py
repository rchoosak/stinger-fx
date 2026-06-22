"""End-to-end: D1H4TrendStrategy through the real FileBacktester.

Synthetic XAUUSD H1 data (respecting the default ForexWeekCalendar's open
hours) trends up — warming the D1 regime and firing a long breakout — then
reverses so the position exits. Exercises the full engine path (folding →
OrderRouter → SimBroker → trade journal) with the production calendar, not the
test seam. Shrunk indicator periods keep the warmup short.
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
from stinger_fx.strategies.aggregation import ForexWeekCalendar

SYMBOL = "XAUUSD"
PARAMS = {
    "symbol": SYMBOL, "daily_anchor_hour": 0,
    "d1_fast_ema": 3, "d1_slow_ema": 6, "d1_slope_lookback": 2, "d1_adx_length": 3,
    "d1_long_adx_min": 10.0, "d1_short_adx_min": 12.0, "d1_exit_ema": 4,
    "h4_fast_ema": 2, "h4_slow_ema": 3, "breakout_lookback": 3, "atr_length": 3,
    "initial_stop_atr": 2.5, "max_breakout_atr": 2.5, "max_channel_breakout_atr": 2.5,
    "chandelier_lookback": 3, "chandelier_atr": 3.0, "allow_short": True,
    "volume": 0.1,
}


def _open_hours_series(start: datetime, n_up: int, n_down: int) -> list[Bar]:
    """Rise then fall, emitting only forex-open H1 slots so the aggregator
    produces clean H4/D1 buckets under the default calendar."""
    cal = ForexWeekCalendar()
    bars: list[Bar] = []
    t, price = start, 1000.0
    target = n_up + n_down
    i = 0
    while i < target:
        if cal.is_open(t):
            slope = 2.0 if i < n_up else -3.0
            o = price
            c = max(price + slope, 1.0)
            bars.append(Bar(
                symbol=SYMBOL, timeframe=Timeframe.H1, time=t,
                open=o, high=max(o, c) + 0.3, low=min(o, c) - 0.3, close=c,
                tick_volume=10, is_closed=True,
            ))
            price = c
            i += 1
        t += timedelta(hours=1)
    return bars


@pytest.fixture
def h1_root(tmp_path: Path) -> tuple[Path, datetime, datetime]:
    root = tmp_path / "parquet"
    store = ParquetStore(root)
    start = datetime(2024, 1, 7, 22, 0, tzinfo=UTC)  # a Sunday FX open
    bars = _open_hours_series(start, n_up=480, n_down=200)
    store.append_bars(SYMBOL, Timeframe.H1, bars)
    return root, bars[0].time, bars[-1].time + timedelta(hours=1)


@pytest.mark.asyncio
async def test_d1h4_backtest_enters_and_exits(
    h1_root: tuple[Path, datetime, datetime], tmp_path: Path
) -> None:
    root, start, end = h1_root
    entry = StrategyEntry(
        id="d1h4_bt",
        class_path="stinger_fx.strategies.examples.d1h4_trend:D1H4TrendStrategy",
        enabled=True,
        params=PARAMS,
    )
    cfg = BacktestRunConfig(
        id="d1h4_bt", mode="file", strategy_id="d1h4_bt",
        symbol=SYMBOL, timeframe=Timeframe.H1, start=start, end=end,
        initial_balance=10_000.0, data_source=root,
        symbol_contract_sizes={SYMBOL: 100.0},
    )
    report = await FileBacktester(
        strategy=entry, parquet_root=root,
        sqlite_store=in_memory_store(), report_dir=tmp_path / "out",
    ).run(cfg)

    # The uptrend should produce at least one long entry that the reversal
    # closes — a full round-trip recorded in the trade journal.
    assert len(report.trades) >= 1, "expected at least one closed trade"
    assert any(t.side == "buy" for t in report.trades)
