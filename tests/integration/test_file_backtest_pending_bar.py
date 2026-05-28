"""End-to-end regression: bar-mode backtest fires pending orders.

Pre-fix bug: ``FileBacktester._replay_bars`` published BarEvent and
checked SL/TP, but never called ``broker.check_pending(...)`` (only the
tick-mode replay did). A strategy that placed a BUY_STOP would see the
order parked in SimBroker's ``_pending`` and stuck there — the backtest
report would show zero trades regardless of how the market moved.

This test runs the full FileBacktester end-to-end with a strategy that
issues exactly one ``ctx.buy_stop(1.1010)`` on its first bar. Seeded
parquet data climbs through 1.1010 in a later bar. After the fix:
  * The pending order triggers when the bar's range covers 1.1010.
  * A position opens and (later) gets closed by SL/TP or end-of-replay
    accounting — either way, ``report.trades`` is non-empty.
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
def rising_bars_root(tmp_path: Path) -> Path:
    """Seed 30 M15 bars where the price climbs through 1.1010 on bar #5
    (zero-indexed). The trigger sits firmly inside that bar's [low, high]
    range, so a pending BUY_STOP placed on bar #0 must fire on bar #5."""
    root = tmp_path / "parquet"
    store = ParquetStore(root)
    base = datetime(2024, 1, 1, tzinfo=UTC)

    bars = []
    for i in range(30):
        # Stair-step upward: 1.1000 → 1.1029. Bar #5 close = 1.1005 but
        # *high* reaches 1.1015 — covers the 1.1010 trigger.
        if i == 5:
            o, h, lo, c = 1.1005, 1.1015, 1.1003, 1.1005
        else:
            base_p = 1.1000 + 0.001 * i
            o = base_p
            c = base_p + 0.0001
            h = base_p + 0.0003
            lo = base_p - 0.0003
        bars.append(
            Bar(
                symbol="EURUSD",
                timeframe=Timeframe.M15,
                time=base + timedelta(minutes=15 * i),
                open=o,
                high=h,
                low=lo,
                close=c,
                tick_volume=100,
                is_closed=True,
            )
        )

    store.append_bars("EURUSD", Timeframe.M15, bars)
    return root


@pytest.mark.asyncio
async def test_bar_mode_backtest_triggers_pending_buy_stop(
    rising_bars_root: Path, tmp_path: Path
) -> None:
    """A BUY_STOP placed on the first bar must fire when a later bar's
    [low, high] range covers the trigger. Pre-fix this never happened
    in bar mode; the order stayed pending for the whole replay."""
    entry = StrategyEntry(
        id="breakout_test",
        class_path="tests.fixtures.breakout_strategy:BreakoutOnce",
        enabled=True,
        params={
            "symbol": "EURUSD",
            "timeframe": "M15",
            "trigger_price": 1.1010,
            "volume": 0.1,
        },
    )
    cfg = BacktestRunConfig(
        id="bar_pending_smoke",
        mode="file",
        strategy_id="breakout_test",
        symbol="EURUSD",
        timeframe=Timeframe.M15,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=15 * 30),
        initial_balance=10_000.0,
        # Default granularity = bar — this is the exact path that was broken.
    )
    bt = FileBacktester(
        strategy=entry,
        parquet_root=rising_bars_root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "reports",
    )
    report = await bt.run(cfg)

    # The headline assertion: the pending order actually fired AND
    # opened+closed a position observable in the report's trade list.
    # Pre-fix this was 0 (pending never triggered → no fill → no trade).
    assert len(report.trades) >= 1, (
        "bar-mode replay did not trigger any pending orders — "
        "_replay_bars is missing the check_pending_bar call. "
        f"report.trades={report.trades}"
    )

    # The trade's entry price should be at or near the trigger (1.1010
    # for a STOP filled at trigger + slippage).
    first_open = report.trades[0].open_price
    assert 1.1009 <= first_open <= 1.1012, (
        f"expected fill near trigger 1.1010, got {first_open}"
    )


@pytest.mark.asyncio
async def test_bar_mode_backtest_no_trade_when_trigger_never_hit(
    tmp_path: Path
) -> None:
    """Symmetric sanity: if the bar range never covers the trigger,
    the pending stays pending and no trade appears. This guards against
    the fix being too permissive (firing whenever any pending exists)."""
    root = tmp_path / "parquet"
    store = ParquetStore(root)
    base = datetime(2024, 1, 1, tzinfo=UTC)

    # All bars stay below 1.1010 — buy_stop never fires.
    bars = [
        Bar(
            symbol="EURUSD",
            timeframe=Timeframe.M15,
            time=base + timedelta(minutes=15 * i),
            open=1.0995,
            high=1.0999,
            low=1.0993,
            close=1.0996,
            tick_volume=100,
            is_closed=True,
        )
        for i in range(20)
    ]
    store.append_bars("EURUSD", Timeframe.M15, bars)

    entry = StrategyEntry(
        id="breakout_test",
        class_path="tests.fixtures.breakout_strategy:BreakoutOnce",
        enabled=True,
        params={
            "symbol": "EURUSD",
            "timeframe": "M15",
            "trigger_price": 1.1010,
            "volume": 0.1,
        },
    )
    cfg = BacktestRunConfig(
        id="bar_pending_no_trigger",
        mode="file",
        strategy_id="breakout_test",
        symbol="EURUSD",
        timeframe=Timeframe.M15,
        start=base,
        end=base + timedelta(minutes=15 * 20),
        initial_balance=10_000.0,
    )
    bt = FileBacktester(
        strategy=entry,
        parquet_root=root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "reports",
    )
    report = await bt.run(cfg)

    assert report.trades == [], (
        f"trigger 1.1010 was never reached but a trade fired anyway — "
        f"check_pending_bar is too permissive. trades={report.trades}"
    )
