"""End-to-end **tick-mode** backtest for OpeningRangeBreakout.

Companion to ``test_opening_range_breakout_backtest.py`` (bar mode).
The bar-mode test seeds two separate Parquet datasets (M1 + M5) and
relies on ``FileBacktester._replay_bars`` to merge feeds.  This file
seeds a **single XAUUSD tick stream**, runs with ``granularity="tick"``,
and verifies the BarAggregator builds M1 and M5 bars from that one
stream — proving the strategy works with the tick-data workflow
operators are likely to use in pre-live validation.

Pre-fix risk this test guards against
=====================================

Tick-mode replay creates one ``BarAggregator`` per declared
subscription — for this strategy, two aggregators on the same tick
stream — and the merger of those two aggregated bar streams must
satisfy three contracts simultaneously:

  * The M5 aggregator delivers enough closed bars for ``atr(5)`` to
    warm up BEFORE the OR window closes (so the entry gate has ATR
    data ready).
  * The M1 aggregator delivers the bars that build the OR window
    AND the post-OR trigger bar with the right open/high/low/close
    geometry to pierce the OR by ``breakout_buffer_atr × ATR``.
  * The two streams are merged in time order so that the
    just-closed M5 confirmation bar IS visible to the strategy when
    the M1 trigger fires on an M5 boundary (e.g. 07:20).

Any future change to ``BarAggregator``'s boundary semantics that
breaks one of those three contracts trips this test, even if the
bar-mode integration test still passes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stinger_fx.backtest import FileBacktester
from stinger_fx.config.models import BacktestRunConfig, StrategyEntry
from stinger_fx.data import in_memory_store
from stinger_fx.data.parquet_store import ParquetStore
from stinger_fx.domain import Tick, Timeframe

SYMBOL = "XAUUSD"
SESSION_HOUR = 7
# Pre-roll: 1.5h of ticks before the anchor so M5 aggregator has 18
# closed M5 bars (more than atr_period+1=6) by the time the OR window
# closes at anchor + 15m.
PRE_HOURS = 1.5
RUN_HOURS = 1   # OR build (15m) + breakout + management
TICK_SPACING = timedelta(seconds=30)  # 2 ticks per minute
SPREAD = 0.20


def _seed_tick_stream(root: Path, ticks: list[Tick]) -> None:
    store = ParquetStore(root)
    store.append_ticks(SYMBOL, ticks)


def _tick(t: datetime, bid: float) -> Tick:
    return Tick(symbol=SYMBOL, time=t, bid=bid, ask=bid + SPREAD)


# ---------------------------------------------------------------------- #
# Tick stream builders                                                     #
# ---------------------------------------------------------------------- #


def _pre_anchor_ticks(anchor: datetime) -> list[Tick]:
    """Mild oscillation around 2340 for PRE_HOURS hours.  Two ticks per
    minute keeps the BarAggregator producing clean M1+M5 bars."""
    ticks: list[Tick] = []
    mid = 2340.0
    n_minutes = int(PRE_HOURS * 60)
    for minute in range(n_minutes):
        t0 = anchor - timedelta(minutes=n_minutes - minute)
        t1 = t0 + TICK_SPACING
        offset = 0.2 if minute % 2 == 0 else -0.2
        ticks.append(_tick(t0, mid + offset))
        ticks.append(_tick(t1, mid - offset))
    return ticks


def _or_window_ticks(
    anchor: datetime, *, high: float, low: float,
) -> list[Tick]:
    """15 minutes of OR-building ticks.  Bars oscillate inside [low,
    high]; specific minutes touch the extremes so the aggregated M1
    bars set the OR boundaries.

    Pattern:
      * Even minute: tick1 high-ish, tick2 low-ish
      * Odd minute: reversed
      * Minute 4 touches the HIGH (tick1 == high)
      * Minute 9 touches the LOW (tick1 == low)
    """
    ticks: list[Tick] = []
    mid = (high + low) / 2
    for minute in range(15):
        t0 = anchor + timedelta(minutes=minute)
        t1 = t0 + TICK_SPACING
        if minute == 4:
            bid1, bid2 = high, mid + 0.5
        elif minute == 9:
            bid1, bid2 = low, mid - 0.5
        else:
            if minute % 2 == 0:
                bid1, bid2 = mid + 0.5, mid - 0.5
            else:
                bid1, bid2 = mid - 0.5, mid + 0.5
        ticks.append(_tick(t0, bid1))
        ticks.append(_tick(t1, bid2))
    return ticks


def _breakout_ticks(
    anchor: datetime, *, or_high: float, breakout_to: float,
) -> list[Tick]:
    """After the OR window: monotonic climb past or_high, then stable.
    The 5-minute M5 bar covering [anchor+15, anchor+20) closes well
    above or_high → confirms the breakout when the M1 at anchor+20
    fires."""
    ticks: list[Tick] = []
    total_minutes = (RUN_HOURS * 60) - 15   # minutes after OR window
    # Climb for first 10 minutes from or_high → breakout_to.
    climb_minutes = 10
    step = (breakout_to - or_high) / climb_minutes
    for i in range(total_minutes):
        minute = 15 + i
        t0 = anchor + timedelta(minutes=minute)
        t1 = t0 + TICK_SPACING
        if i < climb_minutes:
            price = or_high + step * i
            ticks.append(_tick(t0, price))
            ticks.append(_tick(t1, price + step))
        else:
            # Stable near breakout_to so SL/TP don't fire prematurely.
            ticks.append(_tick(t0, breakout_to + 0.05))
            ticks.append(_tick(t1, breakout_to - 0.05))
    return ticks


def _narrow_or_ticks(anchor: datetime, *, mid: float) -> list[Tick]:
    """Tick stream where the OR is so narrow (±0.3 USD around mid) that
    ``min_or_range_atr`` blocks the entry even if a breakout fires."""
    ticks: list[Tick] = []
    n_minutes = 15 + (RUN_HOURS * 60 - 15)
    for minute in range(n_minutes):
        t0 = anchor + timedelta(minutes=minute)
        t1 = t0 + TICK_SPACING
        offset = 0.3 if minute % 2 == 0 else -0.3
        ticks.append(_tick(t0, mid + offset))
        ticks.append(_tick(t1, mid - offset))
    return ticks


# ---------------------------------------------------------------------- #
# Config helpers                                                            #
# ---------------------------------------------------------------------- #


def _build_cfg(parquet_root: Path, anchor: datetime) -> BacktestRunConfig:
    return BacktestRunConfig(
        id="orb_tick_smoke",
        mode="file",
        strategy_id="orb",
        symbols=[SYMBOL],
        timeframes=[Timeframe.M1, Timeframe.M5],
        start=anchor - timedelta(hours=PRE_HOURS),
        end=anchor + timedelta(hours=RUN_HOURS),
        data_source=parquet_root,
        initial_balance=10_000.0,
        granularity="tick",   # ← the line under test
    )


def _build_entry(*, min_or_range_atr: float = 0.1) -> StrategyEntry:
    return StrategyEntry(
        id="orb",
        class_path=(
            "stinger_fx.strategies.examples.opening_range_breakout"
            ":OpeningRangeBreakout"
        ),
        enabled=True,
        params={
            "symbol": SYMBOL,
            "entry_timeframe": "M1",
            "structure_timeframe": "M5",
            "session_start_hour_utc": SESSION_HOUR,
            "session_end_hour_utc": 16,
            "opening_range_minutes": 15,
            "max_entry_minutes_from_open": 90,
            "atr_period": 5,
            "breakout_buffer_atr": 0.0,
            "min_or_range_atr": min_or_range_atr,
            "max_or_range_atr": 100.0,
            "sl_mode": "opposite_or",
            "tp_mode": "or_range",
            "min_rr": 0.3,
            "cooldown_bars": 0,
            "max_trades_per_session": 1,
        },
    )


# ---------------------------------------------------------------------- #
# Tests                                                                    #
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_tick_mode_takes_breakout_trade(tmp_path: Path) -> None:
    """Pre-roll + OR-build ticks + post-OR climbing ticks.  After the
    OR freezes at anchor+15m, the next M5 closes above or_high and the
    M1 trigger fires → BUY."""
    anchor = datetime(2024, 1, 1, SESSION_HOUR, 0, tzinfo=UTC)
    or_high, or_low = 2342.0, 2338.0
    breakout_to = 2345.0
    root = tmp_path / "parquet"
    ticks = (
        _pre_anchor_ticks(anchor)
        + _or_window_ticks(anchor, high=or_high, low=or_low)
        + _breakout_ticks(anchor, or_high=or_high, breakout_to=breakout_to)
    )
    _seed_tick_stream(root, ticks)

    bt = FileBacktester(
        strategy=_build_entry(),
        parquet_root=root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "reports",
    )
    report = await bt.run(_build_cfg(root, anchor))

    assert len(report.trades) >= 1, (
        f"expected at least one BUY breakout trade in tick mode — "
        f"got trades={report.trades}.  The aggregator must build "
        f"M1+M5 from the same tick stream, the OR-freeze logic must "
        f"activate on the first post-window bar, and the M5 "
        f"confirmation gate must read the just-closed M5 correctly."
    )
    assert report.trades[0].side == "buy"


@pytest.mark.asyncio
async def test_tick_mode_blocks_when_or_too_narrow(tmp_path: Path) -> None:
    """Ticks stay tightly clustered around mid throughout — the OR
    range is below ``min_or_range_atr × ATR`` and any would-be
    breakout is blocked."""
    anchor = datetime(2024, 1, 1, SESSION_HOUR, 0, tzinfo=UTC)
    root = tmp_path / "parquet"
    ticks = (
        _pre_anchor_ticks(anchor)
        + _narrow_or_ticks(anchor, mid=2340.0)
    )
    _seed_tick_stream(root, ticks)

    bt = FileBacktester(
        # Crank min_or_range_atr so 0.6-USD OR is filtered out.
        strategy=_build_entry(min_or_range_atr=50.0),
        parquet_root=root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "reports",
    )
    report = await bt.run(_build_cfg(root, anchor))

    assert report.trades == [], (
        f"narrow OR must block all entries via min_or_range_atr; "
        f"got {report.trades}"
    )
