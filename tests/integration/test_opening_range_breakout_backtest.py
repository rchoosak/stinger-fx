"""End-to-end multi-feed backtest for OpeningRangeBreakout.

The strategy needs two timeframes wired through the backtester:

  * M1 — to build the opening range AND fire the entry trigger
  * M5 — for the confirmation candle (last closed M5 close beyond OR)
    and the ATR scale

This file seeds synthetic XAU/USD bars at both timeframes into a
Parquet root, hands them to ``FileBacktester`` via the multi-feed
``symbols=["XAUUSD"], timeframes=[M1, M5]`` shape, and asserts:

  * Given a clean OR followed by a directional breakout that is
    confirmed by the next M5 close, the strategy fires a BUY trade
    after the OR window freezes.
  * Given the same OR followed by sideway behaviour (no breakout),
    no trades are taken.

The exact P&L isn't asserted; this verifies wiring (multi-TF dispatch
+ OR freeze + M5 confirmation gate + SL/TP placement), not
profitability under synthetic data.
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

SYMBOL = "XAUUSD"
SESSION_HOUR = 7  # London open
# Run for 2 hours — enough for OR build + breakout + position management.
RUN_HOURS = 2
# Pre-roll M5 history starts at 04:00 UTC — gives 3h of pre-anchor M5
# bars so ATR(14) is warm by the time the trigger fires at ~07:20.
PRE_HOURS = 3


def _bar(
    *, tf: Timeframe, t: datetime, o: float, h: float, lo: float, c: float,
    vol: int = 100,
) -> Bar:
    return Bar(
        symbol=SYMBOL, timeframe=tf, time=t,
        open=o, high=h, low=lo, close=c,
        tick_volume=vol, is_closed=True,
    )


# ---------------------------------------------------------------------- #
# Fixture builders                                                         #
# ---------------------------------------------------------------------- #


def _pre_anchor_m5(anchor: datetime) -> list[Bar]:
    """M5 bars covering [anchor - PRE_HOURS, anchor) UTC.  Tight oscillation
    around 2340 — ATR settles small, OR-range filter relaxed in config."""
    bars: list[Bar] = []
    n = PRE_HOURS * 12  # 12 M5 bars/hour
    mid = 2340.0
    for i in range(n):
        t = anchor - timedelta(minutes=5 * (n - i))
        c = mid + (0.2 if i % 2 == 0 else -0.2)
        bars.append(_bar(
            tf=Timeframe.M5, t=t, o=mid, h=mid + 0.4, lo=mid - 0.4, c=c,
        ))
    return bars


def _or_window_m1(anchor: datetime, *, high: float, low: float) -> list[Bar]:
    """15 M1 bars covering [anchor, anchor + 15m) — builds the OR."""
    bars: list[Bar] = []
    mid = (high + low) / 2
    for i in range(15):
        t = anchor + timedelta(minutes=i)
        if i == 5:
            h, lo, c = high, mid - 0.5, mid + 0.5
        elif i == 10:
            h, lo, c = mid + 0.5, low, mid - 0.5
        else:
            h, lo, c = mid + 0.5, mid - 0.5, mid + (0.1 if i % 2 == 0 else -0.1)
        bars.append(_bar(tf=Timeframe.M1, t=t, o=mid, h=h, lo=lo, c=c))
    return bars


def _or_window_m5(anchor: datetime, *, high: float, low: float) -> list[Bar]:
    """3 M5 bars covering the OR window [anchor, anchor + 15m).  These
    M5 bars sit INSIDE the OR (none confirms a breakout)."""
    bars: list[Bar] = []
    mid = (high + low) / 2
    for i in range(3):
        t = anchor + timedelta(minutes=5 * i)
        c = mid + (0.1 if i % 2 == 0 else -0.1)
        h = high if i == 1 else mid + 0.5    # touch high once
        lo = low if i == 2 else mid - 0.5    # touch low once
        bars.append(_bar(tf=Timeframe.M5, t=t, o=mid, h=h, lo=lo, c=c))
    return bars


def _breakout_m1(
    anchor: datetime, *, or_high: float, breakout_to: float,
) -> list[Bar]:
    """M1 bars from anchor+15m onwards that monotonically climb past
    or_high, then stabilise at breakout_to.

    Specifically:
      * 15..24: climb from or_high to breakout_to over 10 bars
      * 25..end: stable around breakout_to so SL/TP don't fire early
    """
    bars: list[Bar] = []
    total = RUN_HOURS * 60
    climb_end = 25
    step = (breakout_to - or_high) / (climb_end - 15)
    for i in range(15, total):
        t = anchor + timedelta(minutes=i)
        if i < climb_end:
            price = or_high + step * (i - 15)
            bars.append(_bar(
                tf=Timeframe.M1, t=t,
                o=price - step / 2, h=price + 0.2, lo=price - 0.3,
                c=price + step / 2,
            ))
        else:
            c = breakout_to + (0.05 if i % 2 == 0 else -0.05)
            bars.append(_bar(
                tf=Timeframe.M1, t=t,
                o=breakout_to, h=breakout_to + 0.2, lo=breakout_to - 0.2, c=c,
            ))
    return bars


def _breakout_m5(
    anchor: datetime, *, or_high: float, breakout_to: float,
) -> list[Bar]:
    """M5 bars from anchor+15m onwards.  The bar at anchor+15m
    (covering [15, 20)) closes ABOVE or_high — this is the
    confirmation candle that lets the strategy fire."""
    bars: list[Bar] = []
    n = RUN_HOURS * 12   # 12 M5/hour
    step_per_bar = (breakout_to - or_high) / 2   # 2 climbing M5 bars
    for i in range(3, n):   # start at i=3 (= anchor+15m)
        t = anchor + timedelta(minutes=5 * i)
        if i == 3:
            # First post-OR M5: closes solidly above or_high.
            close = or_high + step_per_bar
            bars.append(_bar(
                tf=Timeframe.M5, t=t, o=or_high, h=close + 0.3,
                lo=or_high - 0.2, c=close,
            ))
        elif i == 4:
            close = breakout_to
            prev_close = or_high + step_per_bar
            bars.append(_bar(
                tf=Timeframe.M5, t=t, o=prev_close, h=close + 0.3,
                lo=prev_close - 0.2, c=close,
            ))
        else:
            c = breakout_to + (0.1 if i % 2 == 0 else -0.1)
            bars.append(_bar(
                tf=Timeframe.M5, t=t, o=breakout_to,
                h=breakout_to + 0.3, lo=breakout_to - 0.3, c=c,
            ))
    return bars


def _no_breakout_m1(anchor: datetime, *, or_high: float, or_low: float) -> list[Bar]:
    """M1 bars after the OR window that stay strictly INSIDE the range
    — no breakout fires."""
    bars: list[Bar] = []
    total = RUN_HOURS * 60
    mid = (or_high + or_low) / 2
    for i in range(15, total):
        t = anchor + timedelta(minutes=i)
        c = mid + (0.2 if i % 2 == 0 else -0.2)
        bars.append(_bar(
            tf=Timeframe.M1, t=t,
            o=mid, h=mid + 0.5, lo=mid - 0.5, c=c,
        ))
    return bars


def _no_breakout_m5(anchor: datetime, *, or_high: float, or_low: float) -> list[Bar]:
    """M5 bars after the OR window — also stay inside the range."""
    bars: list[Bar] = []
    n = RUN_HOURS * 12
    mid = (or_high + or_low) / 2
    for i in range(3, n):
        t = anchor + timedelta(minutes=5 * i)
        c = mid + (0.1 if i % 2 == 0 else -0.1)
        bars.append(_bar(
            tf=Timeframe.M5, t=t, o=mid, h=mid + 0.4, lo=mid - 0.4, c=c,
        ))
    return bars


def _seed(root: Path, m1: list[Bar], m5: list[Bar]) -> None:
    store = ParquetStore(root)
    store.append_bars(SYMBOL, Timeframe.M1, m1)
    store.append_bars(SYMBOL, Timeframe.M5, m5)


def _build_cfg(parquet_root: Path, anchor: datetime) -> BacktestRunConfig:
    return BacktestRunConfig(
        id="orb_smoke",
        mode="file",
        strategy_id="orb",
        symbols=[SYMBOL],
        timeframes=[Timeframe.M1, Timeframe.M5],
        # Start = 3h before anchor so the pre-roll M5 bars are loaded.
        start=anchor - timedelta(hours=PRE_HOURS),
        end=anchor + timedelta(hours=RUN_HOURS),
        data_source=parquet_root,
        initial_balance=10_000.0,
    )


def _build_entry() -> StrategyEntry:
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
            "min_or_range_atr": 0.1,
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
async def test_strategy_takes_breakout_trade_after_or_freeze(tmp_path: Path) -> None:
    """OR builds from M1 bars in [07:00, 07:15); first post-OR M5 closes
    above or_high; first post-OR M1 with close > or_high fires a BUY."""
    anchor = datetime(2024, 1, 1, SESSION_HOUR, 0, tzinfo=UTC)
    or_high, or_low = 2342.0, 2338.0
    breakout_to = 2345.0
    root = tmp_path / "parquet"
    m1 = (
        _or_window_m1(anchor, high=or_high, low=or_low)
        + _breakout_m1(anchor, or_high=or_high, breakout_to=breakout_to)
    )
    m5 = (
        _pre_anchor_m5(anchor)
        + _or_window_m5(anchor, high=or_high, low=or_low)
        + _breakout_m5(anchor, or_high=or_high, breakout_to=breakout_to)
    )
    _seed(root, m1, m5)

    bt = FileBacktester(
        strategy=_build_entry(),
        parquet_root=root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "reports",
    )
    report = await bt.run(_build_cfg(root, anchor))

    assert len(report.trades) >= 1, (
        f"expected at least one breakout trade after OR freeze — "
        f"got {report.trades}"
    )
    assert report.trades[0].side == "buy"


@pytest.mark.asyncio
async def test_strategy_skips_when_no_breakout_occurs(tmp_path: Path) -> None:
    """Same OR window, but post-OR bars stay inside the range — no
    M1 close pierces or_high/or_low → zero trades."""
    anchor = datetime(2024, 1, 1, SESSION_HOUR, 0, tzinfo=UTC)
    or_high, or_low = 2342.0, 2338.0
    root = tmp_path / "parquet"
    m1 = (
        _or_window_m1(anchor, high=or_high, low=or_low)
        + _no_breakout_m1(anchor, or_high=or_high, or_low=or_low)
    )
    m5 = (
        _pre_anchor_m5(anchor)
        + _or_window_m5(anchor, high=or_high, low=or_low)
        + _no_breakout_m5(anchor, or_high=or_high, or_low=or_low)
    )
    _seed(root, m1, m5)

    bt = FileBacktester(
        strategy=_build_entry(),
        parquet_root=root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "reports",
    )
    report = await bt.run(_build_cfg(root, anchor))

    assert report.trades == [], (
        f"no breakout occurred — must not trade; got {report.trades}"
    )
