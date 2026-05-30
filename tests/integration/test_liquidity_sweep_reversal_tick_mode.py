"""End-to-end **tick-mode** backtest for LiquiditySweepReversal.

Companion to ``test_liquidity_sweep_reversal_backtest.py`` (bar mode).
That test seeds three separate Parquet datasets (M1 / M5 / M15) and
relies on ``FileBacktester._replay_bars`` to merge feeds. This file
seeds a **single XAUUSD tick stream**, runs with ``granularity="tick"``,
and verifies the BarAggregator builds M1, M5, and M15 bars from that
one stream — proving the strategy works with the tick-data workflow
operators are likely to use in pre-live validation.

Pre-fix risk this test guards against
=====================================

The strategy code is mode-agnostic (it calls ``ctx.history_for(symbol,
tf).bars()`` and trusts the runner to populate). Tick-mode replay
(``_replay_ticks``, Phase 4 PR #19) creates one ``BarAggregator`` per
declared subscription — for this strategy, three aggregators on the
same tick stream — and the merger of those three aggregated bar streams
must satisfy three contracts simultaneously:

  * The M15 aggregator delivers enough closed bars for ``adx(14)``
    before the sweep moment so the regime gate has data to evaluate.
  * The M5 aggregator delivers enough closed bars for
    ``donchian(range_lookback_bars)`` plus ``atr(14)``.
  * The M1 aggregator delivers the *exact* sweep candle whose high
    pierces the range and whose close re-enters — the trigger has to
    happen at a specific minute, not "around" it.

Any future change to ``BarAggregator``'s boundary semantics that breaks
one of those three contracts trips this test, even if the bar-mode
integration test still passes (because bar mode reads bars directly
from Parquet and doesn't exercise the aggregator at all).
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
RANGE_HIGH = 2005.0
RANGE_LOW = 1995.0
RANGE_MID = (RANGE_HIGH + RANGE_LOW) / 2

# Reduced-warmup test params:
#   * adx_period=5 → M15 ADX needs 2×5=10 closed M15 bars = 2.5h
#   * range_lookback_bars=10, atr_period=5 → M5 needs at most 11 bars
#
# We seed 4h of session so the strategy has comfortable headroom on both
# warmups before the sweep candle fires at 03:00. Tick spacing is 30s
# (120 ticks/hour) so each M1 bar gets 2 ticks → high/low form an
# oscillation that the M5 Donchian can pick up as a range.
WARMUP_HOURS = 4
SWEEP_AT_MINUTE = 3 * 60      # sweep candle at 03:00
TICK_SPACING = timedelta(seconds=30)
SPREAD = 0.20                  # XAU spread typical ≈ 20 cents


def _seed_tick_stream(
    root: Path, ticks: list[Tick],
) -> None:
    store = ParquetStore(root)
    store.append_ticks(SYMBOL, ticks)


def _tick(t: datetime, bid: float) -> Tick:
    return Tick(symbol=SYMBOL, time=t, bid=bid, ask=bid + SPREAD)


def _ranging_warmup_ticks(base: datetime) -> list[Tick]:
    """Generate ticks for the warmup period. Two ticks per minute, mid-
    band oscillation that keeps M15 ADX low and M5 Donchian seeing the
    full [RANGE_LOW, RANGE_HIGH] band.

    Pattern:
      * Even minutes  : tick1=MID+1, tick2=MID-1
      * Odd minutes   : tick1=MID-1, tick2=MID+1
      * Every 6th minute reaches RANGE_HIGH so Donchian sees the top
      * Every 6th minute (offset 3) reaches RANGE_LOW so it sees the bottom

    Result: M1 bars wiggle ~1 USD around MID; M5 bars span the full
    range every ~30 minutes; M15 closes oscillate within ~1 USD of MID
    keeping ADX comfortably below max_adx=22.
    """
    ticks: list[Tick] = []
    for minute in range(SWEEP_AT_MINUTE):
        t0 = base + timedelta(minutes=minute)
        t1 = t0 + TICK_SPACING
        if minute % 6 == 0:
            # Touch the top of the range
            bid1, bid2 = RANGE_HIGH - 0.5, RANGE_MID + 0.5
        elif minute % 6 == 3:
            # Touch the bottom
            bid1, bid2 = RANGE_LOW + 0.5, RANGE_MID - 0.5
        else:
            # Boring oscillation
            if minute % 2 == 0:
                bid1, bid2 = RANGE_MID + 1.0, RANGE_MID - 1.0
            else:
                bid1, bid2 = RANGE_MID - 1.0, RANGE_MID + 1.0
        ticks.append(_tick(t0, bid1))
        ticks.append(_tick(t1, bid2))
    return ticks


def _trending_warmup_ticks(base: datetime) -> list[Tick]:
    """Strong monotonic uptrend — each M1 bar prints a higher close than
    the last. M15 closes climb linearly so ADX rises above max_adx and
    the regime gate must reject all sweep attempts.

    To get ADX high enough fast, we use a deterministic ramp where every
    minute closes ~1 USD above the previous (M1 climb is steady, M15
    closes therefore climb by ~15 USD per bar — that's a screaming
    trend, ADX well above 25)."""
    ticks: list[Tick] = []
    price = RANGE_MID
    for minute in range(SWEEP_AT_MINUTE):
        t0 = base + timedelta(minutes=minute)
        t1 = t0 + TICK_SPACING
        # Each minute: open near previous close, close ~1 USD higher
        ticks.append(_tick(t0, price))
        ticks.append(_tick(t1, price + 0.8))
        price += 1.0
    return ticks


def _sweep_and_trail_ticks(base: datetime) -> list[Tick]:
    """At ``SWEEP_AT_MINUTE`` produce a 2-tick M1 bar that satisfies the
    short-sweep geometry:

      * Tick #1 at 03:00:00 = RANGE_HIGH + 3.0   → becomes bar.high
      * Tick #2 at 03:00:30 = RANGE_HIGH - 3.0   → becomes bar.close

    Then 5 trailing minutes (2 ticks each) at mid so the strategy has
    a bar after the trigger to actually emit the signal + the time-exit
    window can do its bookkeeping without interference."""
    ticks: list[Tick] = []
    t_sweep0 = base + timedelta(minutes=SWEEP_AT_MINUTE)
    t_sweep1 = t_sweep0 + TICK_SPACING
    ticks.append(_tick(t_sweep0, RANGE_HIGH + 3.0))   # sweep peak
    ticks.append(_tick(t_sweep1, RANGE_HIGH - 3.0))   # re-entry

    # Trailing minutes back to mid to let the strategy book/manage exit.
    for minute in range(1, 6):
        t0 = base + timedelta(minutes=SWEEP_AT_MINUTE + minute)
        t1 = t0 + TICK_SPACING
        ticks.append(_tick(t0, RANGE_MID + 0.2))
        ticks.append(_tick(t1, RANGE_MID - 0.2))
    return ticks


def _build_cfg(parquet_root: Path, base: datetime) -> BacktestRunConfig:
    return BacktestRunConfig(
        id="sweep_reversal_tick_smoke",
        mode="file",
        strategy_id="sweep_reversal",
        symbols=[SYMBOL],
        timeframes=[Timeframe.M1, Timeframe.M5, Timeframe.M15],
        start=base,
        end=base + timedelta(hours=WARMUP_HOURS),
        data_source=parquet_root,
        initial_balance=10_000.0,
        granularity="tick",   # ← the line under test
    )


def _build_entry() -> StrategyEntry:
    return StrategyEntry(
        id="sweep_reversal",
        class_path=(
            "stinger_fx.strategies.examples.liquidity_sweep_reversal"
            ":LiquiditySweepReversal"
        ),
        enabled=True,
        params={
            "symbol": SYMBOL,
            "entry_timeframe": "M1",
            "structure_timeframe": "M5",
            "regime_timeframe": "M15",
            # Reduced warmup periods so 4h of session is enough.
            "range_lookback_bars": 10,
            "adx_period": 5,
            "atr_period": 5,
            # Tight buffers so the synthetic sweep candle triggers cleanly.
            "sweep_buffer_atr": 0.0,
            "reentry_buffer_atr": 0.0,
            "stop_buffer_atr": 0.2,
            "min_rr": 0.1,
            # Disable session-of-day filter; our base time is UTC 00:00.
            "use_session_filter": False,
            "tp_mode": "mid",
        },
    )


# ----------------------------------------------------------------------
# Test cases
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tick_mode_takes_reversal_trade_in_sideway(tmp_path: Path) -> None:
    """The tick stream encodes 3 hours of sideway oscillation, then a
    single 2-tick sweep candle that pokes above RANGE_HIGH and closes
    back inside. The BarAggregator must produce a closed M1 bar whose
    high > range_high and close <= range_high, and the strategy must
    fire a SHORT reversal off the resulting BarEvent."""
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    root = tmp_path / "parquet"
    ticks = _ranging_warmup_ticks(base) + _sweep_and_trail_ticks(base)
    _seed_tick_stream(root, ticks)

    bt = FileBacktester(
        strategy=_build_entry(),
        parquet_root=root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "reports",
    )
    report = await bt.run(_build_cfg(root, base))

    assert len(report.trades) >= 1, (
        f"expected at least one reversal trade in tick-mode sideway — "
        f"got trades={report.trades}. The aggregator + multi-TF dispatch "
        f"must build M1/M5/M15 bars from the same tick stream and let "
        f"the strategy see the sweep candle."
    )
    # First trade must be the SHORT setup (sweep over range_high).
    assert report.trades[0].side == "sell"


@pytest.mark.asyncio
async def test_tick_mode_blocks_trades_in_trending_regime(tmp_path: Path) -> None:
    """Same sweep candle, but the warmup ticks form a monotonic uptrend
    so M15 ADX rises above ``max_adx`` (default 22). The regime gate
    must block all entries — proves the gate fires correctly when the
    M15 aggregator sees a trending close series, not just when M15 is
    pre-seeded from Parquet."""
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    root = tmp_path / "parquet"
    ticks = _trending_warmup_ticks(base) + _sweep_and_trail_ticks(base)
    _seed_tick_stream(root, ticks)

    bt = FileBacktester(
        strategy=_build_entry(),
        parquet_root=root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "reports",
    )
    report = await bt.run(_build_cfg(root, base))

    assert report.trades == [], (
        f"trending M15 regime (aggregated from monotonic-uptrend ticks) "
        f"must block all sweep entries; got {report.trades}"
    )
