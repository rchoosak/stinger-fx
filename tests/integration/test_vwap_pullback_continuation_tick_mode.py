"""End-to-end **tick-mode** backtest for VwapPullbackContinuation.

Companion to ``test_vwap_pullback_continuation_backtest.py`` (bar mode).
The bar-mode test seeds three separate Parquet datasets (M1 / M5 / M15)
and relies on ``FileBacktester._replay_bars`` to merge feeds. This file
seeds a **single XAUUSD tick stream**, runs with ``granularity="tick"``,
and verifies the BarAggregator builds M1, M5, and M15 bars from that
one stream — proving the strategy works with the tick-data workflow
operators are likely to use in pre-live validation.

Pre-fix risk this test guards against
=====================================

The strategy code is mode-agnostic (it calls ``ctx.history_for(symbol,
tf).bars()`` and trusts the runner to populate). Tick-mode replay
creates one ``BarAggregator`` per declared subscription — for this
strategy, three aggregators on the same tick stream — and the merger
of those three aggregated bar streams must satisfy three contracts
simultaneously:

  * The M15 aggregator delivers enough closed bars for ``adx(5)`` plus
    ``ema(10)`` so the regime gate + EMA bias can evaluate.
  * The M5 aggregator delivers enough closed bars for
    ``donchian(prev_extreme_lookback)`` + ``atr(5)`` + ``vwap_session``
    + the swing-low scan.
  * The M1 aggregator delivers the *exact* trigger candle whose open is
    near VWAP, low pierces down for the wick, and close returns above
    open — the bullish rejection has to print on a specific minute.

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

# Reduced-warmup test params:
#   * adx_period=5 → M15 ADX needs 2×5=10 closed M15 bars = 2.5h
#   * prev_extreme_lookback=10, atr_period=5 → M5 needs at most 11 bars
#
# We seed 4h of session so the strategy has comfortable headroom on both
# warmups before the trigger candle fires at minute 235 (03:55). Tick
# spacing is 20s (3 ticks per minute) so the trigger M1 bar can carry a
# proper open/low/close shape (open ≠ low ≠ close) that the rejection-
# candle classifier needs.
WARMUP_HOURS = 4
TRIGGER_AT_MINUTE = 3 * 60 + 55     # trigger at 03:55
TICK_SPACING = timedelta(seconds=20)
SPREAD = 0.20                       # XAU spread typical ≈ 20 cents


def _seed_tick_stream(root: Path, ticks: list[Tick]) -> None:
    store = ParquetStore(root)
    store.append_ticks(SYMBOL, ticks)


def _tick(t: datetime, bid: float) -> Tick:
    return Tick(symbol=SYMBOL, time=t, bid=bid, ask=bid + SPREAD)


# ---------------------------------------------------------------------- #
# Warmup tick generators                                                   #
# ---------------------------------------------------------------------- #


def _uptrending_warmup_ticks(base: datetime) -> list[Tick]:
    """Strong monotonic uptrend — each minute prints 3 ticks that climb
    by 1 USD overall (0.5 USD per tick step).  M15 closes therefore
    climb by ~15 USD per bar → ADX(5) well above min_adx=18, EMA(4) >
    EMA(10).  VWAP also rises monotonically with the climb."""
    ticks: list[Tick] = []
    price = 1990.0
    for minute in range(TRIGGER_AT_MINUTE):
        t0 = base + timedelta(minutes=minute)
        t1 = t0 + TICK_SPACING
        t2 = t1 + TICK_SPACING
        # Three ticks: opens at `price`, mid step, climbs to `price + 1`.
        ticks.append(_tick(t0, price))
        ticks.append(_tick(t1, price + 0.5))
        ticks.append(_tick(t2, price + 1.0))
        price += 1.0
    return ticks


def _sideway_warmup_ticks(base: datetime) -> list[Tick]:
    """Tight oscillation around mid → M15 ADX stays low.  Three ticks
    per minute alternate above/below mid by small amounts."""
    ticks: list[Tick] = []
    mid = 2010.0
    for minute in range(TRIGGER_AT_MINUTE):
        t0 = base + timedelta(minutes=minute)
        t1 = t0 + TICK_SPACING
        t2 = t1 + TICK_SPACING
        offset = 0.2 if minute % 2 == 0 else -0.2
        ticks.append(_tick(t0, mid + offset))
        ticks.append(_tick(t1, mid))
        ticks.append(_tick(t2, mid - offset))
    return ticks


# ---------------------------------------------------------------------- #
# Trigger + trailing ticks                                                 #
# ---------------------------------------------------------------------- #


def _rejection_and_trail_ticks(
    base: datetime, *, anchor_price: float,
) -> list[Tick]:
    """At ``TRIGGER_AT_MINUTE`` produce a 3-tick M1 bar whose shape
    satisfies ``_is_bullish_rejection``:

      * Tick #1 at xx:00 = ``anchor_price``         → bar.open / bar.high
      * Tick #2 at xx:20 = ``anchor_price − 2.0``   → bar.low (the wick)
      * Tick #3 at xx:40 = ``anchor_price + 0.1``   → bar.close

    body = |close − open| = 0.1, lower_wick = open − low = 2.0,
    upper_wick = high − max(open, close) = 0 → ratio = 20 → easily
    satisfies the default rejection_wick_ratio=2.0.

    Followed by 5 minutes (15 ticks) of mild trailing oscillation so
    the strategy has bars after the trigger to manage the position
    without interference.
    """
    ticks: list[Tick] = []
    t_trig0 = base + timedelta(minutes=TRIGGER_AT_MINUTE)
    t_trig1 = t_trig0 + TICK_SPACING
    t_trig2 = t_trig1 + TICK_SPACING
    ticks.append(_tick(t_trig0, anchor_price))
    ticks.append(_tick(t_trig1, anchor_price - 2.0))   # wick down
    ticks.append(_tick(t_trig2, anchor_price + 0.1))   # close above open
    # Trailing minutes — keep price near anchor so SL/TP don't fire.
    for minute in range(1, 6):
        t0 = base + timedelta(minutes=TRIGGER_AT_MINUTE + minute)
        t1 = t0 + TICK_SPACING
        t2 = t1 + TICK_SPACING
        ticks.append(_tick(t0, anchor_price + 0.1))
        ticks.append(_tick(t1, anchor_price))
        ticks.append(_tick(t2, anchor_price + 0.05))
    return ticks


# ---------------------------------------------------------------------- #
# Config helpers                                                            #
# ---------------------------------------------------------------------- #


def _build_cfg(parquet_root: Path, base: datetime) -> BacktestRunConfig:
    return BacktestRunConfig(
        id="vwap_pullback_tick_smoke",
        mode="file",
        strategy_id="vwap_pullback",
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
        id="vwap_pullback",
        class_path=(
            "stinger_fx.strategies.examples.vwap_pullback_continuation"
            ":VwapPullbackContinuation"
        ),
        enabled=True,
        params={
            "symbol": SYMBOL,
            "entry_timeframe": "M1",
            "structure_timeframe": "M5",
            "regime_timeframe": "M15",
            # Reduced warmup periods so 4h of session is enough.
            "adx_period": 5,
            "min_adx": 18.0,
            "ema_fast": 4,
            "ema_slow": 10,
            "vwap_slope_lookback": 3,
            "vwap_slope_min_atr": 0.01,
            # Generous pullback zone — the synthetic uptrend keeps the
            # entry close well above VWAP because VWAP lags the climb;
            # we want the gate to accept the trigger anyway.  In a real
            # market the trigger would land near VWAP after a deeper
            # pullback; here we're testing wiring, not entry geometry.
            "pullback_zone_atr": 50.0,
            "rejection_wick_ratio": 2.0,
            "atr_period": 5,
            "min_atr": 0.05,
            "swing_lookback": 5,
            "stop_buffer_atr": 0.2,
            "prev_extreme_lookback": 10,
            "tp_mode": "fixed_r",
            "take_profit_r": 1.0,
            "min_rr": 0.1,
            # Disable session-of-day filter; our base time is UTC 00:00.
            "use_session_filter": False,
            "cooldown_bars": 0,
        },
    )


# ---------------------------------------------------------------------- #
# Test cases                                                                #
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_tick_mode_takes_continuation_trade_in_uptrend(tmp_path: Path) -> None:
    """The tick stream encodes ~4 hours of monotonic uptrend, then a
    3-tick rejection candle near the latest price.  The BarAggregator
    must produce a closed M1 bar with the right open/high/low/close
    geometry, the M5/M15 aggregators must keep up with regime + VWAP
    indicators, and the strategy must fire a BUY off the resulting
    BarEvent."""
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    root = tmp_path / "parquet"
    # Anchor the rejection candle near the last climbed price so the
    # M5 swing low (recent climb lows) sits well below it — keeps SL
    # comfortably below entry.
    anchor = 1990.0 + TRIGGER_AT_MINUTE * 1.0  # last climb price
    ticks = (
        _uptrending_warmup_ticks(base)
        + _rejection_and_trail_ticks(base, anchor_price=anchor)
    )
    _seed_tick_stream(root, ticks)

    bt = FileBacktester(
        strategy=_build_entry(),
        parquet_root=root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "reports",
    )
    report = await bt.run(_build_cfg(root, base))

    assert len(report.trades) >= 1, (
        f"expected at least one continuation trade in tick-mode uptrend — "
        f"got trades={report.trades}. The aggregator + multi-TF dispatch "
        f"must build M1/M5/M15 bars from the same tick stream and let "
        f"the strategy see the rejection candle."
    )
    # First trade must be the BUY setup (rejection in an uptrend).
    assert report.trades[0].side == "buy"


@pytest.mark.asyncio
async def test_tick_mode_blocks_trades_in_ranging_regime(tmp_path: Path) -> None:
    """Same rejection candle, but warmup ticks oscillate around a fixed
    mid so M15 ADX stays below ``min_adx`` (default 18).  The regime
    gate must block all entries — proves the gate fires correctly when
    the M15 aggregator sees a flat close series, not just when M15 is
    pre-seeded from Parquet."""
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    root = tmp_path / "parquet"
    # Anchor near the sideway mid so the trigger lands inside the
    # synthetic range (though the gate blocks regardless).
    ticks = (
        _sideway_warmup_ticks(base)
        + _rejection_and_trail_ticks(base, anchor_price=2010.0)
    )
    _seed_tick_stream(root, ticks)

    bt = FileBacktester(
        strategy=_build_entry(),
        parquet_root=root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "reports",
    )
    report = await bt.run(_build_cfg(root, base))

    assert report.trades == [], (
        f"ranging M15 regime (aggregated from oscillating ticks) "
        f"must block all continuation entries; got {report.trades}"
    )
