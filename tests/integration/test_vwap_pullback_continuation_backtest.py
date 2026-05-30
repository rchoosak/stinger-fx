"""End-to-end multi-feed backtest for VwapPullbackContinuation.

The strategy needs three timeframes wired through the backtester:

  * M15 for the ADX regime gate (must be TRENDING) + EMA bias
  * M5 for the session VWAP anchor, slope check, swing-low SL, and
    Donchian prev-extreme TP
  * M1 for the bullish rejection candle that triggers entry

This file seeds synthetic XAU/USD bars at all three timeframes into a
Parquet root, hands them to ``FileBacktester`` via the multi-feed
``symbols=["XAUUSD"], timeframes=[M1, M5, M15]`` shape, and asserts:

  * In a *trending* M15 regime with an M5 pullback to VWAP, the strategy
    takes at least one continuation trade when an M1 bullish rejection
    candle prints inside the pullback zone.
  * In a *ranging* M15 regime the TrendingFilter blocks all entries
    (ADX < ``min_adx``) — zero trades regardless of M1/M5 setup.

The exact P&L isn't asserted; this verifies wiring (multi-TF dispatch +
regime gate + entry trigger + SL/TP placement), not profitability under
synthetic data.
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
WARMUP_HOURS = 4  # M5 = 48 bars, M15 = 16 bars — plenty for adx_period=5
# Trigger lands at 3:55 — the last M1 of the run after enough M5/M15
# history has built up.
TRIGGER_AT_MINUTE = 3 * 60 + 55


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
# Fixture builders                                                        #
# ---------------------------------------------------------------------- #


def _trending_up_m15(base: datetime) -> list[Bar]:
    """Strong monotonic uptrend over 4h (16 M15 bars). ADX(5) settles
    high above min_adx=18 and EMA(4) > EMA(10)."""
    bars: list[Bar] = []
    price = 1990.0
    for i in range(WARMUP_HOURS * 4):  # 4 M15 bars per hour
        bars.append(_bar(
            tf=Timeframe.M15,
            t=base + timedelta(minutes=15 * i),
            o=price, h=price + 1.0, lo=price - 0.2, c=price + 0.9,
        ))
        price += 1.0
    return bars


def _ranging_m15(base: datetime) -> list[Bar]:
    """Tiny-range alternating closes → ADX(5) stays below min_adx →
    TrendingFilter blocks entries."""
    bars: list[Bar] = []
    mid = 2010.0
    for i in range(WARMUP_HOURS * 4):
        c = mid + (0.05 if i % 2 == 0 else -0.05)
        bars.append(_bar(
            tf=Timeframe.M15,
            t=base + timedelta(minutes=15 * i),
            o=mid, h=mid + 0.2, lo=mid - 0.2, c=c,
        ))
    return bars


def _climbing_then_pullback_m5(base: datetime) -> list[Bar]:
    """48 M5 bars over 4h.  Bars 0..35 (3h) climb steadily by +1 USD
    per bar.  Bars 36..47 (1h) pull back hard by −1.5 USD per bar.

    The pullback drives recent M5 lows down through where the M1
    trigger close will land, so the long-side SL (swing_low − ATR
    buffer) sits below the entry — a valid trade geometry.
    """
    bars: list[Bar] = []
    # --- Climb phase: prices 1990 → 2025 ----------------------------
    price = 1990.0
    for i in range(36):
        bars.append(_bar(
            tf=Timeframe.M5,
            t=base + timedelta(minutes=5 * i),
            o=price, h=price + 1.0, lo=price - 0.5, c=price + 0.8,
        ))
        price += 1.0
    # --- Pullback phase: prices 2026 → 2009 -------------------------
    for i in range(12):
        bars.append(_bar(
            tf=Timeframe.M5,
            t=base + timedelta(minutes=5 * (36 + i)),
            o=price, h=price + 0.5, lo=price - 2.0, c=price - 1.5,
        ))
        price -= 1.5
    return bars


def _m1_climb_then_rejection(base: datetime) -> list[Bar]:
    """M1 bars: boring climb for ~3.5h, then a single bullish rejection
    candle at TRIGGER_AT_MINUTE with a long lower wick and a close near
    the M5-pulled-back price level.

    The trigger close (~2011) lands inside the pullback zone above the
    M5 session VWAP (which sits around 2010 thanks to the climb+pullback
    profile) with a long lower wick → ``_is_bullish_rejection`` returns
    True → strategy fires BUY.
    """
    bars: list[Bar] = []
    n = WARMUP_HOURS * 60  # 240 M1 bars
    for i in range(n):
        t = base + timedelta(minutes=i)
        if i == TRIGGER_AT_MINUTE:
            # Hammer-shape: open=2010.9, close=2011.0, high=2011.1,
            # low=2009.5 → body=0.1, lower_wick=1.4 → ratio=14.
            bars.append(_bar(
                tf=Timeframe.M1, t=t,
                o=2010.9, h=2011.1, lo=2009.5, c=2011.0,
            ))
        else:
            # Boring climbing path through the M5 trend; small bars.
            base_price = 1990.0 + (i / (n - 1)) * 20.0  # 1990 → 2010
            c = base_price + (0.05 if i % 2 == 0 else -0.05)
            bars.append(_bar(
                tf=Timeframe.M1, t=t,
                o=base_price, h=base_price + 0.2, lo=base_price - 0.2, c=c,
            ))
    return bars


def _seed(root: Path, m1: list[Bar], m5: list[Bar], m15: list[Bar]) -> None:
    store = ParquetStore(root)
    store.append_bars(SYMBOL, Timeframe.M1, m1)
    store.append_bars(SYMBOL, Timeframe.M5, m5)
    store.append_bars(SYMBOL, Timeframe.M15, m15)


def _build_cfg(parquet_root: Path, base: datetime) -> BacktestRunConfig:
    return BacktestRunConfig(
        id="vwap_pullback_smoke",
        mode="file",
        strategy_id="vwap_pullback",
        symbols=[SYMBOL],
        timeframes=[Timeframe.M1, Timeframe.M5, Timeframe.M15],
        start=base,
        end=base + timedelta(hours=WARMUP_HOURS),
        data_source=parquet_root,
        initial_balance=10_000.0,
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
            "vwap_slope_min_atr": 0.02,
            "pullback_zone_atr": 1.0,    # generous zone for synthetic data
            "rejection_wick_ratio": 2.0,
            "atr_period": 5,
            "min_atr": 0.05,
            "swing_lookback": 5,
            "stop_buffer_atr": 0.2,
            "prev_extreme_lookback": 10,
            "tp_mode": "prev_extreme",
            "min_rr": 0.3,
            # Session anchor at UTC 0 matches our base time.
            "use_session_filter": False,
            "cooldown_bars": 0,
        },
    )


# ---------------------------------------------------------------------- #
# Tests                                                                  #
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_strategy_takes_continuation_trade_in_uptrend(tmp_path: Path) -> None:
    """Trending M15 + climbing-then-pulling-back M5 + bullish-rejection
    M1 → at least one BUY trade fires."""
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    root = tmp_path / "parquet"
    _seed(
        root,
        _m1_climb_then_rejection(base),
        _climbing_then_pullback_m5(base),
        _trending_up_m15(base),
    )

    bt = FileBacktester(
        strategy=_build_entry(),
        parquet_root=root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "reports",
    )
    report = await bt.run(_build_cfg(root, base))

    assert len(report.trades) >= 1, (
        f"expected at least one continuation trade in trending regime — "
        f"got {report.trades}"
    )
    # Continuation strategy in an uptrend → first trade must be BUY.
    assert report.trades[0].side == "buy"


@pytest.mark.asyncio
async def test_strategy_takes_no_trades_in_ranging_regime(tmp_path: Path) -> None:
    """Same M1/M5 setup, but M15 ADX stays below min_adx → TrendingFilter
    blocks all entries → zero trades."""
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    root = tmp_path / "parquet"
    _seed(
        root,
        _m1_climb_then_rejection(base),
        _climbing_then_pullback_m5(base),
        _ranging_m15(base),
    )

    bt = FileBacktester(
        strategy=_build_entry(),
        parquet_root=root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "reports",
    )
    report = await bt.run(_build_cfg(root, base))

    assert report.trades == [], (
        f"ranging M15 regime must block all continuation entries; got "
        f"{report.trades}"
    )
