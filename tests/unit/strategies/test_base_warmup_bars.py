"""Unit tests for ``BaseStrategy.warmup_bars()`` (Plan A2).

The classmethod tells the live runtime how much historical tick
history to backfill per (symbol, tf) before opening the live tick
pump — see ``stinger_fx.runtime._backfill_aggregator``.

These tests pin:
  * The base class returns ``None`` so existing strategies opt into
    the 48-hour conservative default automatically.
  * Subclass overrides return the expected dict for runtime to read.
  * The 3 example strategies (LSR / VPC / ORB) declare warmup correctly
    based on their actual indicator periods + session anchors.
"""

from __future__ import annotations

from stinger_fx.domain import Subscription, Timeframe
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.examples.liquidity_sweep_reversal import (
    LiquiditySweepReversal,
    LiquiditySweepReversalParams,
)
from stinger_fx.strategies.examples.opening_range_breakout import (
    OpeningRangeBreakout,
    OpeningRangeBreakoutParams,
)
from stinger_fx.strategies.examples.vwap_pullback_continuation import (
    VwapPullbackContinuation,
    VwapPullbackContinuationParams,
)
from stinger_fx.strategies.parameters import StrategyParams


def test_base_warmup_bars_returns_none_by_default() -> None:
    """Strategies that don't override ``warmup_bars`` get None, which
    the runtime translates to the 48-hour default backfill."""
    assert BaseStrategy.warmup_bars(StrategyParams()) is None


def test_lsr_warmup_declares_per_feed_windows() -> None:
    params = LiquiditySweepReversalParams()
    declared = LiquiditySweepReversal.warmup_bars(params)
    assert declared is not None
    m1_sub = Subscription(symbol=params.symbol, timeframe=Timeframe.M1)
    m5_sub = Subscription(symbol=params.symbol, timeframe=Timeframe.M5)
    m15_sub = Subscription(symbol=params.symbol, timeframe=Timeframe.M15)
    # M1: entry trigger reads the single incoming bar
    assert declared[m1_sub] == 1
    # M5: max(range_lookback_bars=36, atr_period+1=15) = 36
    assert declared[m5_sub] == max(
        params.range_lookback_bars, params.atr_period + 1,
    )
    # M15: 2 × adx_period = 28 bars before ADX returns non-None
    assert declared[m15_sub] == 2 * params.adx_period


def test_vpc_warmup_covers_session_vwap_window() -> None:
    """VPC's M5 warmup must cover at least 24h so the session VWAP
    starts the first session warm — indicator-only window would be
    too small (only ~25 bars vs the 288 needed for a full UTC day)."""
    params = VwapPullbackContinuationParams()
    declared = VwapPullbackContinuation.warmup_bars(params)
    assert declared is not None
    m5_sub = Subscription(symbol=params.symbol, timeframe=Timeframe.M5)
    # 24h × 12 M5 bars/hour = 288 — must dominate over the
    # indicator-only window.
    assert declared[m5_sub] >= 288


def test_orb_warmup_is_compact() -> None:
    """ORB needs minimal warmup — opening range only spans 15 minutes
    and the M5 ATR window is short.  Plan A2 explicitly highlighted
    ORB as the strategy that should NOT pay the 48h default."""
    params = OpeningRangeBreakoutParams()
    declared = OpeningRangeBreakout.warmup_bars(params)
    assert declared is not None
    m1_sub = Subscription(symbol=params.symbol, timeframe=Timeframe.M1)
    m5_sub = Subscription(symbol=params.symbol, timeframe=Timeframe.M5)
    # M1 at least one OR window
    assert declared[m1_sub] >= params.opening_range_minutes
    # M5 just ATR warmup
    assert declared[m5_sub] == params.atr_period + 1
    # Total ATR-tick budget should be < 24h (proves the strategy isn't
    # paying for VPC-sized session-VWAP backfill).
    total_seconds = (
        declared[m1_sub] * Timeframe.M1.seconds
        + declared[m5_sub] * Timeframe.M5.seconds
    )
    assert total_seconds < 24 * 3600
