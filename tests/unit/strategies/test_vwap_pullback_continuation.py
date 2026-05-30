"""Unit tests for VwapPullbackContinuation.

Strategy is exercised directly: build a multi-feed ``StrategyContext``
with three HistoryViews (M1 / M5 / M15), pre-fill them with hand-
crafted bars, then call ``strat.on_bar(ctx, bar)`` and inspect the
signals captured through ``signal_sink``.

Test parameters use shrunk warmup periods (``adx_period=5``,
``ema_slow=10``, ``swing_lookback=5``, ``prev_extreme_lookback=10``,
``vwap_slope_lookback=3``, ``atr_period=5``) so the fixtures only need
~15 bars per timeframe.  The strategy logic is identical to production
defaults; only the warmup math changes.

Fixture shapes
==============

* **M15 uptrend** — monotonic climbing closes so ADX climbs above
  ``min_adx`` and EMA(fast) > EMA(slow) → ``bias_long``.
* **M15 sideway** — tiny-range alternating closes so ADX stays below
  ``min_adx`` and the regime gate blocks all entries.
* **M5 climb-then-pullback** — bars 0..10 climb steadily, bars 11..13
  pull back into the VWAP zone, the in-progress bar 14 is intentionally
  ignored by the strategy (``m5_bars[:-1]``).  The pullback drives M5
  swing_low below the eventual M1 trigger close so SL geometry is
  valid for a long entry.
* **M1 rejection trigger** — a single bar with a long lower wick
  (bullish) or long upper wick (bearish) that closes inside the
  pullback zone around VWAP.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
import structlog

from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.domain import (
    Bar,
    OrderType,
    Position,
    Side,
    Signal,
    Timeframe,
)
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.examples.vwap_pullback_continuation import (
    VwapPullbackContinuation,
    VwapPullbackContinuationParams,
    _is_bearish_rejection,
    _is_bullish_rejection,
)
from stinger_fx.strategies.managers.trailing import TrailingStopManager

SYMBOL = "XAUUSD"


def _ts(minutes_from_base: int) -> datetime:
    return datetime(2024, 1, 1, 0, 0, tzinfo=UTC) + timedelta(minutes=minutes_from_base)


def _bar(
    *, tf: Timeframe, t: datetime, o: float, h: float, lo: float, c: float,
    vol: int = 1000,
) -> Bar:
    return Bar(
        symbol=SYMBOL, timeframe=tf, time=t,
        open=o, high=h, low=lo, close=c,
        tick_volume=vol, is_closed=True,
    )


# ---------------------------------------------------------------------- #
# Reduced-warmup defaults used across every "happy path" test            #
# ---------------------------------------------------------------------- #

def _short_params(**overrides: object) -> VwapPullbackContinuationParams:
    """Default test params with warmup periods shrunk so fixtures stay
    small.  The session filter is disabled because our bars anchor at
    UTC 00:00 which is outside the production [08:00, 21:00) window —
    the session gate is tested separately."""
    base: dict[str, object] = dict(
        adx_period=5, min_adx=18.0,
        ema_fast=4, ema_slow=10,
        vwap_slope_lookback=3, vwap_slope_min_atr=0.05,
        pullback_zone_atr=0.5,
        rejection_wick_ratio=2.0,
        atr_period=5, min_atr=0.05,
        swing_lookback=5, stop_buffer_atr=0.3,
        tp_mode="prev_extreme", prev_extreme_lookback=10,
        take_profit_r=1.5, min_rr=0.5,
        cooldown_bars=0,
        use_session_filter=False,
    )
    base.update(overrides)
    return VwapPullbackContinuationParams(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------- #
# Fixture builders                                                        #
# ---------------------------------------------------------------------- #


def _build_uptrend_m15(n: int = 15, start: float = 2000.0, step: float = 1.0) -> list[Bar]:
    """Strong monotonic uptrend → ADX(5) climbs above 18 and EMA(4) >
    EMA(10) on the closes."""
    bars: list[Bar] = []
    price = start
    for i in range(n):
        bars.append(_bar(
            tf=Timeframe.M15, t=_ts(15 * i),
            o=price, h=price + 1.0, lo=price - 0.2, c=price + 0.9,
        ))
        price += step
    return bars


def _build_downtrend_m15(n: int = 15, start: float = 2014.0, step: float = 1.0) -> list[Bar]:
    """Monotonic downtrend — ADX(5) high, EMA(4) < EMA(10) → bias_short."""
    bars: list[Bar] = []
    price = start
    for i in range(n):
        bars.append(_bar(
            tf=Timeframe.M15, t=_ts(15 * i),
            o=price, h=price + 0.2, lo=price - 1.0, c=price - 0.9,
        ))
        price -= step
    return bars


def _build_sideway_m15(n: int = 15, mid: float = 2000.0) -> list[Bar]:
    """Tiny-range alternating M15 → ADX stays below min_adx, EMAs
    converge — TrendingFilter blocks entry."""
    bars: list[Bar] = []
    for i in range(n):
        c = mid + (0.05 if i % 2 == 0 else -0.05)
        bars.append(_bar(
            tf=Timeframe.M15, t=_ts(15 * i),
            o=mid, h=mid + 0.1, lo=mid - 0.1, c=c,
        ))
    return bars


def _build_climb_then_pullback_m5(
    *, climb_bars: int = 11, pullback_bars: int = 3, plus_one_in_progress: bool = True,
) -> list[Bar]:
    """Bars 0..climb_bars-1 climb from 2000 by +1 each.  Bars
    climb_bars..climb_bars+pullback_bars-1 pull back hard, dropping ~2
    per bar.  An extra in-progress trailing bar is appended last
    because the strategy slices ``m5_bars[:-1]`` and we want it to
    have data to ignore.

    Resulting closed_m5 (bars 0..climb+pullback-1):
      * VWAP slope positive (overall climb still dominates)
      * swing_low (last 5 bars) sits below the eventual M1 trigger so SL
        is valid for a long entry
      * donchian.upper covers the peak of the climb
    """
    bars: list[Bar] = []
    price = 2000.0
    for i in range(climb_bars):
        bars.append(_bar(
            tf=Timeframe.M5, t=_ts(5 * i),
            o=price, h=price + 1.0, lo=price - 0.5, c=price + 0.8,
        ))
        price += 1.0
    # Pullback — wide bars dropping price by 2 each.
    for i in range(pullback_bars):
        t = _ts(5 * (climb_bars + i))
        bars.append(_bar(
            tf=Timeframe.M5, t=t,
            o=price, h=price + 0.3, lo=price - 2.0, c=price - 1.5,
        ))
        price -= 2.0
    if plus_one_in_progress:
        # Trailing in-progress bar — strategy ignores it via [:-1] slice.
        t = _ts(5 * (climb_bars + pullback_bars))
        bars.append(_bar(
            tf=Timeframe.M5, t=t,
            o=price, h=price + 1.0, lo=price - 0.5, c=price + 0.5,
        ))
    return bars


def _build_climb_then_pullback_down_m5(
    *, climb_bars: int = 11, pullback_bars: int = 3, plus_one_in_progress: bool = True,
) -> list[Bar]:
    """Mirror of the bullish fixture for SHORT setups: bars climb DOWN
    then pull back UP."""
    bars: list[Bar] = []
    price = 2014.0
    for i in range(climb_bars):
        bars.append(_bar(
            tf=Timeframe.M5, t=_ts(5 * i),
            o=price, h=price + 0.5, lo=price - 1.0, c=price - 0.8,
        ))
        price -= 1.0
    for i in range(pullback_bars):
        t = _ts(5 * (climb_bars + i))
        bars.append(_bar(
            tf=Timeframe.M5, t=t,
            o=price, h=price + 2.0, lo=price - 0.3, c=price + 1.5,
        ))
        price += 2.0
    if plus_one_in_progress:
        t = _ts(5 * (climb_bars + pullback_bars))
        bars.append(_bar(
            tf=Timeframe.M5, t=t,
            o=price, h=price + 0.5, lo=price - 1.0, c=price - 0.5,
        ))
    return bars


def _build_flat_m5(n: int = 15, mid: float = 2000.0) -> list[Bar]:
    """Flat M5 series → VWAP is essentially flat → slope ≈ 0."""
    bars: list[Bar] = []
    for i in range(n):
        c = mid + (0.05 if i % 2 == 0 else -0.05)
        bars.append(_bar(
            tf=Timeframe.M5, t=_ts(5 * i),
            o=mid, h=mid + 0.1, lo=mid - 0.1, c=c,
        ))
    return bars


def _bullish_rejection_trigger(
    *, t: datetime, close: float, body: float = 0.1, wick_mult: float = 4.0,
) -> Bar:
    """Hammer-style M1 candle: long lower wick, tiny body, no upper
    wick.  ``wick_mult`` = lower_wick / body."""
    open_ = close - body
    high = close + 0.05
    low = open_ - wick_mult * body
    return _bar(tf=Timeframe.M1, t=t, o=open_, h=high, lo=low, c=close)


def _bearish_rejection_trigger(
    *, t: datetime, close: float, body: float = 0.1, wick_mult: float = 4.0,
) -> Bar:
    """Inverted-hammer M1 candle: long upper wick, tiny body, no lower
    wick."""
    open_ = close + body
    low = close - 0.05
    high = open_ + wick_mult * body
    return _bar(tf=Timeframe.M1, t=t, o=open_, h=high, lo=low, c=close)


# ---------------------------------------------------------------------- #
# Context builder                                                         #
# ---------------------------------------------------------------------- #


def _build_ctx(
    *, params: VwapPullbackContinuationParams,
    m1_bars: list[Bar] | None = None,
    m5_bars: list[Bar] | None = None,
    m15_bars: list[Bar] | None = None,
    positions: list[Position] | None = None,
    clock_start: datetime | None = None,
) -> tuple[StrategyContext, list[Signal]]:
    bus = AsyncEventBus()
    captured: list[Signal] = []

    async def sink(sig: Signal) -> None:
        captured.append(sig)

    subs = VwapPullbackContinuation.subscriptions(params)
    ctx = StrategyContext(
        strategy_id="vwap_test",
        symbol=params.symbol,
        timeframe=params.entry_timeframe,
        params=params,
        clock=SimClock(clock_start or _ts(0)),
        logger=structlog.get_logger("vwap_test"),
        magic=99,
        signal_sink=sink,
        subscriptions=subs,
        bus=bus,
    )
    for b in m1_bars or []:
        view = ctx.history_for(params.symbol, params.entry_timeframe)
        assert view is not None
        view.append_bar(b)
    for b in m5_bars or []:
        view = ctx.history_for(params.symbol, params.structure_timeframe)
        assert view is not None
        view.append_bar(b)
    for b in m15_bars or []:
        view = ctx.history_for(params.symbol, params.regime_timeframe)
        assert view is not None
        view.append_bar(b)
    if positions:
        ctx.position.update([
            pos.model_copy(update={"magic": ctx.magic}) for pos in positions
        ])
    return ctx, captured


async def _start(strat: VwapPullbackContinuation, ctx: StrategyContext) -> None:
    """Run on_start (which instantiates the TrendingFilter)."""
    await strat.on_start(ctx)


# ---------------------------------------------------------------------- #
# Tests                                                                   #
# ---------------------------------------------------------------------- #


# 1. Warmup gate ------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_signal_when_warmup_insufficient() -> None:
    """Fewer M15 bars than 2*adx_period → strategy bails before any
    indicator runs."""
    strat = VwapPullbackContinuation()
    params = _short_params()
    m15 = _build_uptrend_m15(n=5)  # need 10
    m5 = _build_climb_then_pullback_m5()
    trigger = _bullish_rejection_trigger(t=_ts(15 * 40), close=2006.0)
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await _start(strat, ctx)
    await strat.on_bar(ctx, trigger)
    assert captured == [], "warmup must block all entries"


# 2. ADX regime gate -------------------------------------------------------


@pytest.mark.asyncio
async def test_no_signal_when_adx_below_min() -> None:
    """Sideway M15 → ADX below min_adx → TrendingFilter blocks entry."""
    strat = VwapPullbackContinuation()
    params = _short_params()
    m15 = _build_sideway_m15(n=30)
    m5 = _build_climb_then_pullback_m5()
    trigger = _bullish_rejection_trigger(t=_ts(15 * 40), close=2006.0)
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await _start(strat, ctx)
    await strat.on_bar(ctx, trigger)
    assert captured == [], "ranging regime must block trend-continuation entries"


# 3. VWAP slope gate -------------------------------------------------------


@pytest.mark.asyncio
async def test_no_signal_when_vwap_slope_flat() -> None:
    """Flat M5 → VWAP slope ≈ 0 → strategy rejects (slope must exceed
    vwap_slope_min_atr × ATR)."""
    strat = VwapPullbackContinuation()
    # Crank up min slope so even tiny noise is rejected.
    params = _short_params(vwap_slope_min_atr=1.0)
    m15 = _build_uptrend_m15(n=15)
    m5 = _build_flat_m5(n=15)
    trigger = _bullish_rejection_trigger(t=_ts(15 * 40), close=2000.0)
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await _start(strat, ctx)
    await strat.on_bar(ctx, trigger)
    assert captured == [], "flat VWAP must block — no trend to continue"


# 4. Bias vs slope mismatch -----------------------------------------------


@pytest.mark.asyncio
async def test_no_signal_when_ema_bias_disagrees_with_vwap_slope() -> None:
    """M15 says bias_long (EMA fast > slow) but M5 VWAP is sloping
    down → mismatch → no entry.  Catches turning markets where M15 has
    not yet recognised the reversal."""
    strat = VwapPullbackContinuation()
    params = _short_params()
    m15 = _build_uptrend_m15(n=15)              # bias_long
    m5 = _build_climb_then_pullback_down_m5()    # VWAP slope < 0
    trigger = _bullish_rejection_trigger(t=_ts(15 * 40), close=2007.0)
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await _start(strat, ctx)
    await strat.on_bar(ctx, trigger)
    assert captured == [], "bias vs slope disagreement must block entry"


# 5. Pullback zone too far ------------------------------------------------


@pytest.mark.asyncio
async def test_no_signal_when_too_far_from_vwap() -> None:
    """Trigger close way above VWAP (> pullback_zone_atr × ATR) → no
    entry; we wait for a deeper pullback."""
    strat = VwapPullbackContinuation()
    params = _short_params()
    m15 = _build_uptrend_m15(n=15)
    m5 = _build_climb_then_pullback_m5()
    # Trigger close at 2020 — far above any plausible VWAP near 2005.
    trigger = _bullish_rejection_trigger(t=_ts(15 * 40), close=2020.0)
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await _start(strat, ctx)
    await strat.on_bar(ctx, trigger)
    assert captured == [], "outside the pullback zone must skip — wait for pullback"


# 6. Rejection candle requirement -----------------------------------------


@pytest.mark.asyncio
async def test_no_signal_when_no_rejection_candle() -> None:
    """Trigger has no wick (body-only) → not a rejection → no entry."""
    strat = VwapPullbackContinuation()
    params = _short_params()
    m15 = _build_uptrend_m15(n=15)
    m5 = _build_climb_then_pullback_m5()
    # Pure body candle: open=close=high-eps=low+eps.
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(15 * 40),
        o=2005.5, h=2005.6, lo=2005.4, c=2005.5,
    )
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await _start(strat, ctx)
    await strat.on_bar(ctx, trigger)
    assert captured == [], "no rejection wick → skip"


# 7. Full bullish setup → BUY -------------------------------------------


@pytest.mark.asyncio
async def test_long_signal_on_full_bullish_setup() -> None:
    """All gates pass: trending M15 + climbing M5 with pullback +
    bullish rejection candle inside the pullback zone → BUY emitted."""
    strat = VwapPullbackContinuation()
    params = _short_params()
    m15 = _build_uptrend_m15(n=15)
    m5 = _build_climb_then_pullback_m5()
    # Pick a close inside the pullback zone (VWAP lands near 2005; ATR
    # ~2 → zone is roughly [2005, 2006]).
    trigger = _bullish_rejection_trigger(
        t=_ts(15 * 40), close=2006.5, body=0.1, wick_mult=4.0,
    )
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await _start(strat, ctx)
    await strat.on_bar(ctx, trigger)

    assert len(captured) == 1, f"expected one BUY signal, got {captured}"
    sig = captured[0]
    assert sig.side is Side.BUY
    assert sig.order_type is OrderType.MARKET
    assert sig.suggested_sl is not None and sig.suggested_sl < trigger.close
    assert sig.suggested_tp is not None and sig.suggested_tp > trigger.close
    assert "vwap_pullback_long" in sig.comment


# 8. Full bearish setup → SELL -----------------------------------------


@pytest.mark.asyncio
async def test_short_signal_on_full_bearish_setup() -> None:
    """Mirror of test #7 for SHORT entries: downtrending M15 +
    falling-then-rebounding M5 + bearish rejection → SELL."""
    strat = VwapPullbackContinuation()
    params = _short_params()
    m15 = _build_downtrend_m15(n=15)
    m5 = _build_climb_then_pullback_down_m5()
    trigger = _bearish_rejection_trigger(
        t=_ts(15 * 40), close=2007.5, body=0.1, wick_mult=4.0,
    )
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await _start(strat, ctx)
    await strat.on_bar(ctx, trigger)

    assert len(captured) == 1, f"expected one SELL signal, got {captured}"
    sig = captured[0]
    assert sig.side is Side.SELL
    assert sig.suggested_sl is not None and sig.suggested_sl > trigger.close
    assert sig.suggested_tp is not None and sig.suggested_tp < trigger.close
    assert "vwap_pullback_short" in sig.comment


# 9. SL placement (long) ------------------------------------------------


@pytest.mark.asyncio
async def test_sl_uses_recent_swing_low_plus_atr_buffer() -> None:
    """SL = swing_low(last swing_lookback closed M5 lows) - stop_buffer
    × ATR.  We pin the inequality (below swing low) — exact ATR is
    covered by the ATR test."""
    from stinger_fx.strategies.indicators import atr as atr_fn

    strat = VwapPullbackContinuation()
    params = _short_params(stop_buffer_atr=0.5)
    m15 = _build_uptrend_m15(n=15)
    m5 = _build_climb_then_pullback_m5()
    trigger = _bullish_rejection_trigger(
        t=_ts(15 * 40), close=2006.5, body=0.1, wick_mult=4.0,
    )
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await _start(strat, ctx)
    await strat.on_bar(ctx, trigger)

    assert len(captured) == 1
    sig = captured[0]
    closed_m5 = tuple(m5[:-1])
    swing_low = min(b.low for b in closed_m5[-params.swing_lookback:])
    atr_val = atr_fn(m5, params.atr_period)
    assert atr_val is not None
    expected_sl = swing_low - params.stop_buffer_atr * atr_val
    assert sig.suggested_sl is not None
    assert sig.suggested_sl == pytest.approx(expected_sl, rel=1e-6)


# 10. TP mode prev_extreme -----------------------------------------------


@pytest.mark.asyncio
async def test_tp_mode_prev_extreme_uses_donchian_upper() -> None:
    """tp_mode='prev_extreme' for a long must use the highest high
    among the last prev_extreme_lookback closed M5 bars."""
    strat = VwapPullbackContinuation()
    params = _short_params(tp_mode="prev_extreme", prev_extreme_lookback=10)
    m15 = _build_uptrend_m15(n=15)
    m5 = _build_climb_then_pullback_m5()
    trigger = _bullish_rejection_trigger(
        t=_ts(15 * 40), close=2006.5, body=0.1, wick_mult=4.0,
    )
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await _start(strat, ctx)
    await strat.on_bar(ctx, trigger)

    assert len(captured) == 1
    sig = captured[0]
    closed_m5 = m5[:-1]
    expected_tp = max(b.high for b in closed_m5[-params.prev_extreme_lookback:])
    assert sig.suggested_tp is not None
    assert sig.suggested_tp == pytest.approx(expected_tp, rel=1e-6)


# 11. TP mode fixed_r ----------------------------------------------------


@pytest.mark.asyncio
async def test_tp_mode_fixed_r_uses_r_multiple() -> None:
    """tp_mode='fixed_r' → TP = entry + take_profit_r × risk."""
    strat = VwapPullbackContinuation()
    params = _short_params(tp_mode="fixed_r", take_profit_r=2.0)
    m15 = _build_uptrend_m15(n=15)
    m5 = _build_climb_then_pullback_m5()
    trigger = _bullish_rejection_trigger(
        t=_ts(15 * 40), close=2006.5, body=0.1, wick_mult=4.0,
    )
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await _start(strat, ctx)
    await strat.on_bar(ctx, trigger)

    assert len(captured) == 1
    sig = captured[0]
    assert sig.suggested_sl is not None and sig.suggested_tp is not None
    entry = trigger.close
    risk = entry - sig.suggested_sl
    reward = sig.suggested_tp - entry
    assert reward == pytest.approx(2.0 * risk, rel=1e-6)


# 12. TP mode trailing → manager attached + tp=None --------------------


@pytest.mark.asyncio
async def test_tp_mode_trailing_attaches_manager_and_omits_tp() -> None:
    """tp_mode='trailing' attaches a TrailingStopManager and the
    emitted signal has tp=None (SL serves as initial protection)."""
    strat = VwapPullbackContinuation()
    params = _short_params(tp_mode="trailing")
    m15 = _build_uptrend_m15(n=15)
    m5 = _build_climb_then_pullback_m5()
    trigger = _bullish_rejection_trigger(
        t=_ts(15 * 40), close=2006.5, body=0.1, wick_mult=4.0,
    )
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await _start(strat, ctx)
    assert ctx.managers == []
    await strat.on_bar(ctx, trigger)

    assert len(captured) == 1
    sig = captured[0]
    assert sig.suggested_tp is None
    assert sig.suggested_sl is not None
    assert len(ctx.managers) == 1
    assert isinstance(ctx.managers[0], TrailingStopManager)


# 13. RR gate ------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_setup_when_rr_below_min() -> None:
    """Push min_rr absurdly high so even valid geometry fails."""
    strat = VwapPullbackContinuation()
    params = _short_params(min_rr=100.0)
    m15 = _build_uptrend_m15(n=15)
    m5 = _build_climb_then_pullback_m5()
    trigger = _bullish_rejection_trigger(
        t=_ts(15 * 40), close=2006.5, body=0.1, wick_mult=4.0,
    )
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await _start(strat, ctx)
    await strat.on_bar(ctx, trigger)
    assert captured == [], "RR below min_rr must skip"


# 14. Position-open block ------------------------------------------------


@pytest.mark.asyncio
async def test_no_entry_when_position_already_open() -> None:
    strat = VwapPullbackContinuation()
    params = _short_params()
    existing = Position(
        ticket=1, symbol=SYMBOL, side=Side.BUY, volume=0.01,
        open_price=2005.0, open_time=_ts(0), magic=99,
    )
    m15 = _build_uptrend_m15(n=15)
    m5 = _build_climb_then_pullback_m5()
    trigger = _bullish_rejection_trigger(
        t=_ts(15 * 40), close=2006.5, body=0.1, wick_mult=4.0,
    )
    ctx, captured = _build_ctx(
        params=params, m5_bars=m5, m15_bars=m15, positions=[existing],
    )
    await _start(strat, ctx)
    await strat.on_bar(ctx, trigger)
    assert captured == [], "must not double up while a position is open"


# 15. Cooldown gate ------------------------------------------------------


@pytest.mark.asyncio
async def test_cooldown_blocks_consecutive_entries() -> None:
    strat = VwapPullbackContinuation()
    params = _short_params(cooldown_bars=10)
    m15 = _build_uptrend_m15(n=15)
    m5 = _build_climb_then_pullback_m5()
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await _start(strat, ctx)

    # Simulate a position that just closed on the most recent bar.
    strat._bar_index = 100
    strat._last_close_bar_index = 100
    strat._maybe_roll_session(_ts(15 * 40), params)

    trigger = _bullish_rejection_trigger(
        t=_ts(15 * 40), close=2006.5, body=0.1, wick_mult=4.0,
    )
    await strat.on_bar(ctx, trigger)
    assert captured == [], "cooldown must block immediate re-entry"


# 16. Session-counter cap ------------------------------------------------


@pytest.mark.asyncio
async def test_max_trades_per_session_hard_caps() -> None:
    strat = VwapPullbackContinuation()
    params = _short_params(max_trades_per_session=2)
    m15 = _build_uptrend_m15(n=15)
    m5 = _build_climb_then_pullback_m5()
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await _start(strat, ctx)

    strat._maybe_roll_session(_ts(15 * 40), params)
    strat._trades_this_session = 2

    trigger = _bullish_rejection_trigger(
        t=_ts(15 * 40), close=2006.5, body=0.1, wick_mult=4.0,
    )
    await strat.on_bar(ctx, trigger)
    assert captured == [], "session cap must hard-stop new entries"


# 17. Time exit ----------------------------------------------------------


@pytest.mark.asyncio
async def test_time_exit_closes_position_after_max_hold_bars() -> None:
    """Inject an open ticket and fast-forward past max_hold_bars, then
    drive one bar through on_bar — the time-exit branch must publish a
    ClosePositionRequestEvent for the tracked ticket."""
    from stinger_fx.core.events import ClosePositionRequestEvent

    strat = VwapPullbackContinuation()
    params = _short_params(max_hold_bars=3)
    ctx, _ = _build_ctx(params=params)
    await _start(strat, ctx)

    closes: list[ClosePositionRequestEvent] = []

    async def collect(evt: ClosePositionRequestEvent) -> None:
        closes.append(evt)

    ctx._bus.subscribe(ClosePositionRequestEvent, collect)  # type: ignore[union-attr]

    # Ticket 7 opened at bar index 0; we sit at index 3 → on_bar
    # increments to 4 → elapsed = 4 ≥ 3 → close.
    strat._entry_bar_by_ticket = {7: 0}
    strat._bar_index = 3

    bar = _bar(
        tf=Timeframe.M1, t=_ts(10), o=2000, h=2000.5, lo=1999.5, c=2000,
    )
    await strat.on_bar(ctx, bar)
    for _ in range(3):
        await asyncio.sleep(0)

    assert len(closes) == 1, f"expected one ClosePositionRequestEvent, got {closes}"
    assert closes[0].ticket == 7
    assert "time_exit" in closes[0].reason


# ---------------------------------------------------------------------- #
# Module-level helper classifiers                                          #
# ---------------------------------------------------------------------- #


def test_is_bullish_rejection_classifier() -> None:
    """Lower wick ≥ ratio × body AND lower_wick > upper_wick."""
    # Canonical hammer
    hammer = _bar(
        tf=Timeframe.M1, t=_ts(0), o=2000.0, h=2000.1, lo=1999.0, c=2000.05,
    )
    assert _is_bullish_rejection(hammer, ratio=2.0) is True

    # Tiny lower wick fails ratio
    small = _bar(
        tf=Timeframe.M1, t=_ts(0), o=2000.0, h=2001.0, lo=1999.95, c=2000.5,
    )
    assert _is_bullish_rejection(small, ratio=2.0) is False

    # Upper wick wins → not bullish
    upper = _bar(
        tf=Timeframe.M1, t=_ts(0), o=2000.0, h=2002.0, lo=1999.8, c=2000.2,
    )
    assert _is_bullish_rejection(upper, ratio=2.0) is False

    # Body-only (open == close) → False (no body → degenerate)
    body_only = _bar(
        tf=Timeframe.M1, t=_ts(0), o=2000.0, h=2000.0, lo=2000.0, c=2000.0,
    )
    assert _is_bullish_rejection(body_only, ratio=2.0) is False


def test_is_bearish_rejection_classifier() -> None:
    """Upper wick ≥ ratio × body AND upper_wick > lower_wick."""
    # Canonical inverted hammer / shooting star
    shooter = _bar(
        tf=Timeframe.M1, t=_ts(0), o=2000.0, h=2001.0, lo=1999.95, c=1999.95,
    )
    assert _is_bearish_rejection(shooter, ratio=2.0) is True

    # Lower wick wins → not bearish
    lower = _bar(
        tf=Timeframe.M1, t=_ts(0), o=2000.0, h=2000.2, lo=1998.0, c=1999.8,
    )
    assert _is_bearish_rejection(lower, ratio=2.0) is False

    # Body-only → False
    body_only = _bar(
        tf=Timeframe.M1, t=_ts(0), o=2000.0, h=2000.0, lo=2000.0, c=2000.0,
    )
    assert _is_bearish_rejection(body_only, ratio=2.0) is False
