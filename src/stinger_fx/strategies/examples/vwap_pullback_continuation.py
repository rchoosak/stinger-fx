"""VwapPullbackContinuation — trend-following pullback to session VWAP.

This is the *companion* to ``LiquiditySweepReversal`` (sideway-fade)
in the opposite regime: when M15 confirms a directional move and price
has run away from session VWAP, wait for it to pull back into the VWAP
zone, look for a rejection candle, then enter **with** the trend.

Setup family
============

Three timeframes:

  * **M15** — regime + bias. ADX must be at or above ``min_adx``
    (trending market) and EMA(``ema_fast``) vs EMA(``ema_slow``) on
    close prices must agree on the bias direction. The strategy enters
    *only* in the bias direction.
  * **M5**  — structure. Session VWAP is the dynamic anchor; the M5
    bar series feeds VWAP slope (current VWAP vs VWAP from
    ``vwap_slope_lookback`` bars ago), the swing low/high for stops,
    and Donchian for the ``tp_mode="prev_extreme"`` target. ATR scales
    every threshold.
  * **M1**  — entry trigger. The closed M1 bar must (a) be on the
    correct side of VWAP for the bias, (b) be within
    ``pullback_zone_atr × ATR`` of VWAP, and (c) show a rejection wick
    (lower wick ≥ ``rejection_wick_ratio × body`` for bullish, mirror
    for bearish).

The two strategies share most of their scaffolding (multi-TF
subscriptions, ATR scaling, lifecycle hooks, session/cooldown, time-
based exit). The differences live in the entry evaluation and TP
modes.

Risk discipline
===============

* Max one position per ``symbol`` at a time.
* ``max_trades_per_session`` hard caps daily turnover (UTC date).
* ``cooldown_bars`` of the entry timeframe must elapse after the last
  close.
* Trades with computed reward-to-risk below ``min_rr`` are skipped.
* Positions force-closed via ``ctx.close()`` after ``max_hold_bars``.
* VWAP slope must agree with the EMA bias — we don't fade a turning
  market.

Account-level guards (RiskMonitor per-strategy caps, daily-loss limits,
kill-switch) remain the engine's responsibility.

TP modes
========

* ``prev_extreme`` (default): TP = Donchian upper (long) / lower (short)
  over the last ``prev_extreme_lookback`` closed M5 bars — the
  "previous high/low" target from the user spec.
* ``fixed_r``: TP = entry ± ``take_profit_r × risk``. Risk = |sl − entry|.
* ``trailing``: no fixed TP; a ``TrailingStopManager`` is attached at
  entry time. SL still serves as initial protection. The manager
  ratchets the stop in the trend direction once price moves
  ``trailing_activate_atr × ATR`` in favour.

v1 scope notes
==============

* ``avoid_news`` parameter is reserved as a future hook (no calendar
  service in the framework yet) — stored but never read.
* VWAP bands (σ-based outer envelope) are out of scope; the pullback
  zone uses ATR-scaled distance from VWAP instead.
* Split-leg TP (TP1 + TP2 + runner) is out of scope — single TP per
  trade. Operator picks the mode.
* Tick-by-tick rejection detection is out of scope; rejection is
  evaluated on the closed M1 bar.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import Field

from stinger_fx.domain import Bar, Order, Position, Side, Subscription, Timeframe
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.indicators import (
    atr,
    donchian,
    ema,
    vwap_session,
)
from stinger_fx.strategies.managers.trailing import TrailingStopManager
from stinger_fx.strategies.parameters import StrategyParams
from stinger_fx.strategies.regime import TrendingFilter


class VwapPullbackContinuationParams(StrategyParams):
    """Tunables. Defaults are XAUUSD-friendly starting points (the
    strategy was designed around gold's intraday volatility / VWAP
    geometry), but every threshold is ATR-scaled so the same defaults
    transplant to any liquid trending symbol. Expect to re-optimise
    per instrument, session, and broker spread."""

    # --- Feed identity --------------------------------------------------
    symbol: str = "XAUUSD"
    entry_timeframe: Timeframe = Timeframe.M1
    structure_timeframe: Timeframe = Timeframe.M5
    regime_timeframe: Timeframe = Timeframe.M15

    # --- Trending regime gate (the inverse of LSR's ranging gate) -----
    adx_period: int = Field(14, ge=5, le=50)
    min_adx: float = Field(18.0, gt=0)

    # --- M15 trend bias — EMA fast vs slow ---------------------------
    ema_fast: int = Field(8, ge=2, le=200)
    ema_slow: int = Field(21, ge=5, le=500)

    # --- VWAP slope sanity (must agree with bias) --------------------
    vwap_slope_lookback: int = Field(5, ge=1, le=50)
    vwap_slope_min_atr: float = Field(0.05, ge=0)

    # --- Pullback zone -----------------------------------------------
    # Entry only when |M1.close − VWAP| ≤ pullback_zone_atr × ATR.
    pullback_zone_atr: float = Field(0.5, gt=0)

    # --- Rejection candle wick : body ratio --------------------------
    rejection_wick_ratio: float = Field(2.0, gt=1)

    # --- ATR + dead-market filter -----------------------------------
    atr_period: int = Field(14, ge=5, le=50)
    min_atr: float = Field(0.10, gt=0)

    # --- SL: swing extremum on structure tf, padded by ATR ----------
    swing_lookback: int = Field(10, ge=3, le=100)
    stop_buffer_atr: float = Field(0.30, gt=0)

    # --- TP modes ---------------------------------------------------
    tp_mode: Literal["prev_extreme", "fixed_r", "trailing"] = "prev_extreme"
    prev_extreme_lookback: int = Field(20, ge=5, le=200)
    take_profit_r: float = Field(1.5, gt=0)
    min_rr: float = Field(1.0, gt=0)

    # --- Trailing runner (only when tp_mode == "trailing") ----------
    trailing_distance_atr: float = Field(1.5, gt=0)
    trailing_activate_atr: float = Field(1.0, ge=0)
    # XAU point ≈ 0.01 USD; FX-majors use 0.0001. Trailing distance is
    # converted from ATR to "pips" using this scale.
    trailing_point: float = Field(0.01, gt=0)

    # --- Sizing / cadence ------------------------------------------
    volume: float = Field(0.01, gt=0)
    max_hold_bars: int = Field(12, ge=1)
    cooldown_bars: int = Field(5, ge=0)
    max_trades_per_session: int = Field(6, ge=1)

    # --- Time-of-day session gate (UTC; London open → NY close) -----
    use_session_filter: bool = True
    session_start_hour_utc: int = Field(8, ge=0, le=23)
    session_end_hour_utc: int = Field(21, ge=1, le=24)

    # --- Future hooks (v1: no-op) -----------------------------------
    avoid_news: bool = False


class VwapPullbackContinuation(BaseStrategy):
    name = "vwap_pullback_continuation"
    Params = VwapPullbackContinuationParams

    @classmethod
    def subscriptions(cls, params: StrategyParams) -> list[Subscription]:
        assert isinstance(params, VwapPullbackContinuationParams)
        return [
            Subscription(symbol=params.symbol, timeframe=params.entry_timeframe),
            Subscription(symbol=params.symbol, timeframe=params.structure_timeframe),
            Subscription(symbol=params.symbol, timeframe=params.regime_timeframe),
        ]

    @classmethod
    def warmup_bars(
        cls, params: StrategyParams,
    ) -> dict[Subscription, int] | None:
        """Per-feed warmup for live-mode startup backfill (Plan A2).

        VPC's M5 structure-tf needs the **session VWAP** anchor, which
        re-anchors at every UTC midnight (or the configured session
        start hour).  So the M5 window must cover a *full day* worth of
        bars (288 closed M5 bars = 24h) — otherwise the first signal
        after engine start uses a partial VWAP.

        The other periods (swing/Donchian/ATR/EMA) all fit within 24h,
        so we use ``max(24h, indicator-bars)`` to be safe.
        """
        assert isinstance(params, VwapPullbackContinuationParams)
        m5_indicator_bars = max(
            params.swing_lookback,
            params.prev_extreme_lookback,
            params.vwap_slope_lookback + 2,
            params.atr_period + 1,
        ) + 1
        m5_session_bars = 24 * 60 // 5  # 24h of M5 = 288
        return {
            Subscription(symbol=params.symbol, timeframe=params.entry_timeframe): 1,
            Subscription(symbol=params.symbol, timeframe=params.structure_timeframe):
                max(m5_indicator_bars, m5_session_bars),
            Subscription(symbol=params.symbol, timeframe=params.regime_timeframe):
                max(2 * params.adx_period, params.ema_slow + 1),
        }

    def __init__(self) -> None:
        super().__init__()
        self._last_close_bar_index: int | None = None
        self._trades_this_session: int = 0
        self._session_key: str | None = None
        self._entry_bar_by_ticket: dict[int, int] = {}
        self._bar_index: int = 0
        # TrendingFilter is created on_start (needs params from ctx).
        self._trending_filter: TrendingFilter | None = None

    async def on_start(self, ctx: StrategyContext) -> None:
        params = ctx.params
        assert isinstance(params, VwapPullbackContinuationParams)
        self._trending_filter = TrendingFilter(
            adx_period=params.adx_period, threshold=params.min_adx,
        )

    # ------------------------------------------------------------------ #
    # Bar dispatch                                                        #
    # ------------------------------------------------------------------ #

    async def on_bar(self, ctx: StrategyContext, bar: Bar) -> None:
        params = ctx.params
        assert isinstance(params, VwapPullbackContinuationParams)

        # Multi-TF: ignore M5/M15 bars; only act on the entry tf for
        # this strategy's symbol. M5/M15 still get appended to their
        # HistoryView by the runner; we read them via history_for below.
        if bar.symbol != params.symbol or bar.timeframe != params.entry_timeframe:
            return
        self._bar_index += 1

        self._maybe_roll_session(bar.time, params)
        await self._maybe_time_exit(ctx)

        # --- Entry gates -----------------------------------------------
        if params.use_session_filter and not self._in_session(bar.time, params):
            return
        if self._trades_this_session >= params.max_trades_per_session:
            return
        if self._on_cooldown(params):
            return
        if ctx.position.for_symbol(params.symbol):
            return  # one position per symbol

        # --- Multi-TF data fetch + warmup ------------------------------
        m5 = ctx.history_for(params.symbol, params.structure_timeframe)
        m15 = ctx.history_for(params.symbol, params.regime_timeframe)
        if m5 is None or m15 is None:
            return
        m5_bars = m5.bars()
        m15_bars = m15.bars()
        needed_m5 = max(
            params.swing_lookback,
            params.prev_extreme_lookback,
            params.vwap_slope_lookback + 2,
            params.atr_period + 1,
        )
        if len(m5_bars) < needed_m5 + 1:  # +1 because we slice [:-1]
            return
        if len(m15_bars) < max(2 * params.adx_period, params.ema_slow + 1):
            return

        # --- Regime: M15 ADX must indicate trending market -----------
        assert self._trending_filter is not None
        if not self._trending_filter.allows(m15_bars):
            return

        # --- Bias: M15 EMA fast vs slow ------------------------------
        m15_closes = m15.closes()
        ema_f = ema(m15_closes, params.ema_fast)
        ema_s = ema(m15_closes, params.ema_slow)
        if ema_f is None or ema_s is None:
            return
        bias_long = ema_f > ema_s
        bias_short = ema_f < ema_s
        if not (bias_long or bias_short):
            return  # exactly equal — no edge

        # --- ATR scale + dead-market filter -------------------------
        atr_value = atr(m5_bars, params.atr_period)
        if atr_value is None or atr_value < params.min_atr:
            return

        # --- VWAP (closed M5 bars only) + slope check ---------------
        # Slice m5_bars[:-1] before passing to vwap_session/donchian
        # /swing so the in-progress structure bar can't leak into the
        # range the M1 trigger has to evaluate against. Canonical
        # lookahead-bias guard — same as LSR.
        closed_m5 = m5_bars[:-1]
        vwap_now = vwap_session(closed_m5)
        if vwap_now is None:
            return
        if len(closed_m5) <= params.vwap_slope_lookback:
            return
        vwap_then = vwap_session(closed_m5[:-params.vwap_slope_lookback])
        if vwap_then is None:
            return
        slope = vwap_now - vwap_then
        if abs(slope) < params.vwap_slope_min_atr * atr_value:
            return
        # VWAP slope must agree with EMA bias.
        if bias_long and slope < 0:
            return
        if bias_short and slope > 0:
            return

        # --- Build the directional setup ----------------------------
        side = Side.BUY if bias_long else Side.SELL
        dist = bar.close - vwap_now  # positive = above VWAP

        if side is Side.BUY:
            if dist < 0:
                return  # below VWAP — no long
            if dist > params.pullback_zone_atr * atr_value:
                return  # too far — wait for the pullback
            if not _is_bullish_rejection(bar, params.rejection_wick_ratio):
                return
            swing = self._recent_swing_low(closed_m5, params.swing_lookback)
            sl = swing - params.stop_buffer_atr * atr_value
        else:
            if dist > 0:
                return  # above VWAP — no short
            if -dist > params.pullback_zone_atr * atr_value:
                return
            if not _is_bearish_rejection(bar, params.rejection_wick_ratio):
                return
            swing = self._recent_swing_high(closed_m5, params.swing_lookback)
            sl = swing + params.stop_buffer_atr * atr_value

        entry = bar.close
        tp = self._compute_tp(
            params=params, side=side, entry=entry, sl=sl, closed_m5=closed_m5,
        )
        if not self._rr_ok(entry=entry, sl=sl, tp=tp, side=side, params=params):
            return

        # Optional trailing runner — attach before placing the order so
        # the manager's on_tick is already wired when the first tick
        # post-fill arrives.
        if params.tp_mode == "trailing":
            ctx.attach_manager(TrailingStopManager(
                ctx,
                distance_pips=params.trailing_distance_atr * atr_value / params.trailing_point,
                activate_after_pips=params.trailing_activate_atr * atr_value / params.trailing_point,
                symbol=params.symbol,
                point=params.trailing_point,
            ))
            tp_arg: float | None = None  # SL serves as initial protection
        else:
            tp_arg = tp

        if side is Side.BUY:
            await ctx.buy(
                params.volume, sl=sl, tp=tp_arg, comment="vwap_pullback_long",
            )
        else:
            await ctx.sell(
                params.volume, sl=sl, tp=tp_arg, comment="vwap_pullback_short",
            )

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def on_order_filled(self, ctx: StrategyContext, order: Order) -> None:
        if order.strategy_id == ctx.strategy_id and order.ticket:
            self._entry_bar_by_ticket[order.ticket] = self._bar_index

    async def on_position_closed(
        self, ctx: StrategyContext, position: Position,
    ) -> None:
        self._entry_bar_by_ticket.pop(position.ticket, None)
        self._last_close_bar_index = self._bar_index
        self._trades_this_session += 1

    # ------------------------------------------------------------------ #
    # TP / RR                                                             #
    # ------------------------------------------------------------------ #

    def _compute_tp(
        self,
        *,
        params: VwapPullbackContinuationParams,
        side: Side,
        entry: float,
        sl: float,
        closed_m5: tuple[Bar, ...],
    ) -> float | None:
        if params.tp_mode == "trailing":
            # The runner exits via TrailingStopManager; no fixed TP is
            # set on the order. We still need a number to evaluate RR
            # against, so use a synthetic TP at one full trailing
            # distance — RR is checked even for trailing trades so we
            # don't open setups whose risk profile is hopeless.
            risk = abs(sl - entry)
            if risk <= 0:
                return None
            if side is Side.BUY:
                return entry + params.take_profit_r * risk
            return entry - params.take_profit_r * risk
        if params.tp_mode == "fixed_r":
            risk = abs(sl - entry)
            if risk <= 0:
                return None
            if side is Side.BUY:
                return entry + params.take_profit_r * risk
            return entry - params.take_profit_r * risk
        # tp_mode == "prev_extreme"
        channels = donchian(
            closed_m5[-params.prev_extreme_lookback:],
            params.prev_extreme_lookback,
        )
        if channels is None:
            return None
        return channels.upper if side is Side.BUY else channels.lower

    @staticmethod
    def _rr_ok(
        *,
        entry: float,
        sl: float,
        tp: float | None,
        side: Side,
        params: VwapPullbackContinuationParams,
    ) -> bool:
        if tp is None:
            return False
        risk = abs(sl - entry)
        reward = (tp - entry) if side is Side.BUY else (entry - tp)
        if risk <= 0 or reward <= 0:
            return False
        return (reward / risk) >= params.min_rr

    # ------------------------------------------------------------------ #
    # Swing helpers (closed bars only — caller pre-slices [:-1])        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _recent_swing_low(closed_m5: tuple[Bar, ...], n: int) -> float:
        return min(b.low for b in closed_m5[-n:])

    @staticmethod
    def _recent_swing_high(closed_m5: tuple[Bar, ...], n: int) -> float:
        return max(b.high for b in closed_m5[-n:])

    # ------------------------------------------------------------------ #
    # Session / cooldown / time exit  (lifted from LSR — same shape)     #
    # ------------------------------------------------------------------ #

    def _maybe_roll_session(
        self, now: datetime, params: VwapPullbackContinuationParams,
    ) -> None:
        key = self._session_start(now, params.session_start_hour_utc).date().isoformat()
        if self._session_key != key:
            self._session_key = key
            self._trades_this_session = 0

    @staticmethod
    def _session_start(now: datetime, session_start_hour_utc: int) -> datetime:
        boundary = now.replace(
            hour=session_start_hour_utc, minute=0, second=0, microsecond=0,
        )
        if now < boundary:
            boundary -= timedelta(days=1)
        return boundary

    @staticmethod
    def _in_session(
        now: datetime, params: VwapPullbackContinuationParams,
    ) -> bool:
        hour = now.hour
        return params.session_start_hour_utc <= hour < params.session_end_hour_utc

    def _on_cooldown(self, params: VwapPullbackContinuationParams) -> bool:
        if self._last_close_bar_index is None:
            return False
        return (self._bar_index - self._last_close_bar_index) < params.cooldown_bars

    async def _maybe_time_exit(self, ctx: StrategyContext) -> None:
        if not self._entry_bar_by_ticket:
            return
        params = ctx.params
        assert isinstance(params, VwapPullbackContinuationParams)
        for ticket, entry_index in list(self._entry_bar_by_ticket.items()):
            if (self._bar_index - entry_index) >= params.max_hold_bars:
                await ctx.close(ticket, reason="vwap_pullback_time_exit")


# --- Module-level helpers (inline rejection-candle classifiers) -----------


def _is_bullish_rejection(bar: Bar, ratio: float) -> bool:
    """Long lower wick relative to body and longer than the upper wick.
    The canonical "hammer / pin bar" shape that says sellers tried but
    failed."""
    body = abs(bar.close - bar.open)
    if body <= 0:
        return False
    lower_wick = min(bar.open, bar.close) - bar.low
    upper_wick = bar.high - max(bar.open, bar.close)
    return lower_wick >= ratio * body and lower_wick > upper_wick


def _is_bearish_rejection(bar: Bar, ratio: float) -> bool:
    """Long upper wick — shooting star / inverted hammer. Buyers tried
    but failed."""
    body = abs(bar.close - bar.open)
    if body <= 0:
        return False
    upper_wick = bar.high - max(bar.open, bar.close)
    lower_wick = min(bar.open, bar.close) - bar.low
    return upper_wick >= ratio * body and upper_wick > lower_wick
