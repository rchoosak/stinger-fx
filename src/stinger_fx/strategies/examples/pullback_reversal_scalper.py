"""PullbackReversalScalper — M1 Stoch RSI + RSI entries with M5 trend filter.

Concept
=======

Trade **with** the M5 trend, but enter at M1 extremes during a sharp
pullback against that trend. The setup is selective by design — both
the higher-timeframe direction AND the lower-timeframe extremes have to
align before a trade fires.

BUY setup
  * **M5 uptrend confirmation**
      - ``close > EMA50``
      - ``EMA20 > EMA50``
      - HH/HL: latest M5 ``high`` > ``high`` ``swing_lookback`` bars ago
        AND latest M5 ``low`` > ``low`` ``swing_lookback`` bars ago
  * **M5 RSI < m5_rsi_buy_max** (default 50) — price is *currently
    pulling back* within the larger uptrend. This is the "Buy the dip"
    confirmation: we don't enter a strong uptrend that just broke out,
    we enter the moment it cools off.
  * **M1 entry**:
      - Stoch RSI K crosses up D
      - the cross happens with K below ``stoch_rsi_oversold`` (default 20)
      - M1 RSI also below ``m1_rsi_oversold`` (default 20) — confirms
        oversold context, not just a K/D cross in neutral land

SELL setup is symmetric — M5 downtrend + RSI > 50 + M1 cross down in OB.

Exits
=====

* **Stoch-K reversal** — close BUY when M1 Stoch RSI K > ``stoch_rsi_exit_long``
  (default 80); close SELL when K < ``stoch_rsi_exit_short`` (default 20).
  Doesn't wait for a K/D cross — the strategy is intentionally fast-out.
* **Time stop** — close after ``max_hold_bars_m1`` M1 bars regardless.
  Safety net for cases the Stoch never reaches the exit threshold.
* **Initial protective SL** — ATR-scaled, configurable via ``sl_atr_mult``.
* **Fixed TP** — disabled by default (``tp_atr_mult=0``); set non-zero
  to add a profit ceiling alongside the Stoch-K exit.

Tuning knobs
============

Every threshold (RSI levels, Stoch RSI levels, lookbacks, ATR multipliers,
cooldown, max-hold) is exposed in ``PullbackReversalScalperParams`` so the
operator can re-fit per symbol / session.
"""
from __future__ import annotations

import logging
from typing import ClassVar

from pydantic import Field

from stinger_fx.domain import Bar, Order, Position, Side, Timeframe
from stinger_fx.domain.symbols import Subscription
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.indicators import atr, ema, rsi, stoch_rsi
from stinger_fx.strategies.managers.trailing import TrailingStopManager
from stinger_fx.strategies.parameters import StrategyParams

logger = logging.getLogger("stinger.strategies.pullback_reversal_scalper")


class PullbackReversalScalperParams(StrategyParams):
    """Tunables for the pullback-reversal scalper. Defaults are a
    sensible starting point for XAU/USD but expect to re-optimise per
    instrument / session."""

    # --- Feeds ----------------------------------------------------------
    symbol: str = "XAUUSD"
    entry_timeframe: Timeframe = Timeframe.M1
    structure_timeframe: Timeframe = Timeframe.M5

    # --- M5 structure ---------------------------------------------------
    # When ``use_m5_filter`` is False, every M5 gate below is skipped —
    # the strategy degenerates to a pure-M1 mean-reversion scalper that
    # only requires the M1 Stoch RSI cross + RSI extreme. This is the
    # default because the strict M5 trend + pullback combo rarely all
    # aligns on real XAU data; flip it back ON if you want the original
    # selective behavior.
    use_m5_filter: bool = False
    ema_fast_period: int = Field(20, ge=2, le=200)
    ema_slow_period: int = Field(50, ge=5, le=400)
    m5_rsi_period: int = Field(14, ge=2, le=50)
    m5_rsi_buy_max: float = Field(50.0, gt=0, lt=100)    # BUY: m5_rsi < this
    m5_rsi_sell_min: float = Field(50.0, gt=0, lt=100)   # SELL: m5_rsi > this
    swing_lookback: int = Field(5, ge=1, le=50)           # bars back for HH/HL check

    # --- M1 entry triggers ---------------------------------------------
    m1_rsi_period: int = Field(14, ge=2, le=50)
    m1_rsi_oversold: float = Field(20.0, gt=0, lt=100)
    m1_rsi_overbought: float = Field(80.0, gt=0, lt=100)
    stoch_rsi_period: int = Field(14, ge=2, le=50)
    stoch_rsi_k_smooth: int = Field(3, ge=1, le=10)
    stoch_rsi_d_smooth: int = Field(3, ge=1, le=10)
    stoch_rsi_oversold: float = Field(20.0, gt=0, lt=100)
    stoch_rsi_overbought: float = Field(80.0, gt=0, lt=100)

    # --- Exits ---------------------------------------------------------
    stoch_rsi_exit_long: float = Field(80.0, gt=0, lt=100)
    stoch_rsi_exit_short: float = Field(20.0, gt=0, lt=100)

    # --- Risk + sizing -------------------------------------------------
    volume: float = Field(0.01, gt=0)
    atr_period: int = Field(14, ge=2, le=50)
    sl_atr_mult: float = Field(1.0, gt=0)
    tp_atr_mult: float = Field(0.0, ge=0)         # 0 → no fixed TP
    max_hold_bars_m1: int = Field(30, ge=1, le=10_000)
    cooldown_bars_m1: int = Field(2, ge=0, le=10_000)

    # --- Trailing stop (optional, default OFF) -------------------------
    # When enabled, a ``TrailingStopManager`` is attached on every entry
    # using ATR-scaled distance + activation. The manager ratchets the SL
    # in our favour as price moves with the trade — it never moves
    # against us, so the initial SL is the worst-case stop.
    enable_trailing: bool = False
    # How far behind price the trailing SL stays, in ATR units.
    trailing_distance_atr: float = Field(0.8, gt=0)
    # How much profit (in ATR units) the trade must accumulate before the
    # trail starts tightening. 0 = trail from entry.
    trailing_activate_atr: float = Field(0.5, ge=0)
    # Symbol's quote-point value used to convert ATR (price units) → pips,
    # which is what TrailingStopManager expects. XAU = 0.01, FX = 0.0001.
    trailing_point: float = Field(0.01, gt=0)


class PullbackReversalScalper(BaseStrategy):
    name: ClassVar[str] = "pullback_reversal_scalper"
    Params: ClassVar[type[StrategyParams]] = PullbackReversalScalperParams

    @classmethod
    def subscriptions(cls, params: StrategyParams) -> list[Subscription]:
        assert isinstance(params, PullbackReversalScalperParams)
        return [
            Subscription(symbol=params.symbol, timeframe=params.entry_timeframe),
            Subscription(symbol=params.symbol, timeframe=params.structure_timeframe),
        ]

    @classmethod
    def warmup_bars(
        cls, params: StrategyParams,
    ) -> dict[Subscription, int] | None:
        """Per-feed warmup for live-mode startup backfill.

        Stoch RSI needs roughly ``rsi_period + stoch_period + k_smooth +
        d_smooth`` bars before it produces a non-None value; we add a
        small buffer for the K/D cross detection to have a valid prior
        sample. M5 structure needs the slow EMA window + the swing
        lookback comparison.
        """
        assert isinstance(params, PullbackReversalScalperParams)
        m1_need = (
            params.m1_rsi_period
            + params.stoch_rsi_period
            + params.stoch_rsi_k_smooth
            + params.stoch_rsi_d_smooth
            + 5
        )
        m5_need = max(
            params.ema_slow_period,
            params.m5_rsi_period,
            params.swing_lookback + 1,
        ) + 5
        return {
            Subscription(symbol=params.symbol, timeframe=params.entry_timeframe):
                m1_need,
            Subscription(symbol=params.symbol, timeframe=params.structure_timeframe):
                m5_need,
        }

    def __init__(self) -> None:
        super().__init__()
        # Cross-detection state — populated once we have two consecutive
        # Stoch RSI samples on the M1 entry timeframe.
        self._prev_k: float | None = None
        self._prev_d: float | None = None
        # Open-position tracking — `ctx.close(...)` only sends a request,
        # so we need to remember the ticket + side until on_position_closed
        # confirms the exit.
        self._open_ticket: int | None = None
        self._open_side: Side | None = None
        self._open_bars: int = 0
        self._cooldown_left: int = 0

    # ------------------------------------------------------------------ #
    # Bar dispatch                                                        #
    # ------------------------------------------------------------------ #

    async def on_bar(self, ctx: StrategyContext, bar: Bar) -> None:
        params = ctx.params
        assert isinstance(params, PullbackReversalScalperParams)

        # Multi-TF: only act on the entry tf. M5 bars still feed the
        # corresponding HistoryView automatically via the runner.
        if bar.symbol != params.symbol or bar.timeframe != params.entry_timeframe:
            return

        # 1) If a position is open, check for an exit on this bar and
        #    return — never stack a new entry on top.
        if self._open_ticket is not None:
            self._open_bars += 1
            await self._maybe_exit(ctx, bar, params)
            return

        # 2) Cooldown after the previous close.
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return

        # 3) Read M5 structure (optional — gated by `use_m5_filter`).
        #
        # When the filter is OFF (default) we skip the M5 trend / RSI /
        # HH/HL gates entirely and let M1 indicators decide. That's the
        # "fire on M1 extreme, ignore the bigger picture" mode. Flip
        # `use_m5_filter=True` to restore the strict pullback-with-trend
        # behavior described in the strategy's docstring.
        m5_buy_ok = True
        m5_sell_ok = True
        if params.use_m5_filter:
            m5_view = ctx.history_for(params.symbol, params.structure_timeframe)
            if m5_view is None:
                return
            m5_bars = m5_view.bars()
            if len(m5_bars) < max(
                params.ema_slow_period,
                params.m5_rsi_period,
                params.swing_lookback + 1,
            ):
                return
            m5_closes = [b.close for b in m5_bars]
            ema_fast = ema(m5_closes, params.ema_fast_period)
            ema_slow = ema(m5_closes, params.ema_slow_period)
            m5_rsi = rsi(m5_closes, params.m5_rsi_period)
            if ema_fast is None or ema_slow is None or m5_rsi is None:
                return
            latest_m5 = m5_bars[-1]
            ref_m5 = m5_bars[-1 - params.swing_lookback]
            is_uptrend = (
                latest_m5.close > ema_slow
                and ema_fast > ema_slow
                and latest_m5.high > ref_m5.high
                and latest_m5.low > ref_m5.low
            )
            is_downtrend = (
                latest_m5.close < ema_slow
                and ema_fast < ema_slow
                and latest_m5.high < ref_m5.high
                and latest_m5.low < ref_m5.low
            )
            m5_buy_ok = is_uptrend and m5_rsi < params.m5_rsi_buy_max
            m5_sell_ok = is_downtrend and m5_rsi > params.m5_rsi_sell_min

        # 4) Compute M1 RSI + Stoch RSI for entry gating.
        m1_view = ctx.history_for(params.symbol, params.entry_timeframe)
        if m1_view is None:
            return
        m1_bars = m1_view.bars()
        m1_closes = [b.close for b in m1_bars]
        if len(m1_closes) < params.m1_rsi_period + params.stoch_rsi_period:
            return
        m1_rsi_v = rsi(m1_closes, params.m1_rsi_period)
        srsi = stoch_rsi(
            m1_closes,
            rsi_period=params.m1_rsi_period,
            stoch_period=params.stoch_rsi_period,
            k_smooth=params.stoch_rsi_k_smooth,
            d_smooth=params.stoch_rsi_d_smooth,
        )
        if m1_rsi_v is None or srsi is None:
            return
        k, d = srsi.k, srsi.d
        prev_k, prev_d = self._prev_k, self._prev_d
        # Update state regardless of whether we trade this bar — we need
        # a fresh "prev" sample on every subsequent bar.
        self._prev_k, self._prev_d = k, d
        if prev_k is None or prev_d is None:
            return
        cross_up = prev_k < prev_d and k >= d
        cross_down = prev_k > prev_d and k <= d

        # 5) ATR for SL distance.
        atr_v = atr(list(m1_bars), params.atr_period)
        if atr_v is None or atr_v <= 0:
            return
        sl_dist = atr_v * params.sl_atr_mult
        tp_dist = atr_v * params.tp_atr_mult if params.tp_atr_mult > 0 else None

        # 6) BUY — (optional M5 gate) + M1 Stoch K up out of OS + M1 oversold.
        if (
            m5_buy_ok
            and cross_up
            and k < params.stoch_rsi_oversold
            and m1_rsi_v < params.m1_rsi_oversold
        ):
            entry = bar.close
            await ctx.buy(
                volume=params.volume,
                sl=entry - sl_dist,
                tp=(entry + tp_dist) if tp_dist is not None else None,
                comment="prs_scalp_buy",
            )
            self._attach_trailing_if_enabled(ctx, params, atr_v)
            return

        # 7) SELL — symmetric.
        if (
            m5_sell_ok
            and cross_down
            and k > params.stoch_rsi_overbought
            and m1_rsi_v > params.m1_rsi_overbought
        ):
            entry = bar.close
            await ctx.sell(
                volume=params.volume,
                sl=entry + sl_dist,
                tp=(entry - tp_dist) if tp_dist is not None else None,
                comment="prs_scalp_sell",
            )
            self._attach_trailing_if_enabled(ctx, params, atr_v)

    # ------------------------------------------------------------------ #
    # Trailing-stop wiring                                                #
    # ------------------------------------------------------------------ #

    def _attach_trailing_if_enabled(
        self,
        ctx: StrategyContext,
        params: PullbackReversalScalperParams,
        atr_value: float,
    ) -> None:
        """Attach a TrailingStopManager scaled to the entry-time ATR.

        Same per-entry pattern OpeningRangeBreakout uses — the manager
        starts ratcheting the SL on the next tick. Distance + activation
        are expressed in ATR multiples and converted to pips via
        ``trailing_point`` (the symbol's quote-point value).
        No-op when ``enable_trailing`` is False so the default path has
        zero overhead.
        """
        if not params.enable_trailing:
            return
        ctx.attach_manager(
            TrailingStopManager(
                ctx,
                distance_pips=params.trailing_distance_atr * atr_value
                / params.trailing_point,
                activate_after_pips=params.trailing_activate_atr * atr_value
                / params.trailing_point,
                symbol=params.symbol,
                point=params.trailing_point,
            )
        )

    # ------------------------------------------------------------------ #
    # Exit decision                                                       #
    # ------------------------------------------------------------------ #

    async def _maybe_exit(
        self,
        ctx: StrategyContext,
        bar: Bar,
        params: PullbackReversalScalperParams,
    ) -> None:
        """Called on every entry-tf bar while a position is open.

        Closes the position when either:
          * Stoch RSI K crosses the configured exit threshold for the
            direction (long: K > 80; short: K < 20), or
          * the position has been held for ``max_hold_bars_m1`` bars
            (safety time stop — even if Stoch never reaches the exit).
        """
        m1_view = ctx.history_for(params.symbol, params.entry_timeframe)
        if m1_view is None:
            return
        m1_closes = [b.close for b in m1_view.bars()]
        srsi = stoch_rsi(
            m1_closes,
            rsi_period=params.m1_rsi_period,
            stoch_period=params.stoch_rsi_period,
            k_smooth=params.stoch_rsi_k_smooth,
            d_smooth=params.stoch_rsi_d_smooth,
        )
        if srsi is None:
            return
        k = srsi.k
        # Keep the cross-detection samples fresh while we're in a trade,
        # so the next entry can see a valid "prev" the moment we exit.
        self._prev_k = k
        self._prev_d = srsi.d

        exit_now = (
            (self._open_side is Side.BUY and k > params.stoch_rsi_exit_long)
            or (self._open_side is Side.SELL and k < params.stoch_rsi_exit_short)
            or (self._open_bars >= params.max_hold_bars_m1)
        )
        if exit_now and self._open_ticket is not None:
            await ctx.close(self._open_ticket, reason="prs_scalp_exit")

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def on_order_filled(self, ctx: StrategyContext, order: Order) -> None:
        # Only track tickets our own strategy opened. Multiple strategies
        # may share a broker in multi-strategy live setups.
        if order.strategy_id != ctx.strategy_id or not order.ticket:
            return
        self._open_ticket = order.ticket
        self._open_side = order.side
        self._open_bars = 0

    async def on_position_closed(
        self, ctx: StrategyContext, position: Position,
    ) -> None:
        if position.ticket != self._open_ticket:
            return
        params = ctx.params
        assert isinstance(params, PullbackReversalScalperParams)
        self._open_ticket = None
        self._open_side = None
        self._open_bars = 0
        self._cooldown_left = params.cooldown_bars_m1
