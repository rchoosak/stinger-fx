"""MultiTimeframeConfluenceScalper — M1 RSI pullback inside aligned M5+M15 trend.

Concept
=======

The most "boring" of the family — only fire when **three timeframes
agree on direction**, then time the entry off a short-term pullback
on the M1.  Selectivity is the point: PRS / MBS / OFIS all over-trade
and pay the spread.  This one waits.

BUY setup
  * **M15 trend**: ``ema(M15 closes, m15_ema_fast) > ema(M15 closes,
    m15_ema_slow)`` — the bigger picture is up.
  * **M5 trend**: same shape on M5 (``m5_ema_fast > m5_ema_slow``).
  * **M1 trigger**: the M1 ``rsi`` must **cross up** out of oversold —
    the *previous* M1 bar's RSI was ``< m1_rsi_oversold`` and the
    *current* bar's RSI is ``>= m1_rsi_oversold``.  Waiting for the
    cross-back (instead of firing while still oversold) avoids buying
    into a falling knife — entry only after price shows it has turned.
  * Session-avoid / vol-gate / cooldown gates all pass (reused
    verbatim from :mod:`momentum_breakout_scalper`).

SELL setup is symmetric — M15 down + M5 down + M1 RSI crossing *down*
out of overbought (prev ``> m1_rsi_overbought``, current ``<=``).

Exits
=====

Same shape as MBS / OFIS so all three are A/B-comparable on the
same harness:

* Fixed ATR take-profit (default ``tp_atr_mult = 2.0``).
* Initial protective SL (default ``sl_atr_mult = 1.0``).
* Time stop after ``max_hold_bars_m1`` bars.
* Optional trailing manager.

Why this should win the family on retail data
=============================================

Multi-TF confluence is selective by design — typically 3–8 trades per
day on XAU/USD M1 vs 40–80 for MBS and 350+ for OFIS — so spread
cost per dollar of P&L is dramatically lower.  Academic literature
puts the unconditional win rate of "trend + pullback" setups in the
55–65% range when measured on out-of-sample data, with R:R 1.5–2:1
target.
"""
from __future__ import annotations

import logging
from typing import ClassVar

from pydantic import Field

from stinger_fx.domain import Bar, Order, Position, Side, Timeframe
from stinger_fx.domain.symbols import Subscription
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.indicators import atr, ema, rsi, sma
from stinger_fx.strategies.managers.trailing import TrailingStopManager
from stinger_fx.strategies.parameters import StrategyParams

logger = logging.getLogger("stinger.strategies.multi_tf_confluence_scalper")


class MultiTFConfluenceScalperParams(StrategyParams):
    """Tunables for the multi-TF confluence scalper.  Non-confluence
    knobs (session / vol / risk) mirror the winning MBS defaults so
    A/B testing isolates the impact of the 3-TF trend filter."""

    # --- Feeds ---------------------------------------------------------
    symbol: str = "XAUUSD"
    entry_timeframe: Timeframe = Timeframe.M1
    mid_timeframe: Timeframe = Timeframe.M5
    high_timeframe: Timeframe = Timeframe.M15

    # --- M5 trend filter ----------------------------------------------
    m5_ema_fast: int = Field(20, ge=2, le=200)
    m5_ema_slow: int = Field(50, ge=5, le=400)

    # --- M15 trend filter ---------------------------------------------
    # Defaults chosen so the M15 lookback is longer than the M5 one
    # — otherwise the higher TF wouldn't be adding any "bigger
    # picture" filter information.
    m15_ema_fast: int = Field(21, ge=2, le=200)
    m15_ema_slow: int = Field(55, ge=5, le=400)

    # --- M1 entry trigger ---------------------------------------------
    m1_rsi_period: int = Field(14, ge=2, le=50)
    m1_rsi_oversold: float = Field(30.0, gt=0, lt=100)
    m1_rsi_overbought: float = Field(70.0, gt=0, lt=100)

    # --- Session avoidance --------------------------------------------
    avoid_hours_utc: list[int] = Field(default_factory=lambda: [13, 14, 15, 18])

    # --- Volatility gate ----------------------------------------------
    vol_gate_lookback: int = Field(50, ge=2, le=500)
    vol_gate_mult: float = Field(1.5, gt=0)

    # --- Risk + sizing ------------------------------------------------
    volume: float = Field(0.01, gt=0)
    atr_period: int = Field(14, ge=2, le=50)
    sl_atr_mult: float = Field(1.0, gt=0)
    tp_atr_mult: float = Field(2.0, ge=0)
    max_hold_bars_m1: int = Field(30, ge=1, le=10_000)
    cooldown_bars_m1: int = Field(5, ge=0, le=10_000)

    # --- Trailing stop (optional) -------------------------------------
    enable_trailing: bool = False
    trailing_distance_atr: float = Field(0.8, gt=0)
    trailing_activate_atr: float = Field(0.5, ge=0)
    trailing_point: float = Field(0.01, gt=0)


def _true_range_series(bars: list[Bar]) -> list[float]:
    """Per-bar True Range — same helper PRS/MBS/OFIS inline.

    Kept inline so each strategy file is independently readable; a
    shared mixin can fold these together later if the duplication
    becomes annoying.
    """
    out: list[float] = []
    if len(bars) < 2:
        return out
    prev_close = bars[0].close
    for b in bars[1:]:
        tr = max(b.high - b.low, abs(b.high - prev_close), abs(b.low - prev_close))
        out.append(tr)
        prev_close = b.close
    return out


class MultiTFConfluenceScalper(BaseStrategy):
    name: ClassVar[str] = "multi_tf_confluence_scalper"
    Params: ClassVar[type[StrategyParams]] = MultiTFConfluenceScalperParams

    @classmethod
    def subscriptions(cls, params: StrategyParams) -> list[Subscription]:
        assert isinstance(params, MultiTFConfluenceScalperParams)
        return [
            Subscription(symbol=params.symbol, timeframe=params.entry_timeframe),
            Subscription(symbol=params.symbol, timeframe=params.mid_timeframe),
            Subscription(symbol=params.symbol, timeframe=params.high_timeframe),
        ]

    @classmethod
    def warmup_bars(
        cls, params: StrategyParams,
    ) -> dict[Subscription, int] | None:
        """Each feed gets enough bars to seed its slowest EMA."""
        assert isinstance(params, MultiTFConfluenceScalperParams)
        m1_need = max(
            params.m1_rsi_period + 1,
            params.atr_period + 1,
            params.vol_gate_lookback + 1,
        ) + 5
        m5_need = max(params.m5_ema_fast, params.m5_ema_slow) + 5
        m15_need = max(params.m15_ema_fast, params.m15_ema_slow) + 5
        return {
            Subscription(symbol=params.symbol, timeframe=params.entry_timeframe):
                m1_need,
            Subscription(symbol=params.symbol, timeframe=params.mid_timeframe):
                m5_need,
            Subscription(symbol=params.symbol, timeframe=params.high_timeframe):
                m15_need,
        }

    def __init__(self) -> None:
        super().__init__()
        self._open_ticket: int | None = None
        self._open_side: Side | None = None
        self._open_bars: int = 0
        self._cooldown_left: int = 0
        # Last M1 bar's RSI, updated on every M1 bar (even while a
        # position is open) so the entry cross test compares strictly
        # adjacent bars rather than a stale post-exit value.
        self._prev_m1_rsi: float | None = None

    # ------------------------------------------------------------------ #
    # Bar dispatch                                                        #
    # ------------------------------------------------------------------ #

    async def on_bar(self, ctx: StrategyContext, bar: Bar) -> None:
        params = ctx.params
        assert isinstance(params, MultiTFConfluenceScalperParams)

        # Multi-TF: only the entry tf drives decisions.  M5 / M15 still
        # feed their HistoryViews automatically via the runner.
        if bar.symbol != params.symbol or bar.timeframe != params.entry_timeframe:
            return

        # --- M1 RSI, tracked on EVERY M1 bar -------------------------------
        # Computed up front (before the gates) so `_prev_m1_rsi` always
        # holds the *immediately preceding* M1 bar's value.  That keeps the
        # cross test below strictly bar-to-bar even across bars where a
        # position is open or a gate fires — otherwise "previous" RSI could
        # be dozens of bars stale right after an exit, faking a cross.
        m1_view = ctx.history_for(params.symbol, params.entry_timeframe)
        if m1_view is None:
            return
        m1_bars = list(m1_view.bars())
        m1_rsi_v: float | None = None
        if len(m1_bars) >= params.m1_rsi_period + 1:
            m1_rsi_v = rsi([b.close for b in m1_bars], params.m1_rsi_period)
        prev_rsi = self._prev_m1_rsi
        self._prev_m1_rsi = m1_rsi_v

        # 1) Position open → advance hold counter + time exit.
        if self._open_ticket is not None:
            self._open_bars += 1
            await self._maybe_time_exit(ctx, params)
            return

        # 2) Cooldown.
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return

        # 3) Session-avoid gate.
        if bar.time.hour in params.avoid_hours_utc:
            return

        # 4) Enough history + a valid current/previous RSI pair.  No prior
        #    RSI → nothing to cross *from*, so we can't confirm a turn yet.
        need = max(
            params.vol_gate_lookback + 1,
            params.atr_period + 1,
            params.m1_rsi_period + 1,
        )
        if len(m1_bars) < need or m1_rsi_v is None or prev_rsi is None:
            return

        # 5) Vol gate (same shape as MBS).
        tr_series = _true_range_series(m1_bars)
        if len(tr_series) < params.vol_gate_lookback + 1:
            return
        tr_now = tr_series[-1]
        tr_baseline_series = tr_series[-(params.vol_gate_lookback + 1):-1]
        tr_baseline = sma(tr_baseline_series, params.vol_gate_lookback)
        if tr_baseline is None or tr_baseline <= 0:
            return
        if tr_now > params.vol_gate_mult * tr_baseline:
            return

        # 6) ATR for SL/TP distance.
        atr_v = atr(m1_bars, params.atr_period)
        if atr_v is None or atr_v <= 0:
            return
        sl_dist = atr_v * params.sl_atr_mult
        tp_dist = atr_v * params.tp_atr_mult if params.tp_atr_mult > 0 else None

        # 7) M5 trend filter.
        m5_view = ctx.history_for(params.symbol, params.mid_timeframe)
        if m5_view is None:
            return
        m5_bars = list(m5_view.bars())
        if len(m5_bars) < params.m5_ema_slow:
            return
        m5_closes = [b.close for b in m5_bars]
        m5_ema_fast = ema(m5_closes, params.m5_ema_fast)
        m5_ema_slow = ema(m5_closes, params.m5_ema_slow)
        if m5_ema_fast is None or m5_ema_slow is None:
            return
        m5_up = m5_ema_fast > m5_ema_slow
        m5_down = m5_ema_fast < m5_ema_slow

        # 8) M15 trend filter.
        m15_view = ctx.history_for(params.symbol, params.high_timeframe)
        if m15_view is None:
            return
        m15_bars = list(m15_view.bars())
        if len(m15_bars) < params.m15_ema_slow:
            return
        m15_closes = [b.close for b in m15_bars]
        m15_ema_fast = ema(m15_closes, params.m15_ema_fast)
        m15_ema_slow = ema(m15_closes, params.m15_ema_slow)
        if m15_ema_fast is None or m15_ema_slow is None:
            return
        m15_up = m15_ema_fast > m15_ema_slow
        m15_down = m15_ema_fast < m15_ema_slow

        # 9) BUY — both higher TFs UP + M1 RSI crossing UP out of the
        #    oversold zone (prev strictly below, current at/above).  The
        #    cross-back is the confirmation that the pullback has turned.
        crossed_up = prev_rsi < params.m1_rsi_oversold <= m1_rsi_v
        if m5_up and m15_up and crossed_up:
            entry = bar.close
            await ctx.buy(
                volume=params.volume,
                sl=entry - sl_dist,
                tp=(entry + tp_dist) if tp_dist is not None else None,
                comment="mtc_buy",
            )
            self._attach_trailing_if_enabled(ctx, params, atr_v)
            return

        # 10) SELL — both higher TFs DOWN + M1 RSI crossing DOWN out of
        #     the overbought zone (prev strictly above, current at/below).
        crossed_down = prev_rsi > params.m1_rsi_overbought >= m1_rsi_v
        if m5_down and m15_down and crossed_down:
            entry = bar.close
            await ctx.sell(
                volume=params.volume,
                sl=entry + sl_dist,
                tp=(entry - tp_dist) if tp_dist is not None else None,
                comment="mtc_sell",
            )
            self._attach_trailing_if_enabled(ctx, params, atr_v)

    # ------------------------------------------------------------------ #
    # Trailing-stop wiring                                                #
    # ------------------------------------------------------------------ #

    def _attach_trailing_if_enabled(
        self,
        ctx: StrategyContext,
        params: MultiTFConfluenceScalperParams,
        atr_value: float,
    ) -> None:
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

    async def _maybe_time_exit(
        self,
        ctx: StrategyContext,
        params: MultiTFConfluenceScalperParams,
    ) -> None:
        if (
            self._open_ticket is not None
            and self._open_bars >= params.max_hold_bars_m1
        ):
            await ctx.close(self._open_ticket, reason="mtc_time_exit")

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def on_order_filled(self, ctx: StrategyContext, order: Order) -> None:
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
        assert isinstance(params, MultiTFConfluenceScalperParams)
        self._open_ticket = None
        self._open_side = None
        self._open_bars = 0
        self._cooldown_left = params.cooldown_bars_m1
