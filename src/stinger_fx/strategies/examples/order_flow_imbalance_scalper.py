"""OrderFlowImbalanceScalper — retail-feasible OFI approximation on tick data.

Important disclaimer
====================

This is **NOT institutional HFT**.  Real HFT order-flow imbalance requires:

  * Level-2 order book messages (adds/cancels/executes) tagged with
    aggressor side.
  * Co-location and sub-ms latency.
  * Maker-rebate economics.

We have Dukascopy *quote* ticks (``bid``, ``ask``, ``volume`` = tick
count) with **no aggressor flag**.  So this strategy approximates the
aggressor side via the **quote-mid shift rule**:

  * Compute the mid-price ``m = (bid + ask) / 2`` for each tick.
  * If ``m > m_prev`` → up-tick → attribute the tick's volume to
    *buyer* aggression (signed +volume).
  * If ``m < m_prev`` → down-tick → *seller* aggression (signed
    −volume).
  * If equal → contribute 0.

Academic work (Lee-Ready 1991 and successors) reports ~30–40%
misclassification of this proxy vs the ground-truth aggressor side.
So realistic expectations on M1 retail data are **45–55% win rate,
RR 1–1.5:1**, not the 65–75% × 2:1 of true HFT.  Treat the strategy
as a tick-volume-informed scalper with directional bias from a noisy
proxy, and treat the institutional comparison as inspiration only.

Concept
=======

The strategy fires on M1 bar close — the same trigger cadence as
:mod:`momentum_breakout_scalper` so spread cost stays comparable.

BUY when:

  * The signed-volume deque (built by ``on_tick``) sums to
    ``> +ofi_threshold`` over the last ``ofi_window_seconds``.
  * Optional M5 trend filter agrees (EMA-fast > EMA-slow).
  * Session-avoid / vol-gate / cooldown all pass.

SELL is symmetric.  Exits, sizing, and trailing all reuse the MBS
shape (ATR SL/TP + time stop + optional trailing manager) so the
two strategies are directly A/B-testable on the same harness.
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import ClassVar

from pydantic import Field

from stinger_fx.domain import Bar, Order, Position, Side, Timeframe
from stinger_fx.domain.symbols import Subscription
from stinger_fx.domain.ticks import Tick
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.indicators import atr, ema, sma
from stinger_fx.strategies.managers.trailing import TrailingStopManager
from stinger_fx.strategies.parameters import StrategyParams

logger = logging.getLogger("stinger.strategies.order_flow_imbalance_scalper")


class OrderFlowImbalanceScalperParams(StrategyParams):
    """Tunables for the OFI-approximation scalper.  Defaults mirror the
    winning MBS round-2 combo for the non-OFI knobs, so an A/B test
    isolates the impact of the OFI signal itself."""

    # --- Feeds ----------------------------------------------------------
    symbol: str = "XAUUSD"
    entry_timeframe: Timeframe = Timeframe.M1
    structure_timeframe: Timeframe = Timeframe.M5

    # --- OFI proxy (quote-mid shift rule) -------------------------------
    # Maximum samples kept in the signed-volume deque.  Acts as a hard
    # cap so memory + per-bar aggregation stay bounded even if the
    # window-seconds prune is loose.
    ofi_lookback_ticks: int = Field(200, ge=10, le=10_000)
    # Drop samples older than this on every read.  Prevents stale-data
    # leakage in low-tick-rate periods (e.g. Asian session lulls).
    ofi_window_seconds: int = Field(60, ge=1, le=3600)
    # |signed-volume sum| must exceed this to fire.  Units = number of
    # ticks (since Dukascopy ``tick.volume`` is a count, not size).
    ofi_threshold: float = Field(200.0, gt=0)

    # --- M5 trend filter -----------------------------------------------
    require_m5_trend: bool = True
    ema_fast_period: int = Field(20, ge=2, le=200)
    ema_slow_period: int = Field(50, ge=5, le=400)

    # --- Session avoidance ---------------------------------------------
    avoid_hours_utc: list[int] = Field(default_factory=lambda: [13, 14, 15, 18])

    # --- Volatility gate ------------------------------------------------
    vol_gate_lookback: int = Field(50, ge=2, le=500)
    vol_gate_mult: float = Field(1.5, gt=0)

    # --- Risk + sizing -------------------------------------------------
    volume: float = Field(0.01, gt=0)
    atr_period: int = Field(14, ge=2, le=50)
    sl_atr_mult: float = Field(1.0, gt=0)
    # Default 2.0 reflects the MBS round-2 winner — asymmetric RR
    # compensates for noisy signal + time-stop interim closes.
    tp_atr_mult: float = Field(2.0, ge=0)
    max_hold_bars_m1: int = Field(30, ge=1, le=10_000)
    cooldown_bars_m1: int = Field(5, ge=0, le=10_000)

    # --- Trailing stop (optional) --------------------------------------
    enable_trailing: bool = False
    trailing_distance_atr: float = Field(0.8, gt=0)
    trailing_activate_atr: float = Field(0.5, ge=0)
    trailing_point: float = Field(0.01, gt=0)


def _true_range_series(bars: list[Bar]) -> list[float]:
    """Per-bar True Range = max(H-L, |H - prev_close|, |L - prev_close|).

    Copy of the helper from ``momentum_breakout_scalper`` — keeping it
    inline rather than importing across strategies so each strategy's
    dependencies stay obvious from a single file read.
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


class OrderFlowImbalanceScalper(BaseStrategy):
    name: ClassVar[str] = "order_flow_imbalance_scalper"
    Params: ClassVar[type[StrategyParams]] = OrderFlowImbalanceScalperParams

    @classmethod
    def subscriptions(cls, params: StrategyParams) -> list[Subscription]:
        assert isinstance(params, OrderFlowImbalanceScalperParams)
        return [
            Subscription(symbol=params.symbol, timeframe=params.entry_timeframe),
            Subscription(symbol=params.symbol, timeframe=params.structure_timeframe),
        ]

    @classmethod
    def warmup_bars(
        cls, params: StrategyParams,
    ) -> dict[Subscription, int] | None:
        """Per-feed warmup — same shape as MBS minus the Donchian window."""
        assert isinstance(params, OrderFlowImbalanceScalperParams)
        m1_need = max(params.atr_period + 1, params.vol_gate_lookback + 1) + 5
        m5_need = max(params.ema_slow_period, params.ema_fast_period) + 5
        return {
            Subscription(symbol=params.symbol, timeframe=params.entry_timeframe):
                m1_need,
            Subscription(symbol=params.symbol, timeframe=params.structure_timeframe):
                m5_need,
        }

    def __init__(self) -> None:
        super().__init__()
        # Signed-volume deque populated by ``on_tick``.  Each entry is
        # ``(tick.time, signed_volume)``.  ``deque(maxlen=...)`` enforces
        # the hard cap; we apply a time-based prune on every read for
        # the soft window.
        self._ofi_deque: deque[tuple[datetime, int]] = deque(maxlen=200)
        # Previous mid-price seen on the entry-symbol — used to assign
        # the sign on the *next* tick.  None until the first tick.
        self._prev_mid: float | None = None
        # Position state — identical semantics to MBS / PRS.
        self._open_ticket: int | None = None
        self._open_side: Side | None = None
        self._open_bars: int = 0
        self._cooldown_left: int = 0

    # ------------------------------------------------------------------ #
    # Tick accumulator                                                    #
    # ------------------------------------------------------------------ #

    async def on_tick(self, ctx: StrategyContext, tick: Tick) -> None:
        """Build the signed-volume deque.

        Only ticks on the strategy's primary symbol are accumulated —
        multi-symbol setups must not cross-pollinate.  Volume comes from
        ``tick.volume`` (Dukascopy stores tick-count there); we apply
        the sign via the mid-price shift rule.
        """
        params = ctx.params
        assert isinstance(params, OrderFlowImbalanceScalperParams)
        if tick.symbol != params.symbol:
            return

        # Resize the deque maxlen to match the param at runtime — params
        # are hot-reloadable, so the user might bump the cap.  Cheap:
        # only fires when the cap actually differs.
        if self._ofi_deque.maxlen != params.ofi_lookback_ticks:
            self._ofi_deque = deque(self._ofi_deque, maxlen=params.ofi_lookback_ticks)

        cur_mid = tick.mid
        prev_mid = self._prev_mid
        self._prev_mid = cur_mid
        if prev_mid is None:
            # First-ever tick — no direction to assign.  Seed and return.
            return
        if cur_mid > prev_mid:
            sign = 1
        elif cur_mid < prev_mid:
            sign = -1
        else:
            sign = 0
        # tick.volume may be 0 in extremely quiet periods; that becomes
        # a zero contribution, which is fine.
        self._ofi_deque.append((tick.time, sign * tick.volume))

    # ------------------------------------------------------------------ #
    # Bar dispatch                                                        #
    # ------------------------------------------------------------------ #

    async def on_bar(self, ctx: StrategyContext, bar: Bar) -> None:
        params = ctx.params
        assert isinstance(params, OrderFlowImbalanceScalperParams)

        # Multi-TF: only the entry tf drives decisions.
        if bar.symbol != params.symbol or bar.timeframe != params.entry_timeframe:
            return

        # 1) Position open → advance hold counter, check time exit.
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

        # 4) M1 history checks for ATR + vol-gate baseline.
        m1_view = ctx.history_for(params.symbol, params.entry_timeframe)
        if m1_view is None:
            return
        m1_bars = list(m1_view.bars())
        need = max(params.vol_gate_lookback + 1, params.atr_period + 1)
        if len(m1_bars) < need:
            return

        # 5) Vol gate — current TR vs SMA(TR, lookback) of prior bars.
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

        # 7) Compute OFI sum over the freshness window.
        ofi_sum = self._compute_ofi_sum(bar.time, params)

        # 8) Optional M5 trend filter.
        m5_buy_ok = True
        m5_sell_ok = True
        if params.require_m5_trend:
            m5_view = ctx.history_for(params.symbol, params.structure_timeframe)
            if m5_view is None:
                return
            m5_bars = list(m5_view.bars())
            if len(m5_bars) < params.ema_slow_period:
                return
            m5_closes = [b.close for b in m5_bars]
            ema_fast = ema(m5_closes, params.ema_fast_period)
            ema_slow = ema(m5_closes, params.ema_slow_period)
            if ema_fast is None or ema_slow is None:
                return
            m5_buy_ok = ema_fast > ema_slow
            m5_sell_ok = ema_fast < ema_slow

        # 9) BUY when OFI strongly positive.
        if m5_buy_ok and ofi_sum > params.ofi_threshold:
            entry = bar.close
            await ctx.buy(
                volume=params.volume,
                sl=entry - sl_dist,
                tp=(entry + tp_dist) if tp_dist is not None else None,
                comment="ofis_buy",
            )
            self._attach_trailing_if_enabled(ctx, params, atr_v)
            return

        # 10) SELL when OFI strongly negative.
        if m5_sell_ok and ofi_sum < -params.ofi_threshold:
            entry = bar.close
            await ctx.sell(
                volume=params.volume,
                sl=entry + sl_dist,
                tp=(entry - tp_dist) if tp_dist is not None else None,
                comment="ofis_sell",
            )
            self._attach_trailing_if_enabled(ctx, params, atr_v)

    # ------------------------------------------------------------------ #
    # OFI helper                                                          #
    # ------------------------------------------------------------------ #

    def _compute_ofi_sum(
        self,
        now: datetime,
        params: OrderFlowImbalanceScalperParams,
    ) -> int:
        """Sum signed-volume entries inside the freshness window.

        Stale entries (older than ``ofi_window_seconds`` before ``now``)
        are pruned from the LEFT of the deque so subsequent calls don't
        keep re-scanning them.  The deque stays time-ordered by
        construction (``on_tick`` appends in chronological tick order),
        so left-popping until the head is fresh is O(stale-count)
        amortised.
        """
        cutoff = now - timedelta(seconds=params.ofi_window_seconds)
        while self._ofi_deque and self._ofi_deque[0][0] < cutoff:
            self._ofi_deque.popleft()
        return sum(v for _, v in self._ofi_deque)

    # ------------------------------------------------------------------ #
    # Trailing-stop wiring                                                #
    # ------------------------------------------------------------------ #

    def _attach_trailing_if_enabled(
        self,
        ctx: StrategyContext,
        params: OrderFlowImbalanceScalperParams,
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
        params: OrderFlowImbalanceScalperParams,
    ) -> None:
        if (
            self._open_ticket is not None
            and self._open_bars >= params.max_hold_bars_m1
        ):
            await ctx.close(self._open_ticket, reason="ofis_time_exit")

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
        assert isinstance(params, OrderFlowImbalanceScalperParams)
        self._open_ticket = None
        self._open_side = None
        self._open_bars = 0
        self._cooldown_left = params.cooldown_bars_m1
