"""MomentumBreakoutScalper — Donchian breakout + session/vol gates + ATR TP.

Concept
=======

The opposite premise of :mod:`pullback_reversal_scalper`. Where PRS buys
the dip in an oversold M1, this one buys *strength* — a fresh M1 high
that breaks the prior N-bar range — and only when the larger M5 trend
agrees. The sweep on May-2026 XAU/USD showed PRS's mean-reversion
entries get steam-rolled by momentum continuation; this strategy
trades *with* that momentum instead.

BUY setup
  * M1 ``bar.close > donchian(previous N bars).upper`` — genuine new high
    (the current bar is excluded from the Donchian window, otherwise the
    test would always be ``close ≤ bar.high``).
  * Optional M5 trend filter (``require_m5_trend``): EMA-fast > EMA-slow.
  * Bar's UTC hour NOT in ``avoid_hours_utc`` — skips London-PM / NY-open
    spike windows (loss cluster on XAU was 13–18 UTC).
  * Current ATR < ``vol_gate_mult × sma(True-Range, vol_gate_lookback)``
    — drops the spike candles that get SL'd within seconds.

SELL setup
  Symmetric: close < lower Donchian, M5 trend down, same gates.

Exits
=====

Three independent paths; whichever fires first wins:

* **Fixed ATR take-profit** — ``entry ± tp_atr_mult × atr_at_entry``.
  Set on the order so the broker handles it. Default ``1.5`` so the
  reward is consistently above the typical ``1.0 × ATR`` stop.
* **Initial protective SL** — ``entry ± sl_atr_mult × atr_at_entry``.
* **Time stop** — ``max_hold_bars_m1`` M1 bars (safety net for sideways
  drift that hits neither TP nor SL).
* **Optional trailing** — :class:`TrailingStopManager` attached at
  entry. OFF by default; same wiring as PRS.

Position-state machine + cooldown semantics are intentionally identical
to PRS so the two strategies can be backtested on the same harness.
"""
from __future__ import annotations

import logging
from typing import ClassVar

from pydantic import Field

from stinger_fx.domain import Bar, Order, Position, Side, Timeframe
from stinger_fx.domain.symbols import Subscription
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.indicators import atr, donchian, ema, sma
from stinger_fx.strategies.managers.trailing import TrailingStopManager
from stinger_fx.strategies.parameters import StrategyParams

logger = logging.getLogger("stinger.strategies.momentum_breakout_scalper")


class MomentumBreakoutScalperParams(StrategyParams):
    """Tunables for the momentum-breakout scalper. Defaults are an
    XAU/USD-friendly starting point picked to mirror the PRS knobs we
    swept — re-optimise per symbol / session."""

    # --- Feeds ----------------------------------------------------------
    symbol: str = "XAUUSD"
    entry_timeframe: Timeframe = Timeframe.M1
    structure_timeframe: Timeframe = Timeframe.M5

    # --- Breakout entry -------------------------------------------------
    # Donchian length on the entry timeframe. The current bar is excluded
    # from the window so ``bar.close > upper`` actually fires on a new
    # high rather than degenerating to ``close <= bar.high``.
    donchian_period: int = Field(20, ge=2, le=500)

    # --- M5 trend filter ------------------------------------------------
    # When True, BUY requires EMA-fast > EMA-slow on the structure feed
    # (and the inverse for SELL). When False, the strategy fires on M1
    # breakouts alone — useful for backtesting the breakout signal in
    # isolation.
    require_m5_trend: bool = True
    ema_fast_period: int = Field(20, ge=2, le=200)
    ema_slow_period: int = Field(50, ge=5, le=400)

    # --- Session avoidance ---------------------------------------------
    # UTC hours during which no NEW entry is permitted. Default skips the
    # London-PM (13–15) and NY-mid (18) spike windows where the worst PRS
    # losses clustered. Empty list = no session filter.
    avoid_hours_utc: list[int] = Field(default_factory=lambda: [13, 14, 15, 18])

    # --- Volatility gate ------------------------------------------------
    # Skip the entry when the current bar's True Range is more than
    # ``vol_gate_mult`` × the SMA of TR over ``vol_gate_lookback`` prior
    # M1 bars. Set ``vol_gate_mult`` to a large value (e.g. 99) to
    # effectively disable.
    vol_gate_lookback: int = Field(50, ge=2, le=500)
    vol_gate_mult: float = Field(1.8, gt=0)

    # --- Risk + sizing -------------------------------------------------
    volume: float = Field(0.01, gt=0)
    atr_period: int = Field(14, ge=2, le=50)
    sl_atr_mult: float = Field(1.0, gt=0)
    tp_atr_mult: float = Field(1.5, ge=0)         # 0 = no fixed TP
    max_hold_bars_m1: int = Field(30, ge=1, le=10_000)
    cooldown_bars_m1: int = Field(5, ge=0, le=10_000)

    # --- Trailing stop (optional) --------------------------------------
    enable_trailing: bool = False
    trailing_distance_atr: float = Field(0.8, gt=0)
    trailing_activate_atr: float = Field(0.5, ge=0)
    trailing_point: float = Field(0.01, gt=0)


def _true_range_series(bars: list[Bar]) -> list[float]:
    """Per-bar True Range = max(H-L, |H - prev_close|, |L - prev_close|).

    Inline helper because the indicators package only exposes the
    smoothed ATR — the vol gate wants the raw TR distribution so the
    SMA-baseline comparison isn't itself smoothed.
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


class MomentumBreakoutScalper(BaseStrategy):
    name: ClassVar[str] = "momentum_breakout_scalper"
    Params: ClassVar[type[StrategyParams]] = MomentumBreakoutScalperParams

    @classmethod
    def subscriptions(cls, params: StrategyParams) -> list[Subscription]:
        assert isinstance(params, MomentumBreakoutScalperParams)
        return [
            Subscription(symbol=params.symbol, timeframe=params.entry_timeframe),
            Subscription(symbol=params.symbol, timeframe=params.structure_timeframe),
        ]

    @classmethod
    def warmup_bars(
        cls, params: StrategyParams,
    ) -> dict[Subscription, int] | None:
        """Per-feed warmup for live-mode startup backfill.

        M1 needs enough bars for the longest of: Donchian window (+1 for
        the excluded current bar), ATR period, and the vol-gate lookback.
        M5 needs the slow EMA window.
        """
        assert isinstance(params, MomentumBreakoutScalperParams)
        m1_need = max(
            params.donchian_period + 1,
            params.atr_period + 1,
            params.vol_gate_lookback + 1,
        ) + 5
        m5_need = max(params.ema_slow_period, params.ema_fast_period) + 5
        return {
            Subscription(symbol=params.symbol, timeframe=params.entry_timeframe):
                m1_need,
            Subscription(symbol=params.symbol, timeframe=params.structure_timeframe):
                m5_need,
        }

    def __init__(self) -> None:
        super().__init__()
        # Open-position tracking — same semantics as PRS.
        self._open_ticket: int | None = None
        self._open_side: Side | None = None
        self._open_bars: int = 0
        self._cooldown_left: int = 0

    # ------------------------------------------------------------------ #
    # Bar dispatch                                                        #
    # ------------------------------------------------------------------ #

    async def on_bar(self, ctx: StrategyContext, bar: Bar) -> None:
        params = ctx.params
        assert isinstance(params, MomentumBreakoutScalperParams)

        # Multi-TF: only the entry timeframe drives decisions. M5 bars
        # still feed their HistoryView automatically via the runner.
        if bar.symbol != params.symbol or bar.timeframe != params.entry_timeframe:
            return

        # 1) If a position is open, advance hold counter + check time
        #    exit. Other exits (SL/TP) are broker-side.
        if self._open_ticket is not None:
            self._open_bars += 1
            await self._maybe_time_exit(ctx, params)
            return

        # 2) Cooldown after the previous close.
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return

        # 3) Session-avoid gate — UTC hour check on the bar's timestamp.
        if bar.time.hour in params.avoid_hours_utc:
            return

        # 4) Read M1 history for Donchian + ATR + TR series.
        m1_view = ctx.history_for(params.symbol, params.entry_timeframe)
        if m1_view is None:
            return
        m1_bars = list(m1_view.bars())
        # Largest required window: Donchian needs `period + 1` (we exclude
        # the current bar), TR baseline needs `lookback + 1` pairs, ATR
        # needs `atr_period + 1`. Bail until all are satisfied.
        need = max(
            params.donchian_period + 1,
            params.vol_gate_lookback + 1,
            params.atr_period + 1,
        )
        if len(m1_bars) < need:
            return

        # 5) Donchian on the PRIOR bars only — exclude the current bar so
        #    `bar.close > upper` actually means a new high.
        breakout = donchian(m1_bars[:-1], params.donchian_period)
        if breakout is None:
            return

        # 6) Volatility gate — current TR vs SMA(TR, lookback) of prior bars.
        #    Skip when current TR exceeds the spike threshold.
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

        # 7) ATR for SL/TP distance — same series as PRS uses.
        atr_v = atr(m1_bars, params.atr_period)
        if atr_v is None or atr_v <= 0:
            return
        sl_dist = atr_v * params.sl_atr_mult
        tp_dist = atr_v * params.tp_atr_mult if params.tp_atr_mult > 0 else None

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

        # 9) BUY — breakout above prior N-bar high.
        if m5_buy_ok and bar.close > breakout.upper:
            entry = bar.close
            await ctx.buy(
                volume=params.volume,
                sl=entry - sl_dist,
                tp=(entry + tp_dist) if tp_dist is not None else None,
                comment="mbs_buy",
            )
            self._attach_trailing_if_enabled(ctx, params, atr_v)
            return

        # 10) SELL — symmetric breakdown.
        if m5_sell_ok and bar.close < breakout.lower:
            entry = bar.close
            await ctx.sell(
                volume=params.volume,
                sl=entry + sl_dist,
                tp=(entry - tp_dist) if tp_dist is not None else None,
                comment="mbs_sell",
            )
            self._attach_trailing_if_enabled(ctx, params, atr_v)

    # ------------------------------------------------------------------ #
    # Trailing-stop wiring                                                #
    # ------------------------------------------------------------------ #

    def _attach_trailing_if_enabled(
        self,
        ctx: StrategyContext,
        params: MomentumBreakoutScalperParams,
        atr_value: float,
    ) -> None:
        """Same wiring PRS uses — distance + activation in ATR units,
        converted to pips via the symbol's point value."""
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
        params: MomentumBreakoutScalperParams,
    ) -> None:
        """Close the open position when the safety hold-time elapses.

        Broker handles SL/TP fills directly; we only need to enforce the
        time stop here so a trade that never reaches either side of the
        ATR band doesn't sit forever.
        """
        if (
            self._open_ticket is not None
            and self._open_bars >= params.max_hold_bars_m1
        ):
            await ctx.close(self._open_ticket, reason="mbs_time_exit")

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def on_order_filled(self, ctx: StrategyContext, order: Order) -> None:
        # Only track tickets our own strategy opened — multi-strategy
        # setups may share a broker.
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
        assert isinstance(params, MomentumBreakoutScalperParams)
        self._open_ticket = None
        self._open_side = None
        self._open_bars = 0
        self._cooldown_left = params.cooldown_bars_m1
