"""LiquiditySweepReversal — sideway-regime liquidity-sweep reversal.

Designed against XAU/USD but symbol-agnostic: every threshold scales by
the symbol's own ATR (sweep_buffer, reentry_buffer, stop_buffer, range
width gates), so the same params transplant to any FX pair, index, or
crypto that exhibits the same intraday range → false-breakout → revert
behaviour. Tune ``atr_period`` / ``range_lookback_bars`` for the
instrument's typical bar-rate; tune the ``*_atr`` multipliers for its
typical false-breakout overshoot.

Setup family
============

The strategy enters *against* a brief breakout (a "sweep") when the
market wicks past a recent intraday range and then closes back inside.
The thesis: in a ranging market, breakout attempts that fail to hold
deliver counter-trend setups back toward the range midpoint (or session
equilibrium price).

Three timeframes wire the gates together:

  * **M15** — regime filter. ADX must be below ``max_adx`` (ranging
    market). Trend regimes are skipped entirely because breakouts in
    trends tend to *succeed* and the reversal entry becomes a loss.
  * **M5**  — structure. The recent ``range_lookback_bars`` of closed M5
    bars defines the range. Donchian over that window gives us
    ``range_high`` / ``range_mid`` / ``range_low``. ATR-scaled minimum
    and maximum width filters out boringly-tight and explosively-wide
    ranges where the edge degrades.
  * **M1**  — entry trigger. A single M1 bar must sweep above
    ``range_high + sweep_buffer * atr`` (or below ``range_low - ...``)
    AND close back inside the range with at least ``reentry_buffer * atr``
    of room. Stop sits just past the sweep wick; take-profit picks one of
    three modes (see ``tp_mode``).

Risk discipline (per-strategy, configurable)
============================================

* Max one position per ``symbol`` at a time.
* ``max_trades_per_session`` hard caps daily turnover.
* ``cooldown_bars`` of the entry timeframe must elapse after the previous
  close before a new entry fires.
* Trades with computed reward-to-risk below ``min_rr`` are skipped.
* Open positions are force-closed via ``ctx.close()`` after
  ``max_hold_bars`` of the entry timeframe — sweep reversals are
  inherently short-horizon and a position lingering past that window
  usually means the setup is invalidated even if SL hasn't tripped yet.

Account-level guards (``RiskMonitor`` per-strategy caps, daily-loss
limits, kill-switch) remain the engine's responsibility, not this
strategy's.

v1 scope
========

* ``tp_mode = "vwap"`` falls back to ``mid`` when ``vwap_session`` returns
  None (zero-volume warmup edge case). Tested.
* ``avoid_news`` parameter is reserved as a future hook; v1 ignores it
  (no calendar service in the framework yet).
* No ``BreakEvenManager`` attachment in v1 — fixed SL/TP + time exit
  only. Adding break-even is a small follow-up once backtest expectancy
  is validated.
* No partial close / runner-style exits — exits are binary (SL, TP, or
  time).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import Field

from stinger_fx.domain import Bar, Order, Position, Side, Subscription, Timeframe
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.indicators import (
    DonchianChannels,
    adx,
    atr,
    donchian,
    vwap_session,
)
from stinger_fx.strategies.parameters import StrategyParams


class LiquiditySweepReversalParams(StrategyParams):
    """Tunables. Defaults are XAUUSD-friendly starting points (the
    strategy was originally designed around gold's intraday range
    geometry), but every threshold is ATR-scaled, so the same defaults
    are a reasonable starting point on any liquid intraday symbol.
    Expect to re-optimise per instrument, session, and broker spread."""

    # --- Feed identity --------------------------------------------------
    symbol: str = "XAUUSD"
    entry_timeframe: Timeframe = Timeframe.M1
    structure_timeframe: Timeframe = Timeframe.M5
    regime_timeframe: Timeframe = Timeframe.M15

    # --- Range structure (M5) -------------------------------------------
    range_lookback_bars: int = Field(36, ge=10, le=200)
    min_range_atr: float = Field(0.8, gt=0)
    max_range_atr: float = Field(4.0, gt=0)

    # --- Regime filter (M15 ADX) ----------------------------------------
    adx_period: int = Field(14, ge=5, le=50)
    max_adx: float = Field(22.0, gt=0)

    # --- ATR-scaled buffers ---------------------------------------------
    atr_period: int = Field(14, ge=5, le=50)
    sweep_buffer_atr: float = Field(0.08, ge=0)
    reentry_buffer_atr: float = Field(0.03, ge=0)
    stop_buffer_atr: float = Field(0.12, gt=0)

    # --- Exit policy ----------------------------------------------------
    tp_mode: Literal["mid", "vwap", "fixed_r"] = "mid"
    take_profit_r: float = Field(1.2, gt=0)
    min_rr: float = Field(0.8, gt=0)

    # --- Sizing / cadence -----------------------------------------------
    volume: float = Field(0.01, gt=0)
    max_hold_bars: int = Field(8, ge=1)
    cooldown_bars: int = Field(5, ge=0)
    max_trades_per_session: int = Field(4, ge=1)

    # --- Time-of-day session gate (UTC) ---------------------------------
    use_session_filter: bool = True
    session_start_hour_utc: int = Field(0, ge=0, le=23)
    session_end_hour_utc: int = Field(16, ge=1, le=24)

    # --- Future hooks (v1: no-op) ---------------------------------------
    avoid_news: bool = False


class LiquiditySweepReversal(BaseStrategy):
    name = "liquidity_sweep_reversal"
    Params = LiquiditySweepReversalParams

    @classmethod
    def subscriptions(cls, params: StrategyParams) -> list[Subscription]:
        assert isinstance(params, LiquiditySweepReversalParams)
        return [
            Subscription(symbol=params.symbol, timeframe=params.entry_timeframe),
            Subscription(symbol=params.symbol, timeframe=params.structure_timeframe),
            Subscription(symbol=params.symbol, timeframe=params.regime_timeframe),
        ]

    def __init__(self) -> None:
        super().__init__()
        # Cooldown anchor — index of the entry-tf bar at which the most
        # recent position closed. None until at least one close has fired.
        self._last_close_bar_index: int | None = None
        # Per-session counter; reset on UTC-date rollover.
        self._trades_this_session: int = 0
        self._session_key: str | None = None
        # Ticket → entry-tf bar index at fill time, used by max_hold_bars
        # forced close. Pop on `on_position_closed`.
        self._entry_bar_by_ticket: dict[int, int] = {}
        # Monotonic counter over entry-tf bars; the unit for cooldown_bars,
        # max_hold_bars.
        self._bar_index: int = 0

    # ------------------------------------------------------------------ #
    # Bar dispatch                                                        #
    # ------------------------------------------------------------------ #

    async def on_bar(self, ctx: StrategyContext, bar: Bar) -> None:
        params = ctx.params
        assert isinstance(params, LiquiditySweepReversalParams)

        # Multi-TF: ignore M5/M15 bars; only act on the entry timeframe for
        # the configured symbol. The HistoryView for each subscription is
        # still updated by the runner, so we can read M5/M15 history below.
        if bar.symbol != params.symbol or bar.timeframe != params.entry_timeframe:
            return
        self._bar_index += 1

        # Roll session counters at UTC midnight (or custom session start).
        self._maybe_roll_session(bar.time, params)

        # Time-based exit on any positions we already opened. This runs
        # before entry checks so that a position closed by time on this
        # bar can still allow a fresh setup to fire next bar (subject to
        # cooldown).
        await self._maybe_time_exit(ctx)

        # --- Entry gates --------------------------------------------------
        if params.use_session_filter and not self._in_session(bar.time, params):
            return
        if self._trades_this_session >= params.max_trades_per_session:
            return
        if self._on_cooldown(params):
            return
        if ctx.position.for_symbol(params.symbol):
            # Max one position per symbol; don't double up.
            return

        # --- Multi-TF data fetch + warmup check --------------------------
        m5 = ctx.history_for(params.symbol, params.structure_timeframe)
        m15 = ctx.history_for(params.symbol, params.regime_timeframe)
        if m5 is None or m15 is None:
            return
        m5_bars = m5.bars()
        m15_bars = m15.bars()
        if len(m5_bars) < params.range_lookback_bars + 1:
            return
        if len(m15_bars) < 2 * params.adx_period:
            return

        # --- Regime: M15 ADX must indicate ranging market ---------------
        adx_result = adx(m15_bars, params.adx_period)
        if adx_result is None or adx_result.adx > params.max_adx:
            return

        # --- Structure: Donchian on closed M5 bars only ------------------
        # Slice m5_bars[:-1] before passing to donchian() so the most-
        # recent M5 bar (which may still be the same minute the M1 sweep
        # is forming inside) can't leak into the range the M1 trigger
        # has to break. This is the canonical lookahead-bias guard.
        structure_window = m5_bars[:-1][-params.range_lookback_bars:]
        if len(structure_window) < params.range_lookback_bars:
            return
        channels = donchian(structure_window, params.range_lookback_bars)
        if channels is None:
            return

        # --- ATR scale + width filter -----------------------------------
        atr_value = atr(m5_bars, params.atr_period)
        if atr_value is None or atr_value <= 0:
            return
        range_width = channels.upper - channels.lower
        if range_width < params.min_range_atr * atr_value:
            return
        if range_width > params.max_range_atr * atr_value:
            return

        # --- Build setup -------------------------------------------------
        sweep_buf = params.sweep_buffer_atr * atr_value
        reentry_buf = params.reentry_buffer_atr * atr_value
        stop_buf = params.stop_buffer_atr * atr_value

        await self._maybe_enter_short(
            ctx, params, bar, channels, sweep_buf, reentry_buf, stop_buf, m5_bars,
        )
        # The two enter helpers are mutually exclusive in practice (a
        # single bar can't sweep both high and low), so the short check
        # short-circuits cheaply when no setup is present.
        if ctx.position.for_symbol(params.symbol):
            return
        await self._maybe_enter_long(
            ctx, params, bar, channels, sweep_buf, reentry_buf, stop_buf, m5_bars,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def on_order_filled(self, ctx: StrategyContext, order: Order) -> None:
        # Track ticket → entry-bar index so the time-exit can find this
        # position later. The order-router emits OrderFilledEvent for our
        # strategy_id only; the magic check on ctx.position handles cross-
        # strategy noise.
        if order.strategy_id == ctx.strategy_id and order.ticket:
            self._entry_bar_by_ticket[order.ticket] = self._bar_index

    async def on_position_closed(self, ctx: StrategyContext, position: Position) -> None:
        # Drop the ticket from the time-exit map, set cooldown anchor,
        # and tick the per-session counter so max_trades_per_session
        # captures both wins and losses.
        self._entry_bar_by_ticket.pop(position.ticket, None)
        self._last_close_bar_index = self._bar_index
        self._trades_this_session += 1

    # ------------------------------------------------------------------ #
    # Entry helpers                                                       #
    # ------------------------------------------------------------------ #

    async def _maybe_enter_short(
        self,
        ctx: StrategyContext,
        params: LiquiditySweepReversalParams,
        bar: Bar,
        channels: DonchianChannels,
        sweep_buf: float,
        reentry_buf: float,
        stop_buf: float,
        m5_bars: tuple[Bar, ...],
    ) -> None:
        # Sweep up through the range high, then close back below it.
        swept = bar.high >= channels.upper + sweep_buf
        re_entered = bar.close <= channels.upper - reentry_buf
        if not (swept and re_entered):
            return

        entry = bar.close
        sl = bar.high + stop_buf
        tp = self._compute_tp(
            params=params, side=Side.SELL, entry=entry, sl=sl,
            channels=channels, m5_bars=m5_bars, bar_time=bar.time,
        )
        if tp is None or not self._rr_ok(entry=entry, sl=sl, tp=tp, side=Side.SELL, params=params):
            return
        await ctx.sell(params.volume, sl=sl, tp=tp, comment="sweep_reversal_short")

    async def _maybe_enter_long(
        self,
        ctx: StrategyContext,
        params: LiquiditySweepReversalParams,
        bar: Bar,
        channels: DonchianChannels,
        sweep_buf: float,
        reentry_buf: float,
        stop_buf: float,
        m5_bars: tuple[Bar, ...],
    ) -> None:
        swept = bar.low <= channels.lower - sweep_buf
        re_entered = bar.close >= channels.lower + reentry_buf
        if not (swept and re_entered):
            return

        entry = bar.close
        sl = bar.low - stop_buf
        tp = self._compute_tp(
            params=params, side=Side.BUY, entry=entry, sl=sl,
            channels=channels, m5_bars=m5_bars, bar_time=bar.time,
        )
        if tp is None or not self._rr_ok(entry=entry, sl=sl, tp=tp, side=Side.BUY, params=params):
            return
        await ctx.buy(params.volume, sl=sl, tp=tp, comment="sweep_reversal_long")

    # ------------------------------------------------------------------ #
    # TP / RR helpers                                                     #
    # ------------------------------------------------------------------ #

    def _compute_tp(
        self,
        *,
        params: LiquiditySweepReversalParams,
        side: Side,
        entry: float,
        sl: float,
        channels: DonchianChannels,
        m5_bars: tuple[Bar, ...],
        bar_time: datetime,
    ) -> float | None:
        if params.tp_mode == "mid":
            return channels.middle
        if params.tp_mode == "fixed_r":
            risk = abs(sl - entry)
            if risk <= 0:
                return None
            if side is Side.SELL:
                return entry - params.take_profit_r * risk
            return entry + params.take_profit_r * risk
        # tp_mode == "vwap": anchor at the most recent session start, take
        # VWAP over the bars since that boundary. Falls through to "mid"
        # when there isn't enough session data yet (warmup, or all-zero
        # volume bars).
        boundary = self._session_start(bar_time, params.session_start_hour_utc)
        session_bars = [b for b in m5_bars if b.time >= boundary]
        vwap_value = vwap_session(session_bars)
        if vwap_value is None:
            return channels.middle
        return vwap_value

    @staticmethod
    def _rr_ok(
        *, entry: float, sl: float, tp: float, side: Side,
        params: LiquiditySweepReversalParams,
    ) -> bool:
        risk = abs(sl - entry)
        reward = (entry - tp) if side is Side.SELL else (tp - entry)
        if risk <= 0 or reward <= 0:
            return False
        return (reward / risk) >= params.min_rr

    # ------------------------------------------------------------------ #
    # Session / cooldown bookkeeping                                      #
    # ------------------------------------------------------------------ #

    def _maybe_roll_session(
        self, now: datetime, params: LiquiditySweepReversalParams,
    ) -> None:
        key = self._session_start(now, params.session_start_hour_utc).date().isoformat()
        if self._session_key != key:
            self._session_key = key
            self._trades_this_session = 0

    @staticmethod
    def _session_start(now: datetime, session_start_hour_utc: int) -> datetime:
        """Return the datetime of the session boundary immediately preceding
        ``now``. Anchors at ``session_start_hour_utc`` UTC each day; if
        ``now`` is before that hour today, the boundary is yesterday's."""
        boundary = now.replace(
            hour=session_start_hour_utc, minute=0, second=0, microsecond=0,
        )
        if now < boundary:
            boundary -= timedelta(days=1)
        return boundary

    @staticmethod
    def _in_session(
        now: datetime, params: LiquiditySweepReversalParams,
    ) -> bool:
        # session_end_hour_utc=24 means "until midnight"; treat as 24 hours.
        hour = now.hour
        return params.session_start_hour_utc <= hour < params.session_end_hour_utc

    def _on_cooldown(self, params: LiquiditySweepReversalParams) -> bool:
        if self._last_close_bar_index is None:
            return False
        return (self._bar_index - self._last_close_bar_index) < params.cooldown_bars

    # ------------------------------------------------------------------ #
    # Time exit                                                           #
    # ------------------------------------------------------------------ #

    async def _maybe_time_exit(self, ctx: StrategyContext) -> None:
        if not self._entry_bar_by_ticket:
            return
        params = ctx.params
        assert isinstance(params, LiquiditySweepReversalParams)
        # Iterate a snapshot — ctx.close() may trigger on_position_closed
        # in the same dispatch which would mutate the dict.
        for ticket, entry_index in list(self._entry_bar_by_ticket.items()):
            if (self._bar_index - entry_index) >= params.max_hold_bars:
                await ctx.close(ticket, reason="sweep_reversal_time_exit")
