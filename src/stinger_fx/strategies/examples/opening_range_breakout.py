"""OpeningRangeBreakout — session-opening directional breakout.

The third strategy in the LSR / VPC / ORB family. Where LSR fades
breakouts in *ranging* markets and VPC continues trends after a
pullback, ORB **takes the first breakout** of the session's opening
range.

Setup family
============

At a configured UTC session anchor (default 07:00 = London open), the
strategy:

  1. **Builds the opening range (OR)** by scanning M1 bars in the
     window ``[session_anchor, session_anchor + opening_range_minutes)``
     and freezes ``or_high`` / ``or_low`` once the window closes.
  2. **Waits for a breakout** on the M1 entry timeframe — close beyond
     ``or_high + buffer`` (long) or ``or_low − buffer`` (short).
  3. **Requires M5 confirmation** — the most recently closed M5 bar's
     close must also be beyond the same OR boundary.  This blocks wick
     spikes that immediately revert (the classic ORB false-breakout
     pattern).
  4. **Enters market** with SL on the opposite OR side (configurable)
     and TP at the measured-move target ``entry ± or_range`` (also
     configurable).

The strategy is **one trade per session by design** — re-entries after
a failed breakout are not supported in v1 (controlled by
``max_trades_per_session=1``).  Sessions roll over at the next UTC
session anchor and the OR is rebuilt.

Risk discipline
===============

* Max one position per ``symbol`` at a time.
* ``max_trades_per_session`` hard caps daily turnover (UTC date).
* ``max_entry_minutes_from_open`` blocks "late chase" trades — if no
  breakout fires within ~90 minutes of session open, the session is
  skipped.
* Range-size filters (``min_or_range_atr`` / ``max_or_range_atr``)
  block both dead markets and news-event spikes.
* Trades with computed reward-to-risk below ``min_rr`` are skipped.
* Positions force-closed via ``ctx.close()`` after ``max_hold_bars``
  (ORB exits normally via SL/TP; time exit is a safety net).
* Optional ``skip_friday_late`` guards against weekend-gap exposure.

Account-level guards (RiskMonitor per-strategy caps, daily-loss limits,
kill-switch) remain the engine's responsibility.

SL modes
========

* ``opposite_or`` (default): SL = ``or_low`` for long / ``or_high`` for
  short.  Classic ORB stop — "if price re-enters the range, the
  breakout has failed."
* ``or_mid``: SL = ``(or_high + or_low) / 2``.  Tighter; exits at the
  first sign of fade.
* ``atr``: SL = ``entry ∓ stop_buffer_atr × ATR``.  Volatility-scaled.

TP modes
========

* ``or_range`` (default): TP = ``entry ± or_range``.  Classic measured-
  move target — the breakout "extends" by the same height as the
  range.
* ``fixed_r``: TP = ``entry ± take_profit_r × risk``.
* ``trailing``: no fixed TP; a ``TrailingStopManager`` is attached at
  entry time.  SL still serves as initial protection.

v1 scope notes
==============

* ``avoid_news`` parameter is reserved as a future hook (no calendar
  service in the framework yet) — stored but never read.
* No multi-TF H1 bias filter — both directions are eligible on the
  first valid breakout.
* No volume / tick-activity filter (spec proposes
  ``tick_volume > X × mean``; v1 does not gate on it).
* No ATR-expansion filter (spec proposes
  ``ATR_now > ATR_pre_session``; v1 does not snapshot pre-session
  ATR).
* No multi-leg TP (TP1 + TP2 + runner) — single TP per trade; operator
  picks the mode.
* No pending stop entry (buy-stop above ``or_high + buffer``); v1 uses
  market orders after M5 confirmation to match LSR/VPC.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import Field

from stinger_fx.domain import Bar, Order, Position, Side, Subscription, Timeframe
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.indicators import atr
from stinger_fx.strategies.managers.trailing import TrailingStopManager
from stinger_fx.strategies.parameters import StrategyParams


class OpeningRangeBreakoutParams(StrategyParams):
    """Tunables.  Defaults are XAUUSD + London-open friendly starting
    points; every threshold is ATR-scaled so the same defaults
    transplant to any liquid symbol that has a defined session open.
    Expect to re-optimise per instrument and session."""

    # --- Feed identity ----------------------------------------------
    symbol: str = "XAUUSD"
    entry_timeframe: Timeframe = Timeframe.M1
    structure_timeframe: Timeframe = Timeframe.M5

    # --- Opening range definition -----------------------------------
    session_start_hour_utc: int = Field(7, ge=0, le=23)   # London open
    session_end_hour_utc: int = Field(16, ge=1, le=24)    # block late entries
    opening_range_minutes: int = Field(15, ge=5, le=60)
    # Max minutes after session_open during which an entry may still
    # fire — caps "late chase" trades (spec: ~90).
    max_entry_minutes_from_open: int = Field(90, ge=15, le=360)

    # --- ATR scaling -------------------------------------------------
    atr_period: int = Field(14, ge=5, le=50)
    breakout_buffer_atr: float = Field(0.1, ge=0)
    stop_buffer_atr: float = Field(0.3, gt=0)             # used when sl_mode == "atr"

    # --- Range-size filters -----------------------------------------
    min_or_range_atr: float = Field(0.5, gt=0)            # lower bound
    max_or_range_atr: float = Field(3.0, gt=0)            # upper bound (news guard)

    # --- SL mode -----------------------------------------------------
    sl_mode: Literal["opposite_or", "or_mid", "atr"] = "opposite_or"

    # --- TP mode -----------------------------------------------------
    tp_mode: Literal["or_range", "fixed_r", "trailing"] = "or_range"
    take_profit_r: float = Field(1.0, gt=0)               # for fixed_r / trailing
    min_rr: float = Field(0.8, gt=0)

    # --- Trailing runner (only when tp_mode == "trailing") ---------
    trailing_distance_atr: float = Field(0.8, gt=0)
    trailing_activate_atr: float = Field(0.5, ge=0)
    trailing_point: float = Field(0.01, gt=0)             # XAU point ≈ 0.01

    # --- Sizing / cadence -------------------------------------------
    volume: float = Field(0.01, gt=0)
    max_hold_bars: int = Field(60, ge=1)                  # ORB exits via TP/SL; this is safety
    cooldown_bars: int = Field(0, ge=0)                   # max_trades_per_session is the hard cap
    max_trades_per_session: int = Field(1, ge=1)          # spec default: 1

    # --- Optional Friday late guard ---------------------------------
    skip_friday_late: bool = False
    friday_late_hour_utc: int = Field(20, ge=12, le=23)

    # --- Future hooks (v1: no-op) ----------------------------------
    avoid_news: bool = False


class OpeningRangeBreakout(BaseStrategy):
    name = "opening_range_breakout"
    Params = OpeningRangeBreakoutParams

    @classmethod
    def subscriptions(cls, params: StrategyParams) -> list[Subscription]:
        assert isinstance(params, OpeningRangeBreakoutParams)
        return [
            Subscription(symbol=params.symbol, timeframe=params.entry_timeframe),
            Subscription(symbol=params.symbol, timeframe=params.structure_timeframe),
        ]

    @classmethod
    def warmup_bars(
        cls, params: StrategyParams,
    ) -> dict[Subscription, int] | None:
        """Per-feed warmup for live-mode startup backfill (Plan A2).

        ORB needs only a small window: the M1 entry-tf needs enough
        bars to build the opening range (15 by default) and the M5
        structure-tf needs ``atr_period + 1`` closed bars.  Engines
        started mid-session will still build the OR from the bars seen
        after engine start — backfill makes the FIRST session work end
        to end if the engine boots between session anchor and OR-close.

        We size M1 to cover the largest possible OR window + a buffer
        so the freeze logic finds bars in
        ``[anchor, anchor + opening_range_minutes)``.  M5 is just the
        ATR warmup.
        """
        assert isinstance(params, OpeningRangeBreakoutParams)
        # M1: at least one OR window worth, plus enough buffer to make
        # the freeze-after-window logic find data.
        m1_bars = max(params.opening_range_minutes * 2, 60)
        return {
            Subscription(symbol=params.symbol, timeframe=params.entry_timeframe):
                m1_bars,
            Subscription(symbol=params.symbol, timeframe=params.structure_timeframe):
                params.atr_period + 1,
        }

    def __init__(self) -> None:
        super().__init__()
        # Per-session state — reset by _maybe_roll_session.
        self._or_high: float | None = None
        self._or_low: float | None = None
        self._or_built_at: datetime | None = None
        self._trades_this_session: int = 0
        self._session_key: str | None = None
        self._last_close_bar_index: int | None = None
        self._entry_bar_by_ticket: dict[int, int] = {}
        self._bar_index: int = 0

    # ------------------------------------------------------------------ #
    # Bar dispatch                                                        #
    # ------------------------------------------------------------------ #

    async def on_bar(self, ctx: StrategyContext, bar: Bar) -> None:
        params = ctx.params
        assert isinstance(params, OpeningRangeBreakoutParams)

        # Multi-TF: ignore non-entry-tf bars; runner still appends them
        # to the corresponding HistoryView so we can read M5 below.
        if bar.symbol != params.symbol or bar.timeframe != params.entry_timeframe:
            return
        self._bar_index += 1

        # 1) Session rollover — resets OR + counters at session boundary.
        self._maybe_roll_session(bar.time, params)
        await self._maybe_time_exit(ctx)

        # 2) Skip outside session window.
        if not self._in_session(bar.time, params):
            return

        # 3) Friday late guard (optional).
        if params.skip_friday_late and self._is_friday_late(bar.time, params):
            return

        # 4) Build / freeze the opening range.
        self._maybe_build_or(ctx, params, bar.time)

        # 5) Need a frozen OR before evaluating entries.
        if self._or_high is None or self._or_low is None or self._or_built_at is None:
            return

        # 6) Session cap + cooldown + max-entry-window + open-position.
        if self._trades_this_session >= params.max_trades_per_session:
            return
        if self._on_cooldown(params):
            return
        if self._past_max_entry_minutes(bar.time, params):
            return
        if ctx.position.for_symbol(params.symbol):
            return

        # 7) Warmup for ATR on the structure tf.
        m5 = ctx.history_for(params.symbol, params.structure_timeframe)
        if m5 is None:
            return
        m5_bars = m5.bars()
        if len(m5_bars) < params.atr_period + 1:
            return
        atr_value = atr(m5_bars, params.atr_period)
        if atr_value is None:
            return

        # 8) Range-size filter.
        or_range = self._or_high - self._or_low
        if or_range < params.min_or_range_atr * atr_value:
            return
        if or_range > params.max_or_range_atr * atr_value:
            return

        # 9) Breakout + M5 confirmation.
        # HistoryView only appends closed bars, so [-1] is the most
        # recently closed M5 bar.  When an M1 fires on an M5 boundary
        # (e.g. 07:20:00), the just-closed M5 bar IS [-1]; when an M1
        # fires mid-M5-window, [-1] is the previous M5 — entry waits
        # until the next M5 close to confirm.  Desired semantics.
        buf = params.breakout_buffer_atr * atr_value
        last_closed_m5 = m5_bars[-1]

        if bar.close > self._or_high + buf and last_closed_m5.close > self._or_high:
            await self._enter_long(ctx, params, bar, atr_value, or_range)
        elif bar.close < self._or_low - buf and last_closed_m5.close < self._or_low:
            await self._enter_short(ctx, params, bar, atr_value, or_range)

    # ------------------------------------------------------------------ #
    # Opening range builder                                               #
    # ------------------------------------------------------------------ #

    def _maybe_build_or(
        self,
        ctx: StrategyContext,
        params: OpeningRangeBreakoutParams,
        now: datetime,
    ) -> None:
        """Build/freeze the opening range once per session.  After the
        session anchor (e.g. 07:00 UTC for London), scan M1 bars in
        the ``[session_anchor, session_anchor + opening_range_minutes)``
        window and freeze or_high/or_low.  The freeze happens on the
        first bar that arrives *after* the window closes — the OR
        itself never includes the in-progress entry bar."""
        if self._or_high is not None:
            return  # already frozen for this session
        anchor = self._session_start(now, params.session_start_hour_utc)
        or_end = anchor + timedelta(minutes=params.opening_range_minutes)
        if now < or_end:
            return  # still inside the OR-build window
        m1 = ctx.history_for(params.symbol, params.entry_timeframe)
        if m1 is None:
            return
        window_bars = [b for b in m1.bars() if anchor <= b.time < or_end]
        if not window_bars:
            return  # no data — skip session
        self._or_high = max(b.high for b in window_bars)
        self._or_low = min(b.low for b in window_bars)
        self._or_built_at = or_end
        ctx.logger.info(
            "or_frozen",
            or_high=self._or_high,
            or_low=self._or_low,
            or_range=self._or_high - self._or_low,
        )

    # ------------------------------------------------------------------ #
    # Entry helpers                                                       #
    # ------------------------------------------------------------------ #

    async def _enter_long(
        self,
        ctx: StrategyContext,
        params: OpeningRangeBreakoutParams,
        bar: Bar,
        atr_value: float,
        or_range: float,
    ) -> None:
        entry = bar.close
        sl = self._compute_sl(params, side=Side.BUY, entry=entry, atr_value=atr_value)
        tp = self._compute_tp(
            params, side=Side.BUY, entry=entry, sl=sl, or_range=or_range,
        )
        if not self._rr_ok(entry=entry, sl=sl, tp=tp, side=Side.BUY, params=params):
            return

        if params.tp_mode == "trailing":
            ctx.attach_manager(TrailingStopManager(
                ctx,
                distance_pips=params.trailing_distance_atr * atr_value / params.trailing_point,
                activate_after_pips=params.trailing_activate_atr * atr_value / params.trailing_point,
                symbol=params.symbol,
                point=params.trailing_point,
            ))
            tp_arg: float | None = None
        else:
            tp_arg = tp

        await ctx.buy(params.volume, sl=sl, tp=tp_arg, comment="orb_long")

    async def _enter_short(
        self,
        ctx: StrategyContext,
        params: OpeningRangeBreakoutParams,
        bar: Bar,
        atr_value: float,
        or_range: float,
    ) -> None:
        entry = bar.close
        sl = self._compute_sl(params, side=Side.SELL, entry=entry, atr_value=atr_value)
        tp = self._compute_tp(
            params, side=Side.SELL, entry=entry, sl=sl, or_range=or_range,
        )
        if not self._rr_ok(entry=entry, sl=sl, tp=tp, side=Side.SELL, params=params):
            return

        if params.tp_mode == "trailing":
            ctx.attach_manager(TrailingStopManager(
                ctx,
                distance_pips=params.trailing_distance_atr * atr_value / params.trailing_point,
                activate_after_pips=params.trailing_activate_atr * atr_value / params.trailing_point,
                symbol=params.symbol,
                point=params.trailing_point,
            ))
            tp_arg: float | None = None
        else:
            tp_arg = tp

        await ctx.sell(params.volume, sl=sl, tp=tp_arg, comment="orb_short")

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
    # SL / TP / RR                                                        #
    # ------------------------------------------------------------------ #

    def _compute_sl(
        self,
        params: OpeningRangeBreakoutParams,
        *,
        side: Side,
        entry: float,
        atr_value: float,
    ) -> float:
        assert self._or_high is not None
        assert self._or_low is not None
        if params.sl_mode == "opposite_or":
            return self._or_low if side is Side.BUY else self._or_high
        if params.sl_mode == "or_mid":
            return (self._or_high + self._or_low) / 2
        # sl_mode == "atr"
        buf = params.stop_buffer_atr * atr_value
        return entry - buf if side is Side.BUY else entry + buf

    def _compute_tp(
        self,
        params: OpeningRangeBreakoutParams,
        *,
        side: Side,
        entry: float,
        sl: float,
        or_range: float,
    ) -> float | None:
        if params.tp_mode == "or_range":
            return (entry + or_range) if side is Side.BUY else (entry - or_range)
        # fixed_r AND trailing both use the R-multiple TP.  For
        # trailing, the manager handles the actual exit; the TP value
        # is used only for the RR sanity check so we don't open
        # setups whose risk profile is hopeless.
        risk = abs(sl - entry)
        if risk <= 0:
            return None
        if side is Side.BUY:
            return entry + params.take_profit_r * risk
        return entry - params.take_profit_r * risk

    @staticmethod
    def _rr_ok(
        *,
        entry: float,
        sl: float,
        tp: float | None,
        side: Side,
        params: OpeningRangeBreakoutParams,
    ) -> bool:
        if tp is None:
            return False
        risk = abs(sl - entry)
        reward = (tp - entry) if side is Side.BUY else (entry - tp)
        if risk <= 0 or reward <= 0:
            return False
        return (reward / risk) >= params.min_rr

    # ------------------------------------------------------------------ #
    # Session / cooldown / time exit  (lifted from LSR/VPC + OR reset)   #
    # ------------------------------------------------------------------ #

    def _maybe_roll_session(
        self, now: datetime, params: OpeningRangeBreakoutParams,
    ) -> None:
        key = self._session_start(now, params.session_start_hour_utc).date().isoformat()
        if self._session_key != key:
            self._session_key = key
            self._trades_this_session = 0
            # ORB-specific: blank the frozen OR so the new session
            # rebuilds it from fresh M1 bars.
            self._or_high = None
            self._or_low = None
            self._or_built_at = None

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
        now: datetime, params: OpeningRangeBreakoutParams,
    ) -> bool:
        hour = now.hour
        return params.session_start_hour_utc <= hour < params.session_end_hour_utc

    def _on_cooldown(self, params: OpeningRangeBreakoutParams) -> bool:
        if self._last_close_bar_index is None:
            return False
        return (self._bar_index - self._last_close_bar_index) < params.cooldown_bars

    def _past_max_entry_minutes(
        self, now: datetime, params: OpeningRangeBreakoutParams,
    ) -> bool:
        if self._or_built_at is None:
            return False
        anchor = self._session_start(now, params.session_start_hour_utc)
        return now >= anchor + timedelta(minutes=params.max_entry_minutes_from_open)

    @staticmethod
    def _is_friday_late(
        now: datetime, params: OpeningRangeBreakoutParams,
    ) -> bool:
        return now.weekday() == 4 and now.hour >= params.friday_late_hour_utc

    async def _maybe_time_exit(self, ctx: StrategyContext) -> None:
        if not self._entry_bar_by_ticket:
            return
        params = ctx.params
        assert isinstance(params, OpeningRangeBreakoutParams)
        for ticket, entry_index in list(self._entry_bar_by_ticket.items()):
            if (self._bar_index - entry_index) >= params.max_hold_bars:
                await ctx.close(ticket, reason="orb_time_exit")
