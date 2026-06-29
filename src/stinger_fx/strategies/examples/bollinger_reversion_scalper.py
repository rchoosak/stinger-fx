"""BollingerReversionScalper — fast intraday mean-reversion scalper.

Fade overstretched moves back to the mean on a short timeframe (M5 by default):
when price closes beyond a Bollinger band *and* RSI confirms exhaustion, enter
*against* the move expecting reversion to the middle band — but only while the
market is ranging (ADX below a ceiling), because fading a strong trend is the
fast way to blow up a mean-reversion book.

In / out fast, many small trades, tight risk:

  * **Entry (long)** — ``close <= lower band`` and ``RSI <= oversold`` and
    ``ADX <= max_adx``.  Short is the mirror.  Each direction is independently
    toggleable.
  * **Exit** — three ways, whichever comes first:
      1. **TP** at the middle band (the mean) — the broker closes it.
      2. **SL** at ``entry -/+ ATR x sl_atr_mult`` — the broker closes it.
      3. **Time-stop** after ``max_hold_bars`` — the strategy force-closes, so a
         stalled fade never becomes a bag-hold.

Safety rails (the point of the design):
  * ATR-based stop on every trade → bounded per-trade loss; the risk engine
    sizes each order to a fixed % of equity from that stop (strategy passes a
    nominal volume + SL, never a fixed lot).
  * ADX ceiling → don't fade strong trends.
  * RSI exhaustion → don't blindly catch a falling knife.
  * ``max_trades_per_session`` cap + ``cooldown_bars`` after each exit → bound
    overtrading and revenge entries.
  * Session-hours gate → trade only liquid hours (thin books = bad fills).
  * One position per symbol at a time (no stacking / no opposite stack).
  * Account-level daily-loss limit + kill switch remain the RiskMonitor's job.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta

from pydantic import Field, model_validator

from stinger_fx.domain import Bar, Order, Position, Side, Subscription, Timeframe
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.indicators import atr, bollinger, ema, rsi
from stinger_fx.strategies.parameters import StrategyParams


class BollingerReversionScalperParams(StrategyParams):
    """Tunables. Defaults are XAUUSD M5 starting points — re-optimise per
    instrument; every threshold is ATR- or indicator-relative so they transplant
    reasonably to other liquid symbols."""

    symbol: str = "XAUUSD"
    entry_timeframe: Timeframe = Timeframe.M5

    # --- Bollinger band (mean + stretch envelope) -------------------------
    bb_period: int = Field(20, ge=2)
    bb_std: float = Field(2.0, gt=0)

    # --- RSI exhaustion confirm -------------------------------------------
    rsi_period: int = Field(14, ge=2)
    rsi_oversold: float = Field(35.0, ge=0, le=100)
    rsi_overbought: float = Field(65.0, ge=0, le=100)

    # --- Stop / target -----------------------------------------------------
    atr_period: int = Field(14, ge=2)
    sl_atr_mult: float = Field(1.5, gt=0)
    min_target_atr: float = Field(0.3, ge=0)
    """Skip the trade if the middle-band target is closer than this x ATR
    (reward too small to be worth the spread/commission)."""

    # --- Trend filter — never fade a strong trend -------------------------
    adx_period: int = Field(14, ge=2)
    max_adx: float = Field(35.0, gt=0)

    # --- Higher-timeframe trend alignment ---------------------------------
    # When set, only fade *with* the higher-TF trend: in an uptrend (HTF close
    # above its EMA) take long fades only (buy the dip); in a downtrend take
    # short fades only (sell the rally). This is the regime fix for mean
    # reversion — fading *against* a trend is the fast way to bleed. None = off
    # (both directions eligible, gated only by ADX).
    trend_filter_timeframe: Timeframe | None = None
    trend_ema_period: int = Field(50, ge=2)

    # --- Fast exit + cadence guards ---------------------------------------
    max_hold_bars: int = Field(12, ge=1)        # M5 x 12 = 1h hard time-stop
    cooldown_bars: int = Field(2, ge=0)
    max_trades_per_session: int = Field(10, ge=1)
    session_start_hour_utc: int = Field(7, ge=0, le=23)
    session_end_hour_utc: int = Field(20, ge=1, le=24)

    # --- Sizing / direction ------------------------------------------------
    volume: float = Field(0.01, gt=0)           # nominal; risk engine resizes
    allow_long: bool = True
    allow_short: bool = True

    @model_validator(mode="after")
    def _check(self) -> BollingerReversionScalperParams:
        if self.session_end_hour_utc <= self.session_start_hour_utc:
            raise ValueError("session_end_hour_utc must be > session_start_hour_utc")
        if self.rsi_overbought <= self.rsi_oversold:
            raise ValueError("rsi_overbought must be > rsi_oversold")
        # Timeframe preconditions the bar/fold code relies on (TICK has no
        # fixed duration; the trend fold needs a strictly-larger TF that the
        # entry TF divides evenly). Reject at load time rather than crashing
        # in warmup_bars() / _fold_trend() or folding degenerate buckets.
        if self.entry_timeframe is Timeframe.TICK:
            raise ValueError("entry_timeframe must be a bar timeframe, not TICK")
        tf = self.trend_filter_timeframe
        if tf is not None:
            if tf is Timeframe.TICK:
                raise ValueError(
                    "trend_filter_timeframe must be a bar timeframe, not TICK"
                )
            if tf in (Timeframe.W1, Timeframe.MN1):
                # _fold_trend buckets by epoch-floor (int(ts) // tf.seconds),
                # which is only canonical for UTC-epoch-aligned timeframes. W1
                # would anchor on Thursday (the epoch weekday), not the Monday
                # the rest of the engine uses, and MN1.seconds is a 30-day
                # approximation that drifts off calendar months. Use an
                # intraday/daily TF (M*, H*, D1).
                raise ValueError(
                    "trend_filter_timeframe does not support W1/MN1 (the internal "
                    "fold buckets by epoch); use an intraday or daily timeframe"
                )
            if tf.seconds <= self.entry_timeframe.seconds:
                raise ValueError(
                    "trend_filter_timeframe must be higher than entry_timeframe "
                    f"(got {tf.value} <= {self.entry_timeframe.value})"
                )
            if tf.seconds % self.entry_timeframe.seconds != 0:
                raise ValueError(
                    "trend_filter_timeframe must be an integer multiple of "
                    f"entry_timeframe (got {tf.value} / {self.entry_timeframe.value})"
                )
        return self


class BollingerReversionScalper(BaseStrategy):
    name = "bollinger_reversion_scalper"
    Params = BollingerReversionScalperParams

    @classmethod
    def subscriptions(cls, params: StrategyParams) -> list[Subscription]:
        assert isinstance(params, BollingerReversionScalperParams)
        # Entry feed only. The higher-TF trend is folded internally from this
        # stream (emit-on-next-bucket) so it's lookahead-free and identical in
        # live and backtest — a separate higher-TF feed would be delivered at
        # its *open* time by the bar-mode backtester (i.e. one bucket early =
        # lookahead), unlike the live aggregator which emits it at its close.
        return [Subscription(symbol=params.symbol, timeframe=params.entry_timeframe)]

    @classmethod
    def warmup_bars(
        cls, params: StrategyParams
    ) -> dict[Subscription, int] | None:
        assert isinstance(params, BollingerReversionScalperParams)
        need = max(
            params.bb_period + 1,
            params.rsi_period + 1,
            params.atr_period + 1,
            2 * params.adx_period,
        )
        tf = params.trend_filter_timeframe
        if tf is not None:
            # Enough entry bars to fold `trend_ema_period`+ higher-TF buckets.
            per_bucket = max(1, tf.seconds // params.entry_timeframe.seconds)
            need = max(need, (params.trend_ema_period + 2) * per_bucket)
        return {
            Subscription(symbol=params.symbol, timeframe=params.entry_timeframe):
                need + 5,
        }

    def __init__(self) -> None:
        super().__init__()
        self._bar_index = 0
        self._session_key: str | None = None
        self._trades_this_session = 0
        self._last_close_index: int | None = None
        self._entry_bar_by_ticket: dict[int, int] = {}
        # Internal higher-TF fold for the trend filter: completed-bucket closes
        # + the in-progress bucket. The in-progress bucket is never used, so the
        # trend direction can't peek at the current (incomplete) higher-TF bar.
        self._trend_closes: deque[float] = deque(maxlen=4096)
        self._trend_key: int | None = None
        self._trend_close: float = 0.0

    # ------------------------------------------------------------------ #
    # Bar dispatch                                                        #
    # ------------------------------------------------------------------ #

    async def on_bar(self, ctx: StrategyContext, bar: Bar) -> None:
        params = ctx.params
        assert isinstance(params, BollingerReversionScalperParams)
        if bar.symbol != params.symbol or bar.timeframe is not params.entry_timeframe:
            return
        self._bar_index += 1
        self._fold_trend(bar, params)

        self._maybe_roll_session(bar.time, params)
        await self._maybe_time_exit(ctx, params)

        if not self._in_session(bar.time, params):
            return
        if ctx.position.for_symbol(params.symbol):
            return  # one position at a time — no stacking
        if self._trades_this_session >= params.max_trades_per_session:
            return
        if self._on_cooldown(params):
            return

        bars = ctx.history_for(params.symbol, params.entry_timeframe)
        if bars is None:
            return
        window = bars.bars()
        need = max(
            params.bb_period + 1, params.rsi_period + 1,
            params.atr_period + 1, 2 * params.adx_period,
        )
        if len(window) < need:
            return
        closes = [b.close for b in window]

        bb = bollinger(closes, params.bb_period, params.bb_std)
        rsi_v = rsi(closes, params.rsi_period)
        atr_v = atr(window, params.atr_period)
        # Streaming ADX off the HistoryView — O(1) per bar (the view keeps Wilder
        # state current) instead of the O(window) double-smoothed recompute.
        adx_res = bars.adx(params.adx_period)
        if bb is None or rsi_v is None or atr_v is None or adx_res is None:
            return
        if adx_res.adx > params.max_adx:
            return  # trending too hard — don't fade

        # Higher-TF trend alignment: only fade *with* the trend (long in an
        # uptrend, short in a downtrend). None of these constrain when the
        # filter is off; when on but not warm yet, hold fire.
        allow_long = params.allow_long
        allow_short = params.allow_short
        if params.trend_filter_timeframe is not None:
            tdir = self._trend_dir(params)
            if tdir is None:
                return
            allow_long = allow_long and tdir > 0
            allow_short = allow_short and tdir < 0

        if allow_long and bar.close <= bb.lower and rsi_v <= params.rsi_oversold:
            await self._enter(ctx, params, Side.BUY, bar.close, bb.middle, atr_v)
        elif (
            allow_short
            and bar.close >= bb.upper
            and rsi_v >= params.rsi_overbought
        ):
            await self._enter(ctx, params, Side.SELL, bar.close, bb.middle, atr_v)

    # ------------------------------------------------------------------ #
    # Entry                                                              #
    # ------------------------------------------------------------------ #

    async def _enter(
        self,
        ctx: StrategyContext,
        params: BollingerReversionScalperParams,
        side: Side,
        entry: float,
        middle: float,
        atr_v: float,
    ) -> None:
        if side is Side.BUY:
            sl = entry - atr_v * params.sl_atr_mult
            tp = middle
            if tp - entry < params.min_target_atr * atr_v:
                return  # reward too small
            await ctx.buy(params.volume, sl=sl, tp=tp, comment="bbr_long")
        else:
            sl = entry + atr_v * params.sl_atr_mult
            tp = middle
            if entry - tp < params.min_target_atr * atr_v:
                return
            await ctx.sell(params.volume, sl=sl, tp=tp, comment="bbr_short")

    def _fold_trend(
        self, bar: Bar, params: BollingerReversionScalperParams
    ) -> None:
        """Fold the entry-TF stream into higher-TF buckets, recording each
        bucket's *final* close only when the next bucket opens (emit-on-next-
        bucket). The in-progress bucket is held separately and never used, so
        the trend filter can't see an incomplete higher-TF bar (no lookahead;
        identical in live and backtest)."""
        tf = params.trend_filter_timeframe
        if tf is None:
            return
        key = int(bar.time.timestamp()) // tf.seconds
        if self._trend_key is None:
            self._trend_key = key
        elif key != self._trend_key:
            self._trend_closes.append(self._trend_close)  # prior bucket closed
            self._trend_key = key
        self._trend_close = bar.close

    def _trend_dir(self, params: BollingerReversionScalperParams) -> int | None:
        """+1 when the higher-TF trend is up (last completed bucket close above
        its EMA), -1 when down, None until enough completed buckets exist."""
        if len(self._trend_closes) < params.trend_ema_period + 1:
            return None
        closes = list(self._trend_closes)
        ema_v = ema(closes, params.trend_ema_period)
        if ema_v is None:
            return None
        return 1 if closes[-1] > ema_v else -1

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def on_order_filled(self, ctx: StrategyContext, order: Order) -> None:
        if order.strategy_id != ctx.strategy_id or not order.ticket:
            return
        # Idempotent: a duplicate / replayed fill for a ticket we already track
        # must not be counted twice. The ticket map doubles as the "already
        # counted" set.
        if order.ticket in self._entry_bar_by_ticket:
            return
        self._entry_bar_by_ticket[order.ticket] = self._bar_index
        # Bill the *entry* to the session it opened in (max_trades_per_session
        # caps entries). Counting on close would mis-bill a trade that opens
        # late in one session and closes in the next (e.g. across a weekend).
        self._trades_this_session += 1

    async def on_position_closed(
        self, ctx: StrategyContext, position: Position
    ) -> None:
        # Idempotent: only act for a ticket we actually track. A duplicate
        # PositionClosedEvent (the router and broker can both emit one) then
        # no-ops instead of re-resetting the cooldown.
        if self._entry_bar_by_ticket.pop(position.ticket, None) is None:
            return
        self._last_close_index = self._bar_index

    # ------------------------------------------------------------------ #
    # Time exit + session / cooldown helpers                              #
    # ------------------------------------------------------------------ #

    async def _maybe_time_exit(
        self, ctx: StrategyContext, params: BollingerReversionScalperParams
    ) -> None:
        if not self._entry_bar_by_ticket:
            return
        for ticket, entry_index in list(self._entry_bar_by_ticket.items()):
            if (self._bar_index - entry_index) >= params.max_hold_bars:
                await ctx.close(ticket, reason="bbr_time_exit")

    def _maybe_roll_session(
        self, now: datetime, params: BollingerReversionScalperParams
    ) -> None:
        key = self._session_start(now, params.session_start_hour_utc).date().isoformat()
        if self._session_key != key:
            self._session_key = key
            self._trades_this_session = 0

    @staticmethod
    def _session_start(now: datetime, session_start_hour_utc: int) -> datetime:
        boundary = now.replace(
            hour=session_start_hour_utc, minute=0, second=0, microsecond=0
        )
        if now < boundary:
            boundary -= timedelta(days=1)
        return boundary

    @staticmethod
    def _in_session(
        now: datetime, params: BollingerReversionScalperParams
    ) -> bool:
        return params.session_start_hour_utc <= now.hour < params.session_end_hour_utc

    def _on_cooldown(self, params: BollingerReversionScalperParams) -> bool:
        if self._last_close_index is None:
            return False
        return (self._bar_index - self._last_close_index) < params.cooldown_bars
