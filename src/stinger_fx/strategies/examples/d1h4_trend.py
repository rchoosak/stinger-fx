"""D1H4TrendStrategy — medium-to-long-term trend following for XAUUSD.

The strategy consumes **H1** bars only and folds them, calendar-aware, into H4
and D1 series (see :mod:`stinger_fx.strategies.aggregation`). It then trades a
classic three-layer trend system:

  1. **D1 regime gate** — only take longs in an established D1 uptrend (and
     shorts in a downtrend, when ``allow_short``). The gate combines an EMA
     stack (``EMA(fast) > EMA(slow)``), a rising/falling slow EMA over a slope
     lookback, price on the right side of the slow EMA, and a minimum ADX. The
     long and short ADX floors are asymmetric (shorts demand a stronger trend).
  2. **H4 breakout entry** — inside a valid regime, enter on an H4 close beyond
     the Donchian channel of the *prior* ``breakout_lookback`` H4 bars (the
     breakout bar itself is excluded), with the H4 fast/slow EMAs aligned.
     False breakouts are filtered: reject if the breakout bar's gap-inclusive
     True Range, or its distance past the channel boundary, exceeds a multiple
     of the previous ATR.
  3. **Chandelier exit** — hold for days/weeks behind a ratcheting Chandelier
     stop (``highest_high(N) − ATR×k`` for longs), plus a D1 regime exit when
     price closes back through the D1 ``exit`` EMA.

Discipline / framework integration:

  * **Single evaluation per completed H4.** Entries and normal exits are decided
    once, when a new H4 bar closes — never re-derived on every H1 inside it.
  * **Market entry, no fixed lot.** The strategy passes a nominal volume plus an
    initial ATR stop; the engine's risk/position-sizing layer computes the real
    size from the stop distance. No pending entries (incompatible with
    risk-percent sizing).
  * **Exit retry.** If an exit is signalled but the broker rejects it or only
    partially closes, the same exit is retried on the next completed H4 — never
    re-fired every H1 (no overtrading). Remaining volume is whatever the broker
    actually still holds.
  * **One position per symbol, no stacking.** An opposite signal can only open
    after the existing position has closed (entries require a flat book).
  * **Durable state.** The ratcheting stop + position identity are persisted
    (when ``state_path`` is set) and restored on restart *only* when strategy,
    symbol, side, entry price and broker ticket all still match a live position;
    otherwise the stale state is cleared.

Live cold-start note: the live warmup backfill only warms the current forming
bar, so the D1 series fills from live H1 bars over time. Until ~``d1_slow_ema +
d1_slope_lookback`` D1 bars exist the regime gate stays neutral and the strategy
holds fire — safe by construction. Backtests include the warmup span in the
replayed data, so they warm within the run. Live and backtest fold the identical
H1 stream, so both evaluate the identical signals.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime

from pydantic import Field

from stinger_fx.domain import Bar, Order, Position, Side, Subscription, Timeframe
from stinger_fx.strategies.aggregation import (
    ForexWeekCalendar,
    MultiTimeframeAggregator,
    SessionCalendar,
)
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.indicators import adx, atr, donchian, ema
from stinger_fx.strategies.parameters import StrategyParams
from stinger_fx.strategies.state_store import (
    InMemoryStateStore,
    JsonFileStateStore,
    PositionState,
    StrategyStateStore,
    reconcile,
)

_H4_CAP = 512
_D1_CAP = 512


class D1H4TrendParams(StrategyParams):
    """Tunables. Defaults are the XAUUSD swing set from the spec."""

    symbol: str = "XAUUSD"
    timeframe: Timeframe = Timeframe.H1
    daily_anchor_hour: int = Field(0, ge=0, le=23)

    # --- Session calendar (configurable; defaults = standard 24x5 FX week) --
    week_open_hour: int = Field(22, ge=0, le=23)    # Sunday open (UTC)
    week_close_hour: int = Field(22, ge=1, le=24)   # Friday close (UTC)
    session_break_hours: tuple[int, ...] = ()
    """UTC hours the instrument is closed *every* trading day (scheduled break,
    not data loss) — e.g. gold's ~21:00 maintenance break => (21,)."""

    # --- D1 regime ---------------------------------------------------------
    d1_fast_ema: int = Field(30, ge=1)
    d1_slow_ema: int = Field(150, ge=2)
    d1_slope_lookback: int = Field(3, ge=1)
    d1_adx_length: int = Field(14, ge=2)
    d1_long_adx_min: float = Field(15.0, ge=0)
    d1_short_adx_min: float = Field(20.0, ge=0)
    d1_exit_ema: int = Field(50, ge=1)

    # --- H4 entry ----------------------------------------------------------
    h4_fast_ema: int = Field(10, ge=1)
    h4_slow_ema: int = Field(30, ge=2)
    breakout_lookback: int = Field(15, ge=1)
    atr_length: int = Field(14, ge=2)
    initial_stop_atr: float = Field(2.5, gt=0)
    max_breakout_atr: float = Field(2.5, gt=0)
    max_channel_breakout_atr: float = Field(2.5, gt=0)

    # --- Chandelier exit ---------------------------------------------------
    chandelier_lookback: int = Field(22, ge=1)
    chandelier_atr: float = Field(3.0, gt=0)

    # --- Sizing / direction ------------------------------------------------
    allow_short: bool = True
    volume: float = Field(0.01, gt=0)   # nominal; risk engine resizes from SL

    # --- Durable state (live) ---------------------------------------------
    state_path: str | None = None        # JSON file; None → in-memory (no persist)


class D1H4TrendStrategy(BaseStrategy):
    name = "d1h4_trend"
    Params = D1H4TrendParams

    @classmethod
    def subscriptions(cls, params: StrategyParams) -> list[Subscription]:
        assert isinstance(params, D1H4TrendParams)
        # H1 only — H4/D1 are folded internally (calendar-aware).
        return [Subscription(symbol=params.symbol, timeframe=Timeframe.H1)]

    @classmethod
    def min_h1_history(cls, params: D1H4TrendParams) -> int:
        """Closed H1 bars needed before the D1 regime gate can warm: enough D1
        bars for ``EMA(d1_slow_ema)`` + its slope lookback (+buffer), × 24."""
        d1_needed = params.d1_slow_ema + params.d1_slope_lookback + 5
        return d1_needed * 24

    @classmethod
    def warmup_bars(
        cls, params: StrategyParams
    ) -> dict[Subscription, int] | None:
        assert isinstance(params, D1H4TrendParams)
        return {
            Subscription(symbol=params.symbol, timeframe=Timeframe.H1):
                cls.min_h1_history(params),
        }

    def __init__(self) -> None:
        super().__init__()
        self._agg: MultiTimeframeAggregator | None = None
        # Test seam: inject a SessionCalendar (None → default ForexWeekCalendar).
        self._calendar: SessionCalendar | None = None
        self._h4: deque[Bar] = deque(maxlen=_H4_CAP)
        self._d1: deque[Bar] = deque(maxlen=_D1_CAP)
        # Active managed position (None when flat). chandelier_stop is the
        # ratcheting protective level.
        self._state: PositionState | None = None
        # Persisted state read at start, awaiting reconcile against live book.
        self._loaded: PositionState | None = None
        self._reconciled = False
        self._store: StrategyStateStore = InMemoryStateStore()
        # An exit was signalled but the position hasn't confirmed closed —
        # retry the same exit on the next completed H4.
        self._pending_exit = False
        # Initial stop of the order we just sent, adopted as the first
        # chandelier level when the fill arrives.
        self._entry_stop: float | None = None
        self._last_eval_h4: datetime | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def on_start(self, ctx: StrategyContext) -> None:
        params = ctx.params
        assert isinstance(params, D1H4TrendParams)
        calendar = self._calendar or ForexWeekCalendar(
            week_open_hour=params.week_open_hour,
            week_close_hour=params.week_close_hour,
            daily_break_hours=frozenset(params.session_break_hours),
        )
        self._agg = MultiTimeframeAggregator(
            params.symbol,
            anchor_hour=params.daily_anchor_hour,
            calendar=calendar,
        )
        if params.state_path:
            self._store = JsonFileStateStore(params.state_path)
        self._loaded = self._store.load(ctx.strategy_id)
        self._reconciled = False
        # Seed the folder from any history already in the view (live restart
        # may pre-fill it; backtest history is empty here and fills via on_bar).
        h1 = ctx.history_for(params.symbol, Timeframe.H1)
        if h1 is not None:
            for bar in h1.bars():
                self._ingest(bar)

    async def on_bar(self, ctx: StrategyContext, bar: Bar) -> None:
        params = ctx.params
        assert isinstance(params, D1H4TrendParams)
        if bar.symbol != params.symbol or bar.timeframe is not Timeframe.H1:
            return
        new_h4 = self._ingest(bar)
        if new_h4 is None:
            return  # no completed H4 this H1 → single-eval-per-H4 holds
        if self._last_eval_h4 is not None and new_h4.time <= self._last_eval_h4:
            return  # guard: never evaluate the same H4 twice
        self._last_eval_h4 = new_h4.time
        self._reconcile_startup(ctx, params)
        await self._evaluate(ctx, params, new_h4)

    def _ingest(self, bar: Bar) -> Bar | None:
        """Feed an H1 bar into the folder; append any completed H4/D1 to the
        deques. Returns the newly-completed H4 bar (or None)."""
        if self._agg is None:
            return None
        res = self._agg.feed(bar)
        if res.d1 is not None:
            self._d1.append(res.d1)
        if res.h4 is not None:
            self._h4.append(res.h4)
        return res.h4

    # ------------------------------------------------------------------ #
    # Per-H4 evaluation                                                  #
    # ------------------------------------------------------------------ #

    async def _evaluate(
        self, ctx: StrategyContext, params: D1H4TrendParams, h4: Bar
    ) -> None:
        position = self._active_position(ctx, params)
        if position is not None:
            await self._manage_open(ctx, params, h4, position)
        else:
            # Flat — drop any stale state, then look for a fresh entry.
            if self._state is not None or self._pending_exit:
                self._clear_state(ctx)
            await self._maybe_enter(ctx, params, h4)

    # ------------------------------------------------------------------ #
    # Entry                                                              #
    # ------------------------------------------------------------------ #

    async def _maybe_enter(
        self, ctx: StrategyContext, params: D1H4TrendParams, h4: Bar
    ) -> None:
        regime = self._d1_regime(params)
        if regime == 0:
            return
        # Need the prior breakout_lookback H4 bars (exclude the breakout bar)
        # plus ATR warmup.
        prior = list(self._h4)[:-1]
        need = max(params.breakout_lookback, params.atr_length + 1)
        if len(prior) < need:
            return
        dc = donchian(prior, params.breakout_lookback)
        atr_prev = atr(prior, params.atr_length)
        if dc is None or atr_prev is None or atr_prev <= 0:
            return
        h4_closes = [b.close for b in self._h4]
        h4_fast = ema(h4_closes, params.h4_fast_ema)
        h4_slow = ema(h4_closes, params.h4_slow_ema)
        if h4_fast is None or h4_slow is None:
            return

        prev_close = prior[-1].close
        true_range = max(
            h4.high - h4.low,
            abs(h4.high - prev_close),
            abs(h4.low - prev_close),
        )
        # Oversized / gap candle filter (covers both gap and big-range bars).
        if true_range > atr_prev * params.max_breakout_atr:
            return

        # Entry reference = the price we actually enter at now (the breakout H4
        # only confirms once the next bucket opens), which is also what the risk
        # engine stamps for sizing. Fall back to the H4 close if no live price.
        ref = ctx.history.last_price()
        entry = ref if ref is not None else h4.close

        if regime > 0:
            if not (h4_fast > h4_slow and h4.close > dc.upper):
                return
            if (h4.close - dc.upper) > atr_prev * params.max_channel_breakout_atr:
                return
            stop = entry - atr_prev * params.initial_stop_atr
            self._entry_stop = stop
            await ctx.buy(params.volume, sl=stop, comment="d1h4_long")
        else:
            if not (h4_fast < h4_slow and h4.close < dc.lower):
                return
            if (dc.lower - h4.close) > atr_prev * params.max_channel_breakout_atr:
                return
            stop = entry + atr_prev * params.initial_stop_atr
            self._entry_stop = stop
            await ctx.sell(params.volume, sl=stop, comment="d1h4_short")

    # ------------------------------------------------------------------ #
    # Open-position management: chandelier ratchet + exits + retry        #
    # ------------------------------------------------------------------ #

    async def _manage_open(
        self,
        ctx: StrategyContext,
        params: D1H4TrendParams,
        h4: Bar,
        position: Position,
    ) -> None:
        # Make sure our durable state tracks this position.
        if self._state is None or self._state.ticket != position.ticket:
            self._adopt_position(ctx, position)
        assert self._state is not None
        is_long = self._state.side == Side.BUY.value

        # 1) Ratchet the Chandelier stop (recomputed on each completed H4).
        new_stop = self._chandelier_stop(params, is_long=is_long)
        if new_stop is not None:
            prev = self._state.chandelier_stop
            ratcheted = max(prev, new_stop) if is_long else min(prev, new_stop)
            if ratcheted != prev:
                self._set_stop(ctx, ratcheted)
                await ctx.move_stop(
                    position.ticket, sl=ratcheted, reason="d1h4_chandelier"
                )

        stop = self._state.chandelier_stop

        # 2) Exit decision (regime exit OR chandelier breach), retried if a
        #    prior exit didn't complete (reject / partial close).
        exit_now = self._pending_exit
        reason = "d1h4_retry"
        if not exit_now:
            if self._regime_exit(params, is_long=is_long):
                exit_now, reason = True, "d1h4_regime_exit"
            elif (is_long and h4.close < stop) or (not is_long and h4.close > stop):
                exit_now, reason = True, "d1h4_chandelier_exit"
        if exit_now:
            self._pending_exit = True
            await ctx.close(position.ticket, reason=reason)

    def _chandelier_stop(
        self, params: D1H4TrendParams, *, is_long: bool
    ) -> float | None:
        bars = list(self._h4)
        if len(bars) < max(params.chandelier_lookback, params.atr_length + 1):
            return None
        atr_v = atr(bars, params.atr_length)
        if atr_v is None or atr_v <= 0:
            return None
        window = bars[-params.chandelier_lookback:]
        if is_long:
            return max(b.high for b in window) - atr_v * params.chandelier_atr
        return min(b.low for b in window) + atr_v * params.chandelier_atr

    def _regime_exit(self, params: D1H4TrendParams, *, is_long: bool) -> bool:
        d1_closes = [b.close for b in self._d1]
        exit_ema = ema(d1_closes, params.d1_exit_ema)
        if exit_ema is None:
            return False
        close = d1_closes[-1]
        return close < exit_ema if is_long else close > exit_ema

    # ------------------------------------------------------------------ #
    # D1 regime                                                          #
    # ------------------------------------------------------------------ #

    def _d1_regime(self, params: D1H4TrendParams) -> int:
        """+1 long regime, -1 short regime, 0 neutral."""
        closes = [b.close for b in self._d1]
        if len(closes) < params.d1_slow_ema + params.d1_slope_lookback + 1:
            return 0
        fast = ema(closes, params.d1_fast_ema)
        slow = ema(closes, params.d1_slow_ema)
        slow_prev = ema(closes[:-params.d1_slope_lookback], params.d1_slow_ema)
        adx_res = adx(list(self._d1), params.d1_adx_length)
        if fast is None or slow is None or slow_prev is None or adx_res is None:
            return 0
        close = closes[-1]
        if (
            close > slow
            and fast > slow
            and slow > slow_prev
            and adx_res.adx >= params.d1_long_adx_min
        ):
            return 1
        if (
            params.allow_short
            and close < slow
            and fast < slow
            and slow < slow_prev
            and adx_res.adx >= params.d1_short_adx_min
        ):
            return -1
        return 0

    # ------------------------------------------------------------------ #
    # State / position plumbing                                          #
    # ------------------------------------------------------------------ #

    def _active_position(
        self, ctx: StrategyContext, params: D1H4TrendParams
    ) -> Position | None:
        own = ctx.position.for_symbol(params.symbol)
        return own[0] if own else None

    def _reconcile_startup(
        self, ctx: StrategyContext, params: D1H4TrendParams
    ) -> None:
        if self._reconciled:
            return
        self._reconciled = True
        positions = ctx.position.for_symbol(params.symbol)
        restored = reconcile(
            self._loaded, positions,
            strategy_id=ctx.strategy_id, symbol=params.symbol,
        )
        if restored is not None:
            self._state = restored
            ctx.logger.info(
                "d1h4_state_restored", ticket=restored.ticket,
                side=restored.side, stop=restored.chandelier_stop,
            )
        else:
            # Stale or absent — never resurrect a stop for a phantom position.
            if self._loaded is not None:
                self._store.clear(ctx.strategy_id)
            self._state = None
            if positions:
                # We own a live position with no valid saved state — manage it
                # with a fresh stop (rebuilt on the next completed H4).
                self._adopt_position(ctx, positions[0])
        self._loaded = None

    def _adopt_position(self, ctx: StrategyContext, position: Position) -> None:
        stop = position.sl if position.sl is not None else position.open_price
        self._state = PositionState(
            strategy_id=ctx.strategy_id,
            symbol=position.symbol,
            side=position.side.value,
            entry_price=position.open_price,
            ticket=position.ticket,
            chandelier_stop=stop,
        )
        self._pending_exit = False
        self._store.save(self._state)

    def _set_stop(self, ctx: StrategyContext, stop: float) -> None:
        assert self._state is not None
        self._state = PositionState(
            strategy_id=self._state.strategy_id,
            symbol=self._state.symbol,
            side=self._state.side,
            entry_price=self._state.entry_price,
            ticket=self._state.ticket,
            chandelier_stop=stop,
        )
        self._store.save(self._state)

    def _clear_state(self, ctx: StrategyContext) -> None:
        self._state = None
        self._pending_exit = False
        self._entry_stop = None
        self._store.clear(ctx.strategy_id)

    # ------------------------------------------------------------------ #
    # Fill / close hooks                                                 #
    # ------------------------------------------------------------------ #

    async def on_order_filled(self, ctx: StrategyContext, order: Order) -> None:
        if order.strategy_id != ctx.strategy_id or not order.ticket:
            return
        stop = order.sl if order.sl is not None else self._entry_stop
        if stop is None:
            stop = order.fill_price if order.fill_price is not None else 0.0
        self._state = PositionState(
            strategy_id=ctx.strategy_id,
            symbol=order.symbol,
            side=order.side.value,
            entry_price=order.fill_price if order.fill_price is not None else 0.0,
            ticket=order.ticket,
            chandelier_stop=stop,
        )
        self._pending_exit = False
        self._entry_stop = None
        self._store.save(self._state)

    async def on_position_closed(
        self, ctx: StrategyContext, position: Position
    ) -> None:
        # Full close confirmed — clear durable state. A partial close keeps the
        # position open (same ticket), so this does not fire and the pending
        # exit is retried on the next H4 against the remaining volume.
        if self._state is not None and position.ticket == self._state.ticket:
            self._clear_state(ctx)
