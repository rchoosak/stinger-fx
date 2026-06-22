"""D1H4TrendStrategy — entry/exit rules, chandelier ratchet, retry, persistence.

The strategy folds H1 → H4/D1 internally, so tests feed a continuous hourly
series (an :class:`AlwaysOpenCalendar` is injected so fixtures don't have to
model weekends). Tiny indicator periods keep the warmup short.
"""

from __future__ import annotations

import asyncio
import itertools
from datetime import UTC, datetime, timedelta

import pytest
import structlog

from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.core.events import (
    ClosePositionRequestEvent,
    ModifyOrderRequestEvent,
)
from stinger_fx.domain import Bar, Order, OrderType, Position, Side, Signal, Timeframe
from stinger_fx.strategies.aggregation import AlwaysOpenCalendar
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.examples.d1h4_trend import (
    D1H4TrendParams,
    D1H4TrendStrategy,
)
from stinger_fx.strategies.state_store import InMemoryStateStore, PositionState

SYMBOL = "XAUUSD"
START = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)


def _params(**over) -> D1H4TrendParams:
    base = dict(
        symbol=SYMBOL, daily_anchor_hour=0,
        d1_fast_ema=3, d1_slow_ema=6, d1_slope_lookback=2, d1_adx_length=3,
        d1_long_adx_min=10.0, d1_short_adx_min=12.0, d1_exit_ema=4,
        h4_fast_ema=2, h4_slow_ema=3, breakout_lookback=3, atr_length=3,
        initial_stop_atr=2.5, max_breakout_atr=2.5, max_channel_breakout_atr=2.5,
        chandelier_lookback=3, chandelier_atr=3.0, allow_short=True, volume=0.1,
    )
    base.update(over)
    return D1H4TrendParams(**base)  # type: ignore[arg-type]


def _h1(t: datetime, o: float, h: float, lo: float, c: float) -> Bar:
    return Bar(symbol=SYMBOL, timeframe=Timeframe.H1, time=t,
               open=o, high=h, low=lo, close=c, tick_volume=10, is_closed=True)


def trend_series(n: int, *, start: datetime = START, price0: float = 1000.0,
                 slope: float = 1.0, wick: float | None = None) -> list[Bar]:
    """Continuous hourly bars rising (slope>0) / falling (slope<0) by ``slope``
    per hour. ``wick`` sets high/low padding; a large wick keeps the Donchian
    channel above the rising closes so the regime warms without a breakout."""
    bars: list[Bar] = []
    t, p = start, price0
    pad = wick if wick is not None else max(abs(slope) * 0.2, 0.05)
    for _ in range(n):
        o, c = p, p + slope
        bars.append(_h1(t, o, max(o, c) + pad, min(o, c) - pad, c))
        p = c
        t += timedelta(hours=1)
    return bars


def flat_series(n: int, *, start: datetime = START, price: float = 1000.0) -> list[Bar]:
    bars: list[Bar] = []
    t = start
    for i in range(n):
        c = price + (0.2 if i % 2 == 0 else -0.2)  # tiny chop, no trend
        bars.append(_h1(t, price, price + 0.4, price - 0.4, c))
        t += timedelta(hours=1)
    return bars


class Harness:
    def __init__(self, params: D1H4TrendParams,
                 positions: list[Position] | None = None,
                 store: InMemoryStateStore | None = None) -> None:
        self.bus = AsyncEventBus()
        self.signals: list[Signal] = []
        self.closes: list[ClosePositionRequestEvent] = []
        self.modifies: list[ModifyOrderRequestEvent] = []

        async def sink(sig: Signal) -> None:
            self.signals.append(sig)

        async def on_close(evt: ClosePositionRequestEvent) -> None:
            self.closes.append(evt)

        async def on_modify(evt: ModifyOrderRequestEvent) -> None:
            self.modifies.append(evt)

        self.bus.subscribe(ClosePositionRequestEvent, on_close, name="t.close")
        self.bus.subscribe(ModifyOrderRequestEvent, on_modify, name="t.modify")

        self.ctx = StrategyContext(
            strategy_id="d1h4_test", symbol=params.symbol,
            timeframe=Timeframe.H1, params=params,
            clock=SimClock(START), logger=structlog.get_logger("d1h4_test"),
            magic=77, signal_sink=sink,
            subscriptions=D1H4TrendStrategy.subscriptions(params), bus=self.bus,
        )
        if positions:
            self.ctx.position.update(
                [p.model_copy(update={"magic": 77}) for p in positions]
            )
        self.strat = D1H4TrendStrategy()
        self.strat._calendar = AlwaysOpenCalendar()
        if store is not None:
            self.strat._store = store

    async def start(self) -> None:
        await self.strat.on_start(self.ctx)

    async def feed(self, bars: list[Bar]) -> None:
        for b in bars:
            # Mirror the runner: append to the feed's HistoryView before
            # dispatching on_bar, so ctx.history.last_price() is populated.
            view = self.ctx.history_for(SYMBOL, Timeframe.H1)
            assert view is not None
            view.append_bar(b)
            await self.strat.on_bar(self.ctx, b)
        await self.drain()

    def set_positions(self, positions: list[Position]) -> None:
        self.ctx.position.update(
            [p.model_copy(update={"magic": 77}) for p in positions]
        )

    async def drain(self) -> None:
        # move_stop / close go through the queue-based bus; let the consumer
        # tasks process everything that was enqueued during the feed.
        for _ in range(2000):
            pending = sum(s.queue.qsize() for s in self.bus._subs if not s.closed)
            if pending == 0:
                break
            await asyncio.sleep(0)
        await asyncio.sleep(0)


def _pos(side: Side, *, open_price: float, ticket: int = 1000,
         sl: float | None = None) -> Position:
    return Position(ticket=ticket, symbol=SYMBOL, side=side, volume=0.1,
                    open_price=open_price, open_time=datetime.now(UTC), sl=sl)


# ----------------------------------------------------------------------- #
# 1–4: regime gating                                                       #
# ----------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_long_regime_breakout_enters() -> None:
    h = Harness(_params())
    await h.start()
    await h.feed(trend_series(24 * 16, slope=1.0))  # strong, warm uptrend
    buys = [s for s in h.signals if s.side is Side.BUY]
    assert buys, "expected at least one long entry in a strong uptrend"
    first = buys[0]
    assert first.order_type is OrderType.MARKET
    assert first.suggested_sl is not None
    assert first.suggested_sl < (first.entry_ref_price or 0)


@pytest.mark.asyncio
async def test_short_regime_breakout_enters() -> None:
    h = Harness(_params())
    await h.start()
    await h.feed(trend_series(24 * 16, price0=2000.0, slope=-1.0))
    sells = [s for s in h.signals if s.side is Side.SELL]
    assert sells, "expected at least one short entry in a strong downtrend"
    assert sells[0].suggested_sl is not None
    assert sells[0].suggested_sl > (sells[0].entry_ref_price or 0)


@pytest.mark.asyncio
async def test_neutral_regime_no_entry() -> None:
    h = Harness(_params())
    await h.start()
    await h.feed(flat_series(24 * 16))
    assert h.signals == []


@pytest.mark.asyncio
async def test_allow_short_false_blocks_shorts() -> None:
    h = Harness(_params(allow_short=False))
    await h.start()
    await h.feed(trend_series(24 * 16, price0=2000.0, slope=-1.0))
    assert [s for s in h.signals if s.side is Side.SELL] == []


# ----------------------------------------------------------------------- #
# 6–8: false-breakout filters (a tightened threshold must block the entry  #
#       that the default threshold takes — proving the filter gates entry) #
# ----------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_true_range_filter_rejects_breakout() -> None:
    series = trend_series(24 * 16, slope=1.0)
    # Control: default threshold enters.
    ctrl = Harness(_params())
    await ctrl.start()
    await ctrl.feed(series)
    assert [s for s in ctrl.signals if s.side is Side.BUY]
    # Tiny max_breakout_atr → every breakout bar's True Range exceeds it → no entry.
    strict = Harness(_params(max_breakout_atr=0.001))
    await strict.start()
    await strict.feed(series)
    assert [s for s in strict.signals if s.side is Side.BUY] == []


@pytest.mark.asyncio
async def test_channel_distance_filter_rejects_breakout() -> None:
    series = trend_series(24 * 16, slope=1.0)
    # Lax TR but tiny channel distance → the close sits too far past the
    # Donchian boundary → rejected.
    strict = Harness(_params(max_breakout_atr=100.0, max_channel_breakout_atr=0.001))
    await strict.start()
    await strict.feed(series)
    assert [s for s in strict.signals if s.side is Side.BUY] == []
    # Same series, lax channel → enters (isolates the channel filter).
    lax = Harness(_params(max_breakout_atr=100.0, max_channel_breakout_atr=100.0))
    await lax.start()
    await lax.feed(series)
    assert [s for s in lax.signals if s.side is Side.BUY]


@pytest.mark.asyncio
async def test_oversized_candle_rejected() -> None:
    # Strong trend (regime long) but a big wick holds the Donchian channel above
    # the rising closes, so no breakout fires during warmup. Then one crafted H4:
    # an oversized-range candle is rejected; the same breakout at normal range
    # enters — at the *default* threshold.
    base = trend_series(24 * 16, slope=1.0, wick=8.0)  # len % 4 == 0 → clean H4
    prev_close = base[-1].close
    nxt = base[-1].time + timedelta(hours=1)

    def breakout(*, extra_range: float) -> list[Bar]:
        top = prev_close + 40.0  # closes ~32 above the channel (passes channel)
        steps = [prev_close + 10, prev_close + 20, prev_close + 30, top]
        out, o, t = [], prev_close, nxt
        for c in steps:
            out.append(_h1(t, o, max(o, c) + extra_range, min(o, c) - 0.5, c))
            o, t = c, t + timedelta(hours=1)
        out.append(_h1(t, top, top + 0.5, top - 0.5, top))  # next bucket → eval
        return out

    big = Harness(_params())
    await big.start()
    await big.feed(base)
    assert big.signals == []  # warmup did not enter
    await big.feed(breakout(extra_range=60.0))   # range ≫ prior ATR
    assert big.signals == []  # oversized candle rejected

    small = Harness(_params())
    await small.start()
    await small.feed(base)
    assert small.signals == []
    await small.feed(breakout(extra_range=1.0))   # normal range
    assert [s for s in small.signals if s.side is Side.BUY]  # enters


# ----------------------------------------------------------------------- #
# 9: initial stop = entry_ref ∓ ATR × initial_stop_atr                     #
# ----------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_initial_stop_scales_with_atr_multiple() -> None:
    series = trend_series(24 * 16, slope=1.0)

    async def first_buy_gap(mult: float) -> float:
        h = Harness(_params(initial_stop_atr=mult))
        await h.start()
        await h.feed(series)
        buy = next(s for s in h.signals if s.side is Side.BUY)
        assert buy.entry_ref_price is not None and buy.suggested_sl is not None
        return buy.entry_ref_price - buy.suggested_sl  # long → positive

    g25 = await first_buy_gap(2.5)
    g50 = await first_buy_gap(5.0)
    assert g25 > 0
    assert g50 == pytest.approx(2.0 * g25, rel=1e-6)  # distance ∝ initial_stop_atr


@pytest.mark.asyncio
async def test_initial_stop_short_above_entry() -> None:
    h = Harness(_params())
    await h.start()
    await h.feed(trend_series(24 * 16, price0=2000.0, slope=-1.0))
    sell = next(s for s in h.signals if s.side is Side.SELL)
    assert sell.suggested_sl is not None and sell.entry_ref_price is not None
    assert sell.suggested_sl > sell.entry_ref_price


# ----------------------------------------------------------------------- #
# Helpers for open-position scenarios                                      #
# ----------------------------------------------------------------------- #

def _next(bars: list[Bar]) -> datetime:
    return bars[-1].time + timedelta(hours=1)


async def _warm_long_open(params: D1H4TrendParams, *, entry: float = 1000.0,
                          ticket: int = 1000, sl: float = 990.0):
    """A harness holding an open long, warmed by a long uptrend."""
    h = Harness(params, positions=[_pos(Side.BUY, open_price=entry, ticket=ticket, sl=sl)])
    await h.start()
    up = trend_series(24 * 10, slope=1.0, price0=entry)
    await h.feed(up)
    return h, up


# ----------------------------------------------------------------------- #
# 10: D1 EMA50 regime exit                                                 #
# ----------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_d1_ema50_regime_exit() -> None:
    # Chandelier kept very loose so the regime exit is the trigger.
    params = _params(chandelier_atr=1000.0)
    h, up = await _warm_long_open(params)
    assert h.closes == []  # holds through the uptrend
    down = trend_series(24 * 8, start=_next(up), price0=up[-1].close, slope=-2.0)
    await h.feed(down)
    assert h.closes, "expected a D1 regime exit after price fell through EMA50"
    assert any("regime_exit" in c.reason for c in h.closes)


# ----------------------------------------------------------------------- #
# 11: Chandelier ratchet never loosens                                     #
# ----------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_chandelier_ratchets_monotonically() -> None:
    h, _ = await _warm_long_open(_params())
    sls = [m.sl for m in h.modifies if m.sl is not None]
    assert len(sls) >= 2, "expected the chandelier to tighten as price rose"
    assert all(b >= a for a, b in itertools.pairwise(sls)), "long stop must not drop"
    assert sls[-1] > sls[0], "stop should ratchet up in an uptrend"


# ----------------------------------------------------------------------- #
# 12: exit retried after a reject / partial close                          #
# ----------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_exit_retried_when_not_confirmed() -> None:
    params = _params(chandelier_atr=1000.0)
    h, up = await _warm_long_open(params)
    down = trend_series(24 * 8, start=_next(up), price0=up[-1].close, slope=-2.0)
    await h.feed(down)
    n = len(h.closes)
    assert n >= 1
    # Position is never removed from the book (broker rejected / partial) →
    # the same exit must be retried on subsequent completed H4s.
    more = trend_series(24 * 2, start=_next(down), price0=down[-1].close, slope=-2.0)
    await h.feed(more)
    assert len(h.closes) > n
    assert {c.ticket for c in h.closes} == {1000}  # always the same position


# ----------------------------------------------------------------------- #
# 13: one evaluation per completed H4 — no H1 overtrading                   #
# ----------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_single_eval_per_completed_h4() -> None:
    params = _params(chandelier_atr=1000.0)
    h, up = await _warm_long_open(params)
    down = trend_series(24 * 8, start=_next(up), price0=up[-1].close, slope=-2.0)
    await h.feed(down)
    assert h.closes  # exit pending now
    p = down[-1].close
    t = _next(down)
    assert int((t - START) / timedelta(hours=1)) % 4 == 0  # bucket boundary
    # The boundary bar completes the prior H4 → one (retry) eval.
    await h.feed([_h1(t, p, p + 0.5, p - 0.5, p)])
    n = len(h.closes)
    # 3 bars strictly inside [t, t+4h) → no completion → no further eval.
    inside = [_h1(t + i * timedelta(hours=1), p, p + 0.5, p - 0.5, p) for i in (1, 2, 3)]
    await h.feed(inside)
    assert len(h.closes) == n, "no eval should happen inside an open H4"
    # The next boundary completes that H4 → exactly one more eval.
    await h.feed([_h1(t + 4 * timedelta(hours=1), p, p + 0.5, p - 0.5, p)])
    assert len(h.closes) == n + 1


# ----------------------------------------------------------------------- #
# 14 / 15: persist + restore vs reject-on-mismatch                          #
# ----------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_restore_chandelier_on_matching_position() -> None:
    store = InMemoryStateStore()
    store.save(PositionState("d1h4_test", SYMBOL, Side.BUY.value,
                             entry_price=1000.0, ticket=1000, chandelier_stop=1234.0))
    h = Harness(_params(), positions=[_pos(Side.BUY, open_price=1000.0, ticket=1000)],
                store=store)
    await h.start()
    # One completed H4 triggers the startup reconcile (too few H4s to ratchet yet).
    await h.feed(trend_series(5, slope=1.0, price0=1000.0))
    assert h.strat._state is not None
    assert h.strat._state.ticket == 1000
    assert h.strat._state.chandelier_stop == 1234.0  # restored, not recomputed


@pytest.mark.asyncio
async def test_no_restore_when_ticket_changed() -> None:
    store = InMemoryStateStore()
    store.save(PositionState("d1h4_test", SYMBOL, Side.BUY.value,
                             entry_price=1000.0, ticket=1000, chandelier_stop=1234.0))
    # Live book has a DIFFERENT ticket — the saved stop must not be resurrected.
    h = Harness(_params(), positions=[_pos(Side.BUY, open_price=1000.0, ticket=999,
                                           sl=995.0)], store=store)
    await h.start()
    await h.feed(trend_series(5, slope=1.0, price0=1000.0))
    assert h.strat._state is not None
    assert h.strat._state.ticket == 999
    assert h.strat._state.chandelier_stop != 1234.0  # stale stop cleared


@pytest.mark.asyncio
async def test_no_restore_when_no_live_position() -> None:
    store = InMemoryStateStore()
    store.save(PositionState("d1h4_test", SYMBOL, Side.BUY.value,
                             entry_price=1000.0, ticket=1000, chandelier_stop=1234.0))
    h = Harness(_params(), store=store)  # flat book
    await h.start()
    await h.feed(trend_series(5, slope=1.0, price0=1000.0))
    assert h.strat._state is None
    assert store.load("d1h4_test") is None  # cleared


# ----------------------------------------------------------------------- #
# 21: no lookahead — the breakout H4 is not acted on until it completes     #
# ----------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_no_lookahead_until_h4_completes() -> None:
    base = trend_series(24 * 16, slope=1.0, wick=8.0)  # regime long, no breakout
    prev_close = base[-1].close
    t = base[-1].time + timedelta(hours=1)
    top = prev_close + 40.0
    steps = [prev_close + 10, prev_close + 20, prev_close + 30, top]
    bucket = []
    o = prev_close
    for c in steps:
        bucket.append(_h1(t, o, max(o, c) + 1.0, min(o, c) - 0.5, c))
        o, t = c, t + timedelta(hours=1)

    h = Harness(_params())
    await h.start()
    await h.feed(base)
    assert h.signals == []
    # All 4 H1 bars of the breakout H4 are in — but the H4 is not yet *complete*.
    await h.feed(bucket)
    assert h.signals == [], "must not trade on an in-progress H4 (no lookahead)"
    # First bar of the next bucket completes the breakout H4 → entry now.
    await h.feed([_h1(t, top, top + 0.5, top - 0.5, top)])
    assert [s for s in h.signals if s.side is Side.BUY]


# ----------------------------------------------------------------------- #
# 22: end-to-end lifecycle — enter → ratchet → exit → flat                  #
# ----------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_end_to_end_lifecycle() -> None:
    h = Harness(_params(chandelier_atr=1000.0))
    await h.start()
    up = trend_series(24 * 14, slope=1.0, price0=1000.0)
    await h.feed(up)
    buys = [s for s in h.signals if s.side is Side.BUY]
    assert buys, "strategy should enter long in the uptrend"
    # Simulate the broker fill so ctx.position + strategy state reflect it.
    entry = buys[0].entry_ref_price or up[-1].close
    order = Order(ticket=4242, strategy_id="d1h4_test", symbol=SYMBOL,
                  side=Side.BUY, type=OrderType.MARKET, volume=0.1,
                  fill_price=entry, sl=buys[0].suggested_sl, status=__import__(
                      "stinger_fx.domain", fromlist=["OrderStatus"]).OrderStatus.FILLED)
    h.set_positions([_pos(Side.BUY, open_price=entry, ticket=4242,
                          sl=buys[0].suggested_sl)])
    await h.strat.on_order_filled(h.ctx, order)
    assert h.strat._state is not None and h.strat._state.ticket == 4242
    # Reverse → exit.
    down = trend_series(24 * 8, start=_next(up), price0=up[-1].close, slope=-2.0)
    await h.feed(down)
    assert h.closes, "strategy should exit when the trend reverses"
    # Simulate the close confirmation → state cleared, ready to trade again.
    await h.strat.on_position_closed(
        h.ctx, _pos(Side.BUY, open_price=entry, ticket=4242))
    assert h.strat._state is None


# ----------------------------------------------------------------------- #
# 5: the Donchian channel excludes the current (breakout) bar              #
# ----------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_donchian_excludes_current_breakout_bar() -> None:
    # The breakout bar has a HUGE high. If the channel wrongly included the
    # current bar, its own high would lift the upper band above the close and
    # block the entry. Excluding it (correct) lets the breakout stand. TR /
    # channel filters are slackened so only the exclusion decides the outcome.
    base = trend_series(24 * 16, slope=1.0, wick=8.0)
    prev_close = base[-1].close
    t = base[-1].time + timedelta(hours=1)
    top = prev_close + 40.0
    steps = [prev_close + 10, prev_close + 20, prev_close + 30, top]
    bucket, o = [], prev_close
    for i, c in enumerate(steps):
        hi = (c + 200.0) if i == 3 else max(o, c) + 1.0  # last H1: enormous high
        bucket.append(_h1(t, o, hi, min(o, c) - 0.5, c))
        o, t = c, t + timedelta(hours=1)
    bucket.append(_h1(t, top, top + 0.5, top - 0.5, top))  # complete the H4

    h = Harness(_params(max_breakout_atr=1000.0, max_channel_breakout_atr=1000.0))
    await h.start()
    await h.feed(base)
    assert h.signals == []
    await h.feed(bucket)
    assert [s for s in h.signals if s.side is Side.BUY], (
        "breakout must stand — the current bar's high is excluded from the channel"
    )
