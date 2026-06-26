"""BollingerReversionScalper — entry/exit rules, filters, risk-safety guards."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
import structlog

from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.core.events import ClosePositionRequestEvent
from stinger_fx.domain import (
    Bar,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
    Signal,
    Timeframe,
)
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.examples.bollinger_reversion_scalper import (
    BollingerReversionScalper,
    BollingerReversionScalperParams,
)

SYMBOL = "XAUUSD"
TF = Timeframe.M5
# A weekday inside the default 07:00–20:00 session.
START = datetime(2024, 1, 8, 8, 0, tzinfo=UTC)


def _params(**over) -> BollingerReversionScalperParams:
    base = dict(
        symbol=SYMBOL, entry_timeframe=TF,
        bb_period=20, bb_std=2.0, rsi_period=14, rsi_oversold=40.0, rsi_overbought=60.0,
        atr_period=14, sl_atr_mult=1.5, min_target_atr=0.0,
        adx_period=14, max_adx=60.0, max_hold_bars=12, cooldown_bars=0,
        max_trades_per_session=10, session_start_hour_utc=7, session_end_hour_utc=20,
        volume=0.1, allow_long=True, allow_short=True,
    )
    base.update(over)
    return BollingerReversionScalperParams(**base)  # type: ignore[arg-type]


def _bar(t: datetime, o: float, h: float, lo: float, c: float) -> Bar:
    return Bar(symbol=SYMBOL, timeframe=TF, time=t, open=o, high=h, low=lo, close=c,
               tick_volume=100, is_closed=True)


def ranging(n: int, *, start: datetime = START, mid: float = 100.0,
            amp: float = 0.5) -> list[Bar]:
    """Choppy oscillation around ``mid`` — builds the bands while keeping ADX
    low (no directional movement)."""
    bars, t = [], start
    for i in range(n):
        c = mid + (amp if i % 2 == 0 else -amp)
        o = mid - (amp if i % 2 == 0 else -amp)
        bars.append(_bar(t, o, max(o, c) + 0.1, min(o, c) - 0.1, c))
        t += timedelta(minutes=5)
    return bars


def dip(start: datetime, *, from_px: float, steps: int, step: float) -> list[Bar]:
    """A short directional push (down if step<0) to stretch beyond a band and
    drive RSI to an extreme."""
    bars, t, p = [], start, from_px
    for _ in range(steps):
        o, c = p, p + step
        bars.append(_bar(t, o, max(o, c) + 0.1, min(o, c) - 0.1, c))
        p = c
        t += timedelta(minutes=5)
    return bars


class Harness:
    def __init__(self, params: BollingerReversionScalperParams,
                 positions: list[Position] | None = None) -> None:
        self.bus = AsyncEventBus()
        self.signals: list[Signal] = []
        self.closes: list[ClosePositionRequestEvent] = []

        async def sink(sig: Signal) -> None:
            self.signals.append(sig)

        async def on_close(evt: ClosePositionRequestEvent) -> None:
            self.closes.append(evt)

        self.bus.subscribe(ClosePositionRequestEvent, on_close, name="t.close")
        self.ctx = StrategyContext(
            strategy_id="bbr_test", symbol=SYMBOL, timeframe=TF, params=params,
            clock=SimClock(START), logger=structlog.get_logger("bbr_test"),
            magic=55, signal_sink=sink,
            subscriptions=BollingerReversionScalper.subscriptions(params), bus=self.bus,
        )
        if positions:
            self.ctx.position.update([p.model_copy(update={"magic": 55}) for p in positions])
        self.strat = BollingerReversionScalper()

    async def feed(self, bars: list[Bar]) -> None:
        for b in bars:
            view = self.ctx.history_for(SYMBOL, TF)
            assert view is not None
            view.append_bar(b)
            await self.strat.on_bar(self.ctx, b)
        for _ in range(200):
            if sum(s.queue.qsize() for s in self.bus._subs if not s.closed) == 0:
                break
            await asyncio.sleep(0)
        await asyncio.sleep(0)

    def set_positions(self, positions: list[Position]) -> None:
        self.ctx.position.update([p.model_copy(update={"magic": 55}) for p in positions])


def _pos(side: Side, *, open_price: float, ticket: int = 1) -> Position:
    return Position(ticket=ticket, symbol=SYMBOL, side=side, volume=0.1,
                    open_price=open_price, open_time=datetime.now(UTC))


def _long_setup() -> list[Bar]:
    """Ranging then a sharp dip → close below lower band, RSI oversold, ADX low."""
    base = ranging(26, mid=100.0, amp=0.5)
    d = dip(base[-1].time + timedelta(minutes=5), from_px=base[-1].close,
            steps=5, step=-1.0)
    return base + d


def _short_setup() -> list[Bar]:
    base = ranging(26, mid=100.0, amp=0.5)
    d = dip(base[-1].time + timedelta(minutes=5), from_px=base[-1].close,
            steps=5, step=+1.0)
    return base + d


# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_long_entry_fades_oversold_to_mean() -> None:
    h = Harness(_params())
    await h.feed(_long_setup())
    buys = [s for s in h.signals if s.side is Side.BUY]
    assert buys, "expected a long fade when oversold below the lower band"
    s = buys[0]
    assert s.order_type is OrderType.MARKET
    assert s.suggested_sl is not None and s.suggested_tp is not None
    assert s.suggested_sl < s.suggested_tp           # long: stop below target
    assert s.suggested_sl < (s.entry_ref_price or 0)  # stop below entry
    assert s.suggested_tp > (s.entry_ref_price or 0)  # target (mean) above entry


@pytest.mark.asyncio
async def test_short_entry_fades_overbought_to_mean() -> None:
    h = Harness(_params())
    await h.feed(_short_setup())
    sells = [s for s in h.signals if s.side is Side.SELL]
    assert sells, "expected a short fade when overbought above the upper band"
    s = sells[0]
    assert s.suggested_sl is not None and s.suggested_tp is not None
    assert s.suggested_sl > (s.entry_ref_price or 0)
    assert s.suggested_tp < (s.entry_ref_price or 0)


@pytest.mark.asyncio
async def test_adx_filter_blocks_when_trending() -> None:
    # Same dip, but a tiny max_adx → the (mild) directional push trips the
    # trend filter and the fade is skipped.
    h = Harness(_params(max_adx=1.0))
    await h.feed(_long_setup())
    assert [s for s in h.signals if s.side is Side.BUY] == []


@pytest.mark.asyncio
async def test_rsi_filter_blocks_without_exhaustion() -> None:
    # Require an extreme RSI the mild dip can't reach → no entry.
    h = Harness(_params(rsi_oversold=1.0))
    await h.feed(_long_setup())
    assert [s for s in h.signals if s.side is Side.BUY] == []


@pytest.mark.asyncio
async def test_allow_long_false_blocks_longs() -> None:
    h = Harness(_params(allow_long=False))
    await h.feed(_long_setup())
    assert [s for s in h.signals if s.side is Side.BUY] == []


@pytest.mark.asyncio
async def test_no_stacking_when_position_open() -> None:
    h = Harness(_params(), positions=[_pos(Side.BUY, open_price=100.0)])
    await h.feed(_long_setup())
    assert h.signals == []  # already holding → no new entry


@pytest.mark.asyncio
async def test_outside_session_no_entry() -> None:
    # Shift the series to 03:00 UTC (before the 07:00 session open).
    night = datetime(2024, 1, 8, 3, 0, tzinfo=UTC)
    base = ranging(26, start=night, mid=100.0, amp=0.5)
    d = dip(base[-1].time + timedelta(minutes=5), from_px=base[-1].close, steps=4, step=-0.8)
    h = Harness(_params())
    await h.feed(base + d)
    assert h.signals == []


@pytest.mark.asyncio
async def test_min_target_atr_skips_thin_reward() -> None:
    # Demand a target far from entry → the middle-band fade is too thin → skip.
    h = Harness(_params(min_target_atr=100.0))
    await h.feed(_long_setup())
    assert [s for s in h.signals if s.side is Side.BUY] == []


@pytest.mark.asyncio
async def test_time_exit_force_closes_stale_position() -> None:
    h = Harness(_params(max_hold_bars=12))
    bars = _long_setup()
    await h.feed(bars)
    buys = [s for s in h.signals if s.side is Side.BUY]
    assert buys
    # Simulate the fill, then feed > max_hold_bars more ranging bars.
    order = Order(ticket=321, strategy_id="bbr_test", symbol=SYMBOL, side=Side.BUY,
                  type=OrderType.MARKET, volume=0.1, fill_price=bars[-1].close,
                  status=__import__("stinger_fx.domain", fromlist=["OrderStatus"]).OrderStatus.FILLED)
    h.set_positions([_pos(Side.BUY, open_price=bars[-1].close, ticket=321)])
    await h.strat.on_order_filled(h.ctx, order)
    more = ranging(15, start=bars[-1].time + timedelta(minutes=5),
                   mid=bars[-1].close, amp=0.3)
    await h.feed(more)
    assert any(c.ticket == 321 and "time_exit" in c.reason for c in h.closes)


# --------------------------------------------------------------------------- #
# Cadence cap + config validation                                             #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_max_trades_per_session_caps_entries() -> None:
    # One trade already taken this session; cap=1 → no further entries even on a
    # fresh signal.
    h = Harness(_params(max_trades_per_session=1))
    h.strat._trades_this_session = 1
    h.strat._session_key = BollingerReversionScalper._session_start(
        START, 7).date().isoformat()
    await h.feed(_long_setup())
    assert h.signals == []


def test_subscriptions_single_entry_feed() -> None:
    from stinger_fx.domain import Subscription
    subs = BollingerReversionScalper.subscriptions(_params())
    assert subs == [Subscription(symbol=SYMBOL, timeframe=TF)]


def test_config_defaults_and_validation() -> None:
    from pydantic import ValidationError

    from stinger_fx.core.errors import StrategyError
    from stinger_fx.strategies.registry import validate_params

    p = BollingerReversionScalperParams()
    assert p.symbol == "XAUUSD" and p.entry_timeframe is Timeframe.M5

    # cross-field rules
    with pytest.raises(ValidationError):
        BollingerReversionScalperParams(session_start_hour_utc=20, session_end_hour_utc=10)
    with pytest.raises(ValidationError):
        BollingerReversionScalperParams(rsi_oversold=70.0, rsi_overbought=30.0)
    # field constraints
    for bad in (dict(bb_period=1), dict(sl_atr_mult=0.0), dict(max_adx=0.0),
                dict(volume=0.0), dict(max_hold_bars=0)):
        with pytest.raises(ValidationError):
            BollingerReversionScalperParams(**bad)  # type: ignore[arg-type]
    # frozen
    with pytest.raises(ValidationError):
        p.bb_period = 30  # type: ignore[misc]
    # via the registry loader
    assert isinstance(validate_params(BollingerReversionScalper, {"symbol": "XAUUSD"}),
                      BollingerReversionScalperParams)
    with pytest.raises(StrategyError):
        validate_params(BollingerReversionScalper, {"sl_atr_mult": -1.0})


# --------------------------------------------------------------------------- #
# Higher-TF trend filter — folded internally (lookahead-free), fade with trend #
# --------------------------------------------------------------------------- #

def _trend_params(**over):
    base = dict(trend_filter_timeframe=Timeframe.H1, trend_ema_period=50)
    base.update(over)
    return _params(**base)


def _set_trend(h: Harness, direction: int, *, n: int = 60, base: float = 100.0) -> None:
    """Seed the internal completed-bucket closes with a clear trend ending near
    ``base``. (The setup's dip/spike lands in the *in-progress* bucket, so it
    doesn't pollute these completed closes.)"""
    closes = [base - direction * 0.3 * (n - 1 - i) for i in range(n)]
    h.strat._trend_closes.extend(closes)
    h.strat._trend_key = int(START.timestamp()) // Timeframe.H1.seconds
    h.strat._trend_close = closes[-1]


def _rising_m5(n: int, *, start: datetime = START, base: float = 100.0,
               slope: float = 0.05) -> list[Bar]:
    bars, t, p = [], start, base
    for _ in range(n):
        c = p + slope
        bars.append(_bar(t, p, max(p, c) + 0.05, min(p, c) - 0.05, c))
        p, t = c, t + timedelta(minutes=5)
    return bars


@pytest.mark.asyncio
async def test_trend_up_allows_long_fades() -> None:
    h = Harness(_trend_params())
    _set_trend(h, +1)
    await h.feed(_long_setup())              # buy the oversold dip
    assert [s for s in h.signals if s.side is Side.BUY]


@pytest.mark.asyncio
async def test_trend_up_blocks_short_fades() -> None:
    h = Harness(_trend_params())
    _set_trend(h, +1)
    await h.feed(_short_setup())             # don't fade an uptrend's rally
    assert [s for s in h.signals if s.side is Side.SELL] == []


@pytest.mark.asyncio
async def test_trend_down_blocks_long_fades() -> None:
    h = Harness(_trend_params())
    _set_trend(h, -1)
    await h.feed(_long_setup())
    assert [s for s in h.signals if s.side is Side.BUY] == []


@pytest.mark.asyncio
async def test_trend_down_allows_short_fades() -> None:
    h = Harness(_trend_params())
    _set_trend(h, -1)
    await h.feed(_short_setup())
    assert [s for s in h.signals if s.side is Side.SELL]


@pytest.mark.asyncio
async def test_trend_filter_not_warm_blocks_all() -> None:
    h = Harness(_trend_params())
    _set_trend(h, +1, n=10)                  # < trend_ema_period+1 → not warm
    await h.feed(_long_setup())
    assert h.signals == []


@pytest.mark.asyncio
async def test_internal_fold_builds_higher_tf_trend() -> None:
    # Feed only the entry stream; the strategy folds it into H1 buckets and the
    # trend direction reflects the folded closes — no separate feed, no lookahead.
    h = Harness(_trend_params(trend_ema_period=5))
    await h.feed(_rising_m5(150, slope=0.05))       # ~12 H1 buckets, rising
    assert len(h.strat._trend_closes) >= 6
    assert h.strat._trend_dir(h.ctx.params) == 1    # up
    h2 = Harness(_trend_params(trend_ema_period=5))
    await h2.feed(_rising_m5(150, base=200.0, slope=-0.05))  # falling
    assert h2.strat._trend_dir(h2.ctx.params) == -1


def test_trend_filter_keeps_single_feed() -> None:
    from stinger_fx.domain import Subscription
    # Folded internally → still ONE subscription even with the filter on.
    assert BollingerReversionScalper.subscriptions(_trend_params()) == [
        Subscription(symbol=SYMBOL, timeframe=TF)
    ]
    assert BollingerReversionScalper.subscriptions(_params()) == [
        Subscription(symbol=SYMBOL, timeframe=TF)
    ]


def test_rejects_invalid_timeframe_configs() -> None:
    from pydantic import ValidationError
    # entry/trend = TICK (no fixed duration) → reject
    with pytest.raises(ValidationError):
        BollingerReversionScalperParams(entry_timeframe=Timeframe.TICK)
    with pytest.raises(ValidationError):
        BollingerReversionScalperParams(trend_filter_timeframe=Timeframe.TICK)
    # trend TF not strictly larger than entry TF → reject
    with pytest.raises(ValidationError):
        BollingerReversionScalperParams(
            entry_timeframe=Timeframe.M5, trend_filter_timeframe=Timeframe.M5)
    with pytest.raises(ValidationError):
        BollingerReversionScalperParams(
            entry_timeframe=Timeframe.H1, trend_filter_timeframe=Timeframe.M5)
    # trend TF not an integer multiple of entry TF (H1 / M45 = 1.33) → reject
    with pytest.raises(ValidationError):
        BollingerReversionScalperParams(
            entry_timeframe=Timeframe.M45, trend_filter_timeframe=Timeframe.H1)
    # W1 / MN1 → reject: epoch-floor fold mis-anchors them (W1 = Thursday,
    # MN1 = 30-day approximation), not the calendar boundary.
    with pytest.raises(ValidationError):
        BollingerReversionScalperParams(trend_filter_timeframe=Timeframe.W1)
    with pytest.raises(ValidationError):
        BollingerReversionScalperParams(trend_filter_timeframe=Timeframe.MN1)
    # valid: off, or an epoch-aligned higher TF that divides evenly (incl. D1)
    BollingerReversionScalperParams(trend_filter_timeframe=None)
    BollingerReversionScalperParams(
        entry_timeframe=Timeframe.M5, trend_filter_timeframe=Timeframe.M15)
    BollingerReversionScalperParams(
        entry_timeframe=Timeframe.M5, trend_filter_timeframe=Timeframe.D1)
    BollingerReversionScalperParams(
        entry_timeframe=Timeframe.M5, trend_filter_timeframe=Timeframe.H1)


# --------------------------------------------------------------------------- #
# Session cap is billed to the ENTRY's session (count on fill, not on close)   #
# --------------------------------------------------------------------------- #

def _filled(ticket: int, side: Side = Side.BUY, px: float = 100.0) -> Order:
    return Order(ticket=ticket, strategy_id="bbr_test", symbol=SYMBOL, side=side,
                 type=OrderType.MARKET, volume=0.1, fill_price=px,
                 status=OrderStatus.FILLED)


@pytest.mark.asyncio
async def test_session_cap_counts_entry_not_close() -> None:
    h = Harness(_params())
    await h.strat.on_order_filled(h.ctx, _filled(1))
    assert h.strat._trades_this_session == 1            # billed on the fill
    await h.strat.on_position_closed(h.ctx, _pos(Side.BUY, open_price=100.0, ticket=1))
    assert h.strat._trades_this_session == 1            # NOT re-billed on close


@pytest.mark.asyncio
async def test_session_cap_not_billed_across_session_roll() -> None:
    params = _params()
    h = Harness(params)
    await h.strat.on_order_filled(h.ctx, _filled(7))    # entry in session A
    assert h.strat._trades_this_session == 1
    # The next session-day rolls in (e.g. after a weekend) → quota resets.
    h.strat._maybe_roll_session(START + timedelta(days=3), params)
    assert h.strat._trades_this_session == 0
    # The session-A trade only now closes (in session B) → must NOT eat B's quota.
    await h.strat.on_position_closed(h.ctx, _pos(Side.BUY, open_price=100.0, ticket=7))
    assert h.strat._trades_this_session == 0


@pytest.mark.asyncio
async def test_duplicate_fill_and_close_events_are_idempotent() -> None:
    h = Harness(_params())
    # Duplicate / replayed fill for the same ticket → counted once.
    await h.strat.on_order_filled(h.ctx, _filled(1))
    await h.strat.on_order_filled(h.ctx, _filled(1))
    assert h.strat._trades_this_session == 1
    assert h.strat._entry_bar_by_ticket.get(1) is not None
    # Duplicate close (router + broker can both emit) → no error, no re-bill.
    pos = _pos(Side.BUY, open_price=100.0, ticket=1)
    await h.strat.on_position_closed(h.ctx, pos)
    await h.strat.on_position_closed(h.ctx, pos)
    assert h.strat._trades_this_session == 1
    assert 1 not in h.strat._entry_bar_by_ticket
