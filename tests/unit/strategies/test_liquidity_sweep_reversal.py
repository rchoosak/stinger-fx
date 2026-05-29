"""Unit tests for LiquiditySweepReversal.

The strategy is exercised directly: build a ``StrategyContext`` with
three HistoryViews (M1 / M5 / M15), pre-fill them with hand-crafted
bars, then call ``strat.on_bar(ctx, bar)`` and inspect the signals
captured through ``signal_sink``.

Test fixtures fall into two shapes:

  * **Ranging** — synthetic M15 bars with `high == low + tiny` and
    near-zero close changes so ADX stays below the ``max_adx`` gate.
  * **Trending** — strong monotonic uptrend in M15 closes so ADX climbs
    above ``max_adx`` (the entry must be blocked).

M5 ranges are seeded by hand. M1 entry bars are passed to ``on_bar``
directly with explicit high/low/close geometry to drive each branch.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
import structlog

from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.domain import (
    Bar,
    OrderType,
    Position,
    Side,
    Signal,
    Timeframe,
)
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.examples.liquidity_sweep_reversal import (
    LiquiditySweepReversal,
    LiquiditySweepReversalParams,
)

SYMBOL = "XAUUSD"


def _ts(minutes_from_base: int) -> datetime:
    # Anchor at UTC midnight so trigger timestamps land inside the
    # default session window [0, 16) of LiquiditySweepReversalParams.
    return datetime(2024, 1, 1, 0, 0, tzinfo=UTC) + timedelta(minutes=minutes_from_base)


def _bar(
    *, tf: Timeframe, t: datetime, o: float, h: float, lo: float, c: float,
    vol: int = 1000,
) -> Bar:
    return Bar(
        symbol=SYMBOL, timeframe=tf, time=t,
        open=o, high=h, low=lo, close=c,
        tick_volume=vol, is_closed=True,
    )


def _build_ranging_m15(n: int = 40, mid: float = 2000.0) -> list[Bar]:
    """Tiny-range M15 bars so ADX hovers below 22. Close alternates
    around mid; high/low straddle by a fraction of a dollar."""
    bars: list[Bar] = []
    for i in range(n):
        c = mid + (0.05 if i % 2 == 0 else -0.05)
        bars.append(_bar(
            tf=Timeframe.M15, t=_ts(15 * i),
            o=mid, h=mid + 0.1, lo=mid - 0.1, c=c,
        ))
    return bars


def _build_trending_m15(n: int = 40, start: float = 2000.0) -> list[Bar]:
    """Monotonic upward M15 closes → ADX climbs above 25 → regime gate
    must block any entry attempt."""
    bars: list[Bar] = []
    price = start
    for i in range(n):
        bars.append(_bar(
            tf=Timeframe.M15, t=_ts(15 * i),
            o=price, h=price + 1.0, lo=price - 0.2, c=price + 0.9,
        ))
        price += 1.0
    return bars


def _build_range_m5(
    n: int = 40, *, range_high: float = 2010.0, range_low: float = 1990.0,
) -> list[Bar]:
    """Sideway M5 bars that fill the Donchian channel between
    ``range_low`` and ``range_high`` cleanly. ATR settles around 4 (the
    full range / 5)."""
    bars: list[Bar] = []
    mid = (range_high + range_low) / 2
    half_amp = (range_high - range_low) / 2
    for i in range(n):
        # Walk inside the band; touch both extremes occasionally so
        # Donchian returns the full upper/lower.
        if i == 5:
            h, lo, c = range_high, mid, mid + 1
        elif i == 9:
            h, lo, c = mid, range_low, mid - 1
        else:
            h = mid + half_amp * 0.4
            lo = mid - half_amp * 0.4
            c = mid + (0.1 if i % 2 == 0 else -0.1)
        bars.append(_bar(tf=Timeframe.M5, t=_ts(5 * i), o=mid, h=h, lo=lo, c=c))
    return bars


def _build_ctx(
    *, params: LiquiditySweepReversalParams | None = None,
    m1_bars: list[Bar] | None = None,
    m5_bars: list[Bar] | None = None,
    m15_bars: list[Bar] | None = None,
    positions: list[Position] | None = None,
) -> tuple[StrategyContext, list[Signal]]:
    """Build a multi-feed StrategyContext with primary=M1, and capture
    every Signal emitted via signal_sink into the returned list."""
    p = params or LiquiditySweepReversalParams()
    bus = AsyncEventBus()
    captured: list[Signal] = []

    async def sink(sig: Signal) -> None:
        captured.append(sig)

    subs = LiquiditySweepReversal.subscriptions(p)
    ctx = StrategyContext(
        strategy_id="xau_test",
        symbol=p.symbol,
        timeframe=p.entry_timeframe,
        params=p,
        clock=SimClock(_ts(0)),
        logger=structlog.get_logger("xau_test"),
        magic=99,
        signal_sink=sink,
        subscriptions=subs,
        bus=bus,
    )
    # Seed history views.
    for b in m1_bars or []:
        view = ctx.history_for(p.symbol, p.entry_timeframe)
        assert view is not None
        view.append_bar(b)
    for b in m5_bars or []:
        view = ctx.history_for(p.symbol, p.structure_timeframe)
        assert view is not None
        view.append_bar(b)
    for b in m15_bars or []:
        view = ctx.history_for(p.symbol, p.regime_timeframe)
        assert view is not None
        view.append_bar(b)
    if positions:
        # PositionView filters by magic; re-tag so the test fixture's
        # positions are visible to this strategy's ctx.position.
        ctx.position.update([
            pos.model_copy(update={"magic": ctx.magic}) for pos in positions
        ])
    return ctx, captured


# ----------------------------------------------------------------------
# 1. Warmup gate
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_signal_when_warmup_insufficient() -> None:
    """Fewer than 2*adx_period M15 bars → ADX is None → return early."""
    strat = LiquiditySweepReversal()
    params = LiquiditySweepReversalParams(adx_period=14)
    # Only 10 M15 bars (need 28)
    m15 = _build_ranging_m15(n=10)
    m5 = _build_range_m5(n=40)
    # Proven short-sweep geometry — would fire if not for the warmup
    # gate. Keep it valid for OHLC.
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(5 * 40),
        o=2010.5, h=2012.0, lo=2007.0, c=2008.0,
    )
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await strat.on_bar(ctx, trigger)
    assert captured == [], "warmup must block all entries"


# ----------------------------------------------------------------------
# 2. ADX regime gate
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_signal_when_adx_above_threshold() -> None:
    """Trending M15 → ADX above max_adx → strategy stays flat."""
    strat = LiquiditySweepReversal()
    params = LiquiditySweepReversalParams(max_adx=20.0)
    m15 = _build_trending_m15(n=40)
    m5 = _build_range_m5(n=40)
    # Same proven short-sweep geometry; gate must block on regime alone.
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(15 * 40),
        o=2010.5, h=2012.0, lo=2007.0, c=2008.0,
    )
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await strat.on_bar(ctx, trigger)
    assert captured == [], "trending regime must block sweep entries"


# ----------------------------------------------------------------------
# 3. Short setup
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_short_signal_on_range_high_sweep_and_reentry() -> None:
    """M1 high pierces range_high + sweep_buf, close lands back below
    range_high - reentry_buf → SELL signal with SL above the wick."""
    strat = LiquiditySweepReversal()
    params = LiquiditySweepReversalParams(
        # Tight buffers so the geometry below works cleanly.
        sweep_buffer_atr=0.0,
        reentry_buffer_atr=0.0,
        stop_buffer_atr=0.1,
        min_rr=0.1,
    )
    m15 = _build_ranging_m15(n=40)
    m5 = _build_range_m5(n=40, range_high=2010.0, range_low=1990.0)
    # M1 wick: high=2012 (above range_high=2010), close=2008 (below range_high)
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(15 * 40),
        o=2010.5, h=2012.0, lo=2007.0, c=2008.0,
    )
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await strat.on_bar(ctx, trigger)

    assert len(captured) == 1, f"expected one SELL signal, got {captured}"
    sig = captured[0]
    assert sig.side is Side.SELL
    assert sig.order_type is OrderType.MARKET
    # SL must be above the sweep wick
    assert sig.suggested_sl is not None and sig.suggested_sl > trigger.high


# ----------------------------------------------------------------------
# 4. Long setup
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_long_signal_on_range_low_sweep_and_reentry() -> None:
    strat = LiquiditySweepReversal()
    params = LiquiditySweepReversalParams(
        sweep_buffer_atr=0.0,
        reentry_buffer_atr=0.0,
        stop_buffer_atr=0.1,
        min_rr=0.1,
    )
    m15 = _build_ranging_m15(n=40)
    m5 = _build_range_m5(n=40, range_high=2010.0, range_low=1990.0)
    # M1 wick: low=1988 (below range_low=1990), close=1992 (above range_low)
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(15 * 40),
        o=1990.5, h=1993.0, lo=1988.0, c=1992.0,
    )
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await strat.on_bar(ctx, trigger)

    assert len(captured) == 1
    sig = captured[0]
    assert sig.side is Side.BUY
    assert sig.suggested_sl is not None and sig.suggested_sl < trigger.low


# ----------------------------------------------------------------------
# 5. Close stays outside → no entry
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_entry_when_close_stays_outside_range() -> None:
    """A genuine breakout (high AND close above the range) is NOT a
    sweep reversal — the strategy must ignore it."""
    strat = LiquiditySweepReversal()
    params = LiquiditySweepReversalParams(
        sweep_buffer_atr=0.0, reentry_buffer_atr=0.0,
    )
    m15 = _build_ranging_m15(n=40)
    m5 = _build_range_m5(n=40, range_high=2010.0, range_low=1990.0)
    # Close above the range_high → no reversal setup.
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(15 * 40),
        o=2009, h=2013, lo=2008, c=2012,
    )
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await strat.on_bar(ctx, trigger)
    assert captured == []


# ----------------------------------------------------------------------
# 6. Existing position blocks new entry
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_entry_when_position_already_open() -> None:
    strat = LiquiditySweepReversal()
    params = LiquiditySweepReversalParams(
        sweep_buffer_atr=0.0, reentry_buffer_atr=0.0, min_rr=0.1,
    )
    existing = Position(
        ticket=1, symbol=SYMBOL, side=Side.SELL, volume=0.01,
        open_price=2010.0, open_time=_ts(0), magic=99,
    )
    m15 = _build_ranging_m15(n=40)
    m5 = _build_range_m5(n=40, range_high=2010.0, range_low=1990.0)
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(15 * 40),
        o=2010.5, h=2012.0, lo=2007.0, c=2008.0,
    )
    ctx, captured = _build_ctx(
        params=params, m5_bars=m5, m15_bars=m15, positions=[existing],
    )
    await strat.on_bar(ctx, trigger)
    assert captured == [], "must not double up while a position is open"


# ----------------------------------------------------------------------
# 7. Cooldown
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cooldown_blocks_entry_after_close() -> None:
    """Simulate a close: bump strategy state forward, advance fewer than
    cooldown_bars, and the next setup must be ignored."""
    strat = LiquiditySweepReversal()
    params = LiquiditySweepReversalParams(
        sweep_buffer_atr=0.0, reentry_buffer_atr=0.0, min_rr=0.1,
        cooldown_bars=10,
    )
    m15 = _build_ranging_m15(n=40)
    m5 = _build_range_m5(n=40, range_high=2010.0, range_low=1990.0)
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)

    # Force a "previous close just happened on the most recent bar".
    strat._bar_index = 100
    strat._last_close_bar_index = 100
    strat._maybe_roll_session(_ts(15 * 40), params)

    # Next bar arrives — still inside cooldown (only 1 bar elapsed).
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(15 * 40),
        o=2010.5, h=2012.0, lo=2007.0, c=2008.0,
    )
    await strat.on_bar(ctx, trigger)
    assert captured == [], "cooldown must block immediate re-entry"


# ----------------------------------------------------------------------
# 8. RR gate
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_setup_when_rr_below_min() -> None:
    """A geometry with TP very close to entry → RR below min_rr → skip."""
    strat = LiquiditySweepReversal()
    params = LiquiditySweepReversalParams(
        sweep_buffer_atr=0.0, reentry_buffer_atr=0.0,
        # In mid mode TP = channels.middle = 2000. For SHORT, reward =
        # entry - 2000 ≈ 8. Set min_rr=10 to force rejection.
        min_rr=10.0,
    )
    m15 = _build_ranging_m15(n=40)
    m5 = _build_range_m5(n=40, range_high=2010.0, range_low=1990.0)
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(15 * 40),
        o=2010.5, h=2012.0, lo=2007.0, c=2008.0,
    )
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await strat.on_bar(ctx, trigger)
    assert captured == [], "RR below min_rr must skip"


# ----------------------------------------------------------------------
# 9. Max trades per session
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_trades_per_session_hard_caps() -> None:
    """After N closes within one UTC day, no further setups fire even
    when geometry is perfect."""
    strat = LiquiditySweepReversal()
    params = LiquiditySweepReversalParams(
        sweep_buffer_atr=0.0, reentry_buffer_atr=0.0, min_rr=0.1,
        max_trades_per_session=2,
        cooldown_bars=0,
    )
    m15 = _build_ranging_m15(n=40)
    m5 = _build_range_m5(n=40, range_high=2010.0, range_low=1990.0)
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)

    # Pre-seed counter to the cap.
    strat._maybe_roll_session(_ts(15 * 40), params)
    strat._trades_this_session = 2

    trigger = _bar(
        tf=Timeframe.M1, t=_ts(15 * 40),
        o=2010.5, h=2012.0, lo=2007.0, c=2008.0,
    )
    await strat.on_bar(ctx, trigger)
    assert captured == [], "session cap must hard-stop new entries"


# ----------------------------------------------------------------------
# 10. SL/TP use ATR-scaled buffers
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sl_tp_use_atr_scaled_buffers() -> None:
    """SL must sit at bar.high + stop_buffer_atr * atr for a short.
    TP in mid mode must equal channels.middle."""
    strat = LiquiditySweepReversal()
    params = LiquiditySweepReversalParams(
        sweep_buffer_atr=0.0, reentry_buffer_atr=0.0,
        stop_buffer_atr=0.5,  # large enough that the assertion is sensitive
        tp_mode="mid",
        min_rr=0.1,
    )
    m15 = _build_ranging_m15(n=40)
    m5 = _build_range_m5(n=40, range_high=2010.0, range_low=1990.0)
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(15 * 40),
        o=2010.5, h=2012.0, lo=2007.0, c=2008.0,
    )
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await strat.on_bar(ctx, trigger)

    assert len(captured) == 1
    sig = captured[0]
    # SL geometry: bar.high + stop_buffer * atr; assert > bar.high
    # (we don't pin the exact ATR here — separate ATR test pins it).
    assert sig.suggested_sl is not None and sig.suggested_sl > trigger.high
    # TP = mid of channels — channels are seeded between 1990 and 2010
    # → mid ≈ 2000.
    assert sig.suggested_tp is not None
    assert 1995.0 <= sig.suggested_tp <= 2005.0


# ----------------------------------------------------------------------
# 11. tp_mode=fixed_r uses R-multiple
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tp_mode_fixed_r_uses_r_multiple() -> None:
    strat = LiquiditySweepReversal()
    params = LiquiditySweepReversalParams(
        sweep_buffer_atr=0.0, reentry_buffer_atr=0.0, stop_buffer_atr=0.1,
        tp_mode="fixed_r", take_profit_r=2.0, min_rr=0.1,
    )
    m15 = _build_ranging_m15(n=40)
    m5 = _build_range_m5(n=40, range_high=2010.0, range_low=1990.0)
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(15 * 40),
        o=2010.5, h=2012.0, lo=2007.0, c=2008.0,
    )
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await strat.on_bar(ctx, trigger)

    assert len(captured) == 1
    sig = captured[0]
    assert sig.side is Side.SELL
    assert sig.suggested_sl is not None and sig.suggested_tp is not None
    entry = trigger.close
    risk = sig.suggested_sl - entry
    reward = entry - sig.suggested_tp
    assert reward == pytest.approx(2.0 * risk, rel=1e-6)


# ----------------------------------------------------------------------
# 12. tp_mode=vwap falls back to mid when VWAP is None
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tp_mode_vwap_produces_finite_tp_near_session_vwap() -> None:
    """``tp_mode='vwap'`` must produce a finite TP computed from
    ``vwap_session`` over post-boundary M5 bars. Since seeded bars all
    have positive ``tick_volume`` (default 1000), vwap_session returns
    the weighted typical-price mean; for a symmetric synthetic range
    this lands inside the channel mid ± a small margin."""
    strat = LiquiditySweepReversal()
    params = LiquiditySweepReversalParams(
        sweep_buffer_atr=0.0, reentry_buffer_atr=0.0, stop_buffer_atr=0.1,
        tp_mode="vwap", min_rr=0.1,
    )
    m15 = _build_ranging_m15(n=40)
    m5 = _build_range_m5(n=40, range_high=2010.0, range_low=1990.0)
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(15 * 40),
        o=2010.5, h=2012.0, lo=2007.0, c=2008.0,
    )
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await strat.on_bar(ctx, trigger)

    assert len(captured) == 1
    sig = captured[0]
    assert sig.suggested_tp is not None
    # For the symmetric synthetic range the session VWAP lands near the
    # mid of the channel (~2000).
    assert 1995.0 <= sig.suggested_tp <= 2005.0


@pytest.mark.asyncio
async def test_tp_mode_vwap_falls_back_to_mid_when_session_empty() -> None:
    """If the session boundary sits after every M5 bar (e.g. the test
    config anchors at hour 18 but bars are all from morning), session
    bars is empty → ``vwap_session`` returns None → strategy falls
    through to ``channels.middle``."""
    strat = LiquiditySweepReversal()
    params = LiquiditySweepReversalParams(
        sweep_buffer_atr=0.0, reentry_buffer_atr=0.0, stop_buffer_atr=0.1,
        tp_mode="vwap", min_rr=0.1,
        # M5 bars span 00:00 - 03:15 UTC. Trigger at 10:00 UTC. Boundary
        # anchors at hour 18 today, so the boundary is 18:00 of the
        # PREVIOUS day → all M5 bars are after that boundary → not empty.
        # To get an empty session_bars we have to push the boundary
        # forward of the M5 data. The trick: schedule the trigger BEFORE
        # the boundary so it picks yesterday's 18:00 *one* day before
        # the M5 data starts. We anchor M5 data on Jan 1, trigger on
        # Jan 1, and boundary hour=18 → boundary = Dec 31 18:00. M5 bars
        # (Jan 1) are after that boundary, so not empty.
        # Practical empty-session: park the trigger pre-boundary AND have
        # M5 bars also pre-boundary in the same window. Use a far-back
        # base for M5 by giving them very early timestamps, then put the
        # trigger just past midnight with session_start=23.
        session_start_hour_utc=23, session_end_hour_utc=24,
        use_session_filter=False,
    )
    # Build M5 + M15 history entirely on the previous calendar day so
    # they sit BEFORE the most recent 23:00 boundary (which is at
    # Dec 31 23:00 because the trigger is Jan 1 00:05 → boundary picks
    # yesterday's 23:00). Result: ``[b for b in m5 if b.time >= boundary]``
    # is empty → vwap_session returns None → strategy uses channels.middle.
    shift = timedelta(days=1)
    m5 = [
        b.model_copy(update={"time": b.time - shift})
        for b in _build_range_m5(n=40, range_high=2010.0, range_low=1990.0)
    ]
    m15 = [
        b.model_copy(update={"time": b.time - shift})
        for b in _build_ranging_m15(n=40)
    ]
    trigger = _bar(
        tf=Timeframe.M1, t=datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
        o=2010.5, h=2012.0, lo=2007.0, c=2008.0,
    )
    ctx, captured = _build_ctx(params=params, m5_bars=m5, m15_bars=m15)
    await strat.on_bar(ctx, trigger)

    assert len(captured) == 1
    sig = captured[0]
    assert sig.suggested_tp is not None
    # Empty session bars → vwap_session returns None → TP = channels.middle
    # which for the symmetric synthetic range sits near 2000.
    assert 1998.0 <= sig.suggested_tp <= 2002.0


# ----------------------------------------------------------------------
# Lifecycle helpers (on_order_filled / on_position_closed / time exit)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_position_closed_resets_cooldown_and_session_counter() -> None:
    strat = LiquiditySweepReversal()
    params = LiquiditySweepReversalParams()
    ctx, _ = _build_ctx(params=params)
    # Pre-populate state as if a position has just closed.
    strat._bar_index = 50
    strat._entry_bar_by_ticket = {42: 40}
    await strat.on_position_closed(
        ctx,
        Position(
            ticket=42, symbol=SYMBOL, side=Side.BUY, volume=0.01,
            open_price=2000.0, open_time=_ts(0), magic=99,
        ),
    )
    assert 42 not in strat._entry_bar_by_ticket
    assert strat._last_close_bar_index == 50
    assert strat._trades_this_session == 1


@pytest.mark.asyncio
async def test_time_exit_closes_position_after_max_hold_bars() -> None:
    """Inject an open ticket into the strategy's map, fast-forward the
    bar counter past max_hold_bars, and verify ctx.close(...) was
    invoked (captured via the ClosePositionRequestEvent on the bus)."""
    from stinger_fx.core.events import ClosePositionRequestEvent

    strat = LiquiditySweepReversal()
    params = LiquiditySweepReversalParams(
        max_hold_bars=3, sweep_buffer_atr=0.0, reentry_buffer_atr=0.0,
    )
    ctx, _ = _build_ctx(params=params)

    closes: list[ClosePositionRequestEvent] = []

    async def collect(evt: ClosePositionRequestEvent) -> None:
        closes.append(evt)

    ctx._bus.subscribe(ClosePositionRequestEvent, collect)  # type: ignore[union-attr]

    # Set up: ticket 7 opened at bar index 0; current bar index is 4
    # → elapsed 4 ≥ max_hold_bars=3, time-exit must fire.
    strat._entry_bar_by_ticket = {7: 0}
    strat._bar_index = 3  # will become 4 after the increment in on_bar

    # Drive a bar through on_bar to trigger the time-exit check. Use a
    # non-sweep bar so the entry path stays inactive.
    bar = _bar(tf=Timeframe.M1, t=_ts(10), o=2000, h=2000.5, lo=1999.5, c=2000)
    await strat.on_bar(ctx, bar)
    for _ in range(3):
        await asyncio.sleep(0)

    assert len(closes) == 1, f"expected one ClosePositionRequestEvent, got {closes}"
    assert closes[0].ticket == 7
    assert "time_exit" in closes[0].reason
