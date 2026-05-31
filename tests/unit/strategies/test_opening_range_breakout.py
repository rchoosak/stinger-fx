"""Unit tests for OpeningRangeBreakout.

Strategy is exercised directly: build a multi-feed ``StrategyContext``
with two HistoryViews (M1 + M5), pre-fill them with hand-crafted bars,
then call ``strat.on_bar(ctx, bar)`` and inspect the signals captured
through ``signal_sink``.

Test parameters use shrunk warmup periods (``atr_period=5``) so the M5
fixture only needs ~10 bars.  The session anchor stays at UTC 07:00
(London open default) — fixtures are timestamped accordingly.

Fixture shapes
==============

* **OR window bars** — 15 M1 bars covering ``[07:00, 07:15)`` with a
  known high/low so the freeze pins to predictable values.
* **Pre-OR M5 history** — enough M5 bars before 07:00 so ``atr(5)``
  warmup is satisfied by the time we evaluate a trigger.
* **Trigger M1 bar** — a single bar AT or after ``07:15`` whose close
  pierces the OR boundary by ``breakout_buffer_atr × ATR``.
* **Confirmation M5 bar** — the last entry in the M5 history must
  itself close beyond the same OR boundary; we engineer it explicitly.
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
from stinger_fx.strategies.examples.opening_range_breakout import (
    OpeningRangeBreakout,
    OpeningRangeBreakoutParams,
)
from stinger_fx.strategies.managers.trailing import TrailingStopManager

SYMBOL = "XAUUSD"
SESSION_HOUR = 7   # London open default


def _ts(minutes_from_anchor: int) -> datetime:
    """Anchor at 2024-01-01 07:00 UTC (Monday, London open).  Offsets
    given in minutes from that anchor."""
    return datetime(2024, 1, 1, SESSION_HOUR, 0, tzinfo=UTC) + timedelta(
        minutes=minutes_from_anchor,
    )


def _bar(
    *, tf: Timeframe, t: datetime, o: float, h: float, lo: float, c: float,
    vol: int = 1000,
) -> Bar:
    return Bar(
        symbol=SYMBOL, timeframe=tf, time=t,
        open=o, high=h, low=lo, close=c,
        tick_volume=vol, is_closed=True,
    )


# ---------------------------------------------------------------------- #
# Reduced-warmup defaults                                                 #
# ---------------------------------------------------------------------- #


def _short_params(**overrides: object) -> OpeningRangeBreakoutParams:
    """Default test params.  ATR warmup is 5 so M5 history of ~10 bars
    is enough.  ``min_or_range_atr`` is relaxed so our hand-built OR
    (5 USD wide) passes the filter even when ATR computes as something
    small."""
    base: dict[str, object] = dict(
        atr_period=5,
        breakout_buffer_atr=0.0,        # tight buffer for clean test geometry
        stop_buffer_atr=0.3,
        min_or_range_atr=0.1,           # relaxed: 5-USD OR passes even tiny ATR
        max_or_range_atr=100.0,         # relaxed: don't reject for size
        min_rr=0.3,
        cooldown_bars=0,
        max_trades_per_session=1,
    )
    base.update(overrides)
    return OpeningRangeBreakoutParams(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------- #
# Fixture builders                                                         #
# ---------------------------------------------------------------------- #


def _or_window_m1_bars(*, high: float = 2342.0, low: float = 2338.0) -> list[Bar]:
    """15 M1 bars covering [07:00, 07:15) of the anchor date.  Bars
    oscillate inside [low, high]; bar 5 touches the high, bar 10
    touches the low — so after the window closes ``_or_high`` and
    ``_or_low`` pin exactly to these values."""
    bars: list[Bar] = []
    mid = (high + low) / 2
    for i in range(15):
        t = _ts(i)
        if i == 5:
            h, lo, c = high, mid - 0.5, mid + 0.5
        elif i == 10:
            h, lo, c = mid + 0.5, low, mid - 0.5
        else:
            h, lo, c = mid + 0.5, mid - 0.5, mid + (0.1 if i % 2 == 0 else -0.1)
        bars.append(_bar(tf=Timeframe.M1, t=t, o=mid, h=h, lo=lo, c=c))
    return bars


def _pre_anchor_m5_bars(
    *, n: int = 12, mid: float = 2340.0,
) -> list[Bar]:
    """Seed M5 history BEFORE the session anchor (07:00 UTC).  Each
    bar is 5 min wide; we stack them backwards from 07:00.  Used to
    satisfy the ``atr_period+1`` warmup gate before the trigger fires."""
    bars: list[Bar] = []
    for i in range(n, 0, -1):
        t = _ts(-5 * i)
        # Mild oscillation — ATR ends up a small positive number.
        c = mid + (0.2 if i % 2 == 0 else -0.2)
        bars.append(_bar(
            tf=Timeframe.M5, t=t, o=mid, h=mid + 0.4, lo=mid - 0.4, c=c,
        ))
    return bars


def _m5_confirm_bar(*, close: float, t_offset: int) -> Bar:
    """A single M5 bar that the strategy will read as the most recently
    closed M5 (i.e. last entry of `history_for(M5).bars()`).
    ``close`` is the only field that matters for the M5-confirmation
    gate; we make the bar wide enough that high/low are non-degenerate."""
    return _bar(
        tf=Timeframe.M5, t=_ts(t_offset),
        o=close - 0.2, h=close + 0.5, lo=close - 0.5, c=close,
    )


# ---------------------------------------------------------------------- #
# Context builder                                                         #
# ---------------------------------------------------------------------- #


def _build_ctx(
    *, params: OpeningRangeBreakoutParams,
    m1_bars: list[Bar] | None = None,
    m5_bars: list[Bar] | None = None,
    positions: list[Position] | None = None,
    clock_start: datetime | None = None,
) -> tuple[StrategyContext, list[Signal]]:
    bus = AsyncEventBus()
    captured: list[Signal] = []

    async def sink(sig: Signal) -> None:
        captured.append(sig)

    subs = OpeningRangeBreakout.subscriptions(params)
    ctx = StrategyContext(
        strategy_id="orb_test",
        symbol=params.symbol,
        timeframe=params.entry_timeframe,
        params=params,
        clock=SimClock(clock_start or _ts(0)),
        logger=structlog.get_logger("orb_test"),
        magic=99,
        signal_sink=sink,
        subscriptions=subs,
        bus=bus,
    )
    for b in m1_bars or []:
        view = ctx.history_for(params.symbol, params.entry_timeframe)
        assert view is not None
        view.append_bar(b)
    for b in m5_bars or []:
        view = ctx.history_for(params.symbol, params.structure_timeframe)
        assert view is not None
        view.append_bar(b)
    if positions:
        ctx.position.update([
            pos.model_copy(update={"magic": ctx.magic}) for pos in positions
        ])
    return ctx, captured


# ---------------------------------------------------------------------- #
# Tests — OR build / freeze                                                #
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_no_signal_before_or_window_closes() -> None:
    """A bar timestamped INSIDE the OR-build window must not trigger
    any entry — the OR isn't frozen yet."""
    strat = OpeningRangeBreakout()
    params = _short_params()
    # Trigger arrives at 07:10 (inside the 15-min window).
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(10),
        o=2340, h=2350, lo=2335, c=2350,  # would-be breakout, but OR not ready
    )
    ctx, captured = _build_ctx(
        params=params,
        m1_bars=_or_window_m1_bars()[:10],   # only 10 of 15 OR bars seeded
        m5_bars=_pre_anchor_m5_bars(),
    )
    await strat.on_bar(ctx, trigger)
    assert captured == [], "OR not yet frozen — must not trigger"
    assert strat._or_high is None and strat._or_low is None


@pytest.mark.asyncio
async def test_or_freezes_after_window_with_correct_extremes() -> None:
    """Drive a bar AFTER the OR window — strategy must freeze
    ``_or_high`` / ``_or_low`` to the exact high/low of the seeded
    window bars."""
    strat = OpeningRangeBreakout()
    params = _short_params()
    # Trigger at 07:16 — just past the window.  Bar geometry deliberately
    # benign so no entry fires even if gates pass.
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(16),
        o=2340, h=2340.5, lo=2339.5, c=2340,
    )
    ctx, _captured = _build_ctx(
        params=params,
        m1_bars=[*_or_window_m1_bars(high=2342.0, low=2338.0), trigger],
        m5_bars=_pre_anchor_m5_bars(),
    )
    await strat.on_bar(ctx, trigger)
    assert strat._or_high == pytest.approx(2342.0)
    assert strat._or_low == pytest.approx(2338.0)


@pytest.mark.asyncio
async def test_no_signal_when_atr_warmup_insufficient() -> None:
    """Fewer than atr_period+1 M5 bars → ATR is None → bail."""
    strat = OpeningRangeBreakout()
    params = _short_params()
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(20),
        o=2342, h=2343, lo=2342, c=2343,
    )
    ctx, captured = _build_ctx(
        params=params,
        m1_bars=[*_or_window_m1_bars(), trigger],
        # Only 3 M5 bars — well under atr_period+1=6.
        m5_bars=_pre_anchor_m5_bars(n=3),
    )
    await strat.on_bar(ctx, trigger)
    assert captured == [], "ATR warmup not satisfied — must skip"


# ---------------------------------------------------------------------- #
# Tests — Range-size filters                                               #
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_no_signal_when_or_range_too_small() -> None:
    """Narrow OR (range < min_or_range_atr × ATR) → skip."""
    strat = OpeningRangeBreakout()
    # Crank min_or_range_atr so even a 4-USD range fails (ATR is small).
    params = _short_params(min_or_range_atr=100.0)
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(20),
        o=2342, h=2343, lo=2342, c=2343,
    )
    confirm_m5 = _m5_confirm_bar(close=2343, t_offset=15)
    ctx, captured = _build_ctx(
        params=params,
        m1_bars=[*_or_window_m1_bars(), trigger],
        m5_bars=[*_pre_anchor_m5_bars(), confirm_m5],
    )
    await strat.on_bar(ctx, trigger)
    assert captured == [], "OR too narrow — must skip"


@pytest.mark.asyncio
async def test_no_signal_when_or_range_too_large() -> None:
    """Wide OR (range > max_or_range_atr × ATR) → skip (news guard)."""
    strat = OpeningRangeBreakout()
    # Crank max_or_range_atr down so 4-USD range exceeds it.
    params = _short_params(max_or_range_atr=0.1)
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(20),
        o=2342, h=2343, lo=2342, c=2343,
    )
    confirm_m5 = _m5_confirm_bar(close=2343, t_offset=15)
    ctx, captured = _build_ctx(
        params=params,
        m1_bars=[*_or_window_m1_bars(), trigger],
        m5_bars=[*_pre_anchor_m5_bars(), confirm_m5],
    )
    await strat.on_bar(ctx, trigger)
    assert captured == [], "OR too wide — must skip"


# ---------------------------------------------------------------------- #
# Tests — Breakout + M5 confirmation                                       #
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_long_signal_on_breakout_above_or_with_m5_confirm() -> None:
    """M1 close > or_high AND last closed M5 close > or_high → BUY."""
    strat = OpeningRangeBreakout()
    params = _short_params()
    # or_high = 2342.  Trigger M1 closes at 2343; confirm M5 closes at 2343.
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(20),
        o=2342, h=2343.5, lo=2342, c=2343,
    )
    confirm_m5 = _m5_confirm_bar(close=2343, t_offset=15)
    ctx, captured = _build_ctx(
        params=params,
        m1_bars=[*_or_window_m1_bars(), trigger],
        m5_bars=[*_pre_anchor_m5_bars(), confirm_m5],
    )
    await strat.on_bar(ctx, trigger)

    assert len(captured) == 1, f"expected one BUY signal, got {captured}"
    sig = captured[0]
    assert sig.side is Side.BUY
    assert sig.order_type is OrderType.MARKET
    assert "orb_long" in sig.comment


@pytest.mark.asyncio
async def test_short_signal_on_breakdown_below_or_with_m5_confirm() -> None:
    """Mirror: M1 close < or_low AND last M5 close < or_low → SELL."""
    strat = OpeningRangeBreakout()
    params = _short_params()
    # or_low = 2338.  Trigger closes at 2337; confirm M5 closes at 2337.
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(20),
        o=2338, h=2338, lo=2336.5, c=2337,
    )
    confirm_m5 = _m5_confirm_bar(close=2337, t_offset=15)
    ctx, captured = _build_ctx(
        params=params,
        m1_bars=[*_or_window_m1_bars(), trigger],
        m5_bars=[*_pre_anchor_m5_bars(), confirm_m5],
    )
    await strat.on_bar(ctx, trigger)

    assert len(captured) == 1, f"expected one SELL signal, got {captured}"
    sig = captured[0]
    assert sig.side is Side.SELL
    assert "orb_short" in sig.comment


@pytest.mark.asyncio
async def test_no_signal_when_m1_breaks_but_m5_close_inside_or() -> None:
    """Wick-spike fake breakout: M1 breaks but last closed M5 is still
    inside the OR → no entry."""
    strat = OpeningRangeBreakout()
    params = _short_params()
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(20),
        o=2342, h=2343.5, lo=2342, c=2343,  # M1 breaks above 2342
    )
    # Last closed M5 closes INSIDE the OR (2340 < 2342) → no confirmation.
    confirm_m5 = _m5_confirm_bar(close=2340, t_offset=15)
    ctx, captured = _build_ctx(
        params=params,
        m1_bars=[*_or_window_m1_bars(), trigger],
        m5_bars=[*_pre_anchor_m5_bars(), confirm_m5],
    )
    await strat.on_bar(ctx, trigger)
    assert captured == [], "M5 didn't confirm — must skip"


# ---------------------------------------------------------------------- #
# Tests — Session / window guards                                          #
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_no_signal_when_past_max_entry_window() -> None:
    """Trigger arrives after ``max_entry_minutes_from_open`` → skip."""
    strat = OpeningRangeBreakout()
    params = _short_params(max_entry_minutes_from_open=30)
    # Trigger at 07:45 — 45 min after open, past the 30-min cap.
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(45),
        o=2342, h=2343.5, lo=2342, c=2343,
    )
    confirm_m5 = _m5_confirm_bar(close=2343, t_offset=40)
    ctx, captured = _build_ctx(
        params=params,
        m1_bars=[*_or_window_m1_bars(), trigger],
        m5_bars=[*_pre_anchor_m5_bars(), confirm_m5],
    )
    await strat.on_bar(ctx, trigger)
    assert captured == [], "past max-entry window — must skip"


@pytest.mark.asyncio
async def test_no_signal_when_outside_session_hours() -> None:
    """Bar before session_start_hour_utc → skip entirely."""
    strat = OpeningRangeBreakout()
    params = _short_params()
    # 02:00 UTC — before London open at 07:00.
    pre_open = datetime(2024, 1, 1, 2, 0, tzinfo=UTC)
    trigger = _bar(
        tf=Timeframe.M1, t=pre_open,
        o=2342, h=2343.5, lo=2342, c=2343,
    )
    ctx, captured = _build_ctx(
        params=params, m1_bars=[trigger], m5_bars=_pre_anchor_m5_bars(),
    )
    await strat.on_bar(ctx, trigger)
    assert captured == [], "outside session hours — must skip"


@pytest.mark.asyncio
async def test_no_entry_when_position_already_open() -> None:
    strat = OpeningRangeBreakout()
    params = _short_params()
    existing = Position(
        ticket=1, symbol=SYMBOL, side=Side.BUY, volume=0.01,
        open_price=2340.0, open_time=_ts(0), magic=99,
    )
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(20),
        o=2342, h=2343.5, lo=2342, c=2343,
    )
    confirm_m5 = _m5_confirm_bar(close=2343, t_offset=15)
    ctx, captured = _build_ctx(
        params=params,
        m1_bars=[*_or_window_m1_bars(), trigger],
        m5_bars=[*_pre_anchor_m5_bars(), confirm_m5],
        positions=[existing],
    )
    await strat.on_bar(ctx, trigger)
    assert captured == [], "must not double up while a position is open"


@pytest.mark.asyncio
async def test_max_trades_per_session_hard_caps() -> None:
    """With ``_trades_this_session`` already at the cap, no new entry."""
    strat = OpeningRangeBreakout()
    params = _short_params(max_trades_per_session=1)
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(20),
        o=2342, h=2343.5, lo=2342, c=2343,
    )
    confirm_m5 = _m5_confirm_bar(close=2343, t_offset=15)
    ctx, captured = _build_ctx(
        params=params,
        m1_bars=[*_or_window_m1_bars(), trigger],
        m5_bars=[*_pre_anchor_m5_bars(), confirm_m5],
    )
    # Pre-anchor the session key so the rollover doesn't reset our
    # counter, then bump the counter to the cap.
    strat._maybe_roll_session(_ts(20), params)
    strat._trades_this_session = 1
    await strat.on_bar(ctx, trigger)
    assert captured == [], "session cap must hard-stop new entries"


@pytest.mark.asyncio
async def test_session_rollover_resets_or_and_counters() -> None:
    """Drive a bar at next day's session anchor — the OR + session
    counter must reset to fresh state."""
    strat = OpeningRangeBreakout()
    params = _short_params()
    # Pre-populate as if we were mid-session yesterday with OR frozen
    # and 1 trade already taken.
    yesterday_anchor = datetime(2023, 12, 31, SESSION_HOUR, 0, tzinfo=UTC)
    strat._or_high = 2342.0
    strat._or_low = 2338.0
    strat._or_built_at = yesterday_anchor + timedelta(minutes=15)
    strat._trades_this_session = 1
    strat._session_key = yesterday_anchor.date().isoformat()

    ctx, _ = _build_ctx(params=params)
    # Now drive a bar at today's anchor (a full UTC day later).
    today_bar = _bar(
        tf=Timeframe.M1, t=_ts(0),
        o=2340, h=2340.5, lo=2339.5, c=2340,
    )
    await strat.on_bar(ctx, today_bar)
    assert strat._or_high is None
    assert strat._or_low is None
    assert strat._or_built_at is None
    assert strat._trades_this_session == 0


@pytest.mark.asyncio
async def test_skip_friday_late_blocks_entry() -> None:
    """``skip_friday_late=True`` + Friday >= ``friday_late_hour_utc`` → block."""
    strat = OpeningRangeBreakout()
    params = _short_params(
        skip_friday_late=True, friday_late_hour_utc=14,
        session_start_hour_utc=7, session_end_hour_utc=24,
    )
    # 2024-01-05 was a Friday.  20:00 UTC > 14:00 cutoff.
    friday_late = datetime(2024, 1, 5, 20, 0, tzinfo=UTC)
    trigger = _bar(
        tf=Timeframe.M1, t=friday_late,
        o=2342, h=2343.5, lo=2342, c=2343,
    )
    ctx, captured = _build_ctx(
        params=params, m1_bars=[trigger], m5_bars=_pre_anchor_m5_bars(),
    )
    await strat.on_bar(ctx, trigger)
    assert captured == [], "Friday late must skip when skip_friday_late=True"


# ---------------------------------------------------------------------- #
# Tests — SL modes                                                          #
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sl_mode_opposite_or_long_equals_or_low() -> None:
    strat = OpeningRangeBreakout()
    params = _short_params(sl_mode="opposite_or")
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(20),
        o=2342, h=2343.5, lo=2342, c=2343,
    )
    confirm_m5 = _m5_confirm_bar(close=2343, t_offset=15)
    ctx, captured = _build_ctx(
        params=params,
        m1_bars=[*_or_window_m1_bars(high=2342.0, low=2338.0), trigger],
        m5_bars=[*_pre_anchor_m5_bars(), confirm_m5],
    )
    await strat.on_bar(ctx, trigger)
    assert len(captured) == 1
    sig = captured[0]
    assert sig.suggested_sl == pytest.approx(2338.0)


@pytest.mark.asyncio
async def test_sl_mode_or_mid_equals_midpoint() -> None:
    strat = OpeningRangeBreakout()
    params = _short_params(sl_mode="or_mid")
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(20),
        o=2342, h=2343.5, lo=2342, c=2343,
    )
    confirm_m5 = _m5_confirm_bar(close=2343, t_offset=15)
    ctx, captured = _build_ctx(
        params=params,
        m1_bars=[*_or_window_m1_bars(high=2342.0, low=2338.0), trigger],
        m5_bars=[*_pre_anchor_m5_bars(), confirm_m5],
    )
    await strat.on_bar(ctx, trigger)
    assert len(captured) == 1
    sig = captured[0]
    assert sig.suggested_sl == pytest.approx(2340.0)   # (2342+2338)/2


@pytest.mark.asyncio
async def test_sl_mode_atr_long_equals_entry_minus_buffer() -> None:
    from stinger_fx.strategies.indicators import atr as atr_fn

    strat = OpeningRangeBreakout()
    params = _short_params(sl_mode="atr", stop_buffer_atr=0.5)
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(20),
        o=2342, h=2343.5, lo=2342, c=2343,
    )
    confirm_m5 = _m5_confirm_bar(close=2343, t_offset=15)
    m5_bars = [*_pre_anchor_m5_bars(), confirm_m5]
    ctx, captured = _build_ctx(
        params=params,
        m1_bars=[*_or_window_m1_bars(), trigger],
        m5_bars=m5_bars,
    )
    await strat.on_bar(ctx, trigger)
    assert len(captured) == 1
    sig = captured[0]
    atr_value = atr_fn(m5_bars, params.atr_period)
    assert atr_value is not None
    expected_sl = trigger.close - params.stop_buffer_atr * atr_value
    assert sig.suggested_sl == pytest.approx(expected_sl, rel=1e-6)


# ---------------------------------------------------------------------- #
# Tests — TP modes                                                          #
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_tp_mode_or_range_uses_measured_move() -> None:
    """tp_mode='or_range' for a long → TP = entry + or_range."""
    strat = OpeningRangeBreakout()
    params = _short_params(tp_mode="or_range")
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(20),
        o=2342, h=2343.5, lo=2342, c=2343,
    )
    confirm_m5 = _m5_confirm_bar(close=2343, t_offset=15)
    ctx, captured = _build_ctx(
        params=params,
        m1_bars=[*_or_window_m1_bars(high=2342.0, low=2338.0), trigger],
        m5_bars=[*_pre_anchor_m5_bars(), confirm_m5],
    )
    await strat.on_bar(ctx, trigger)
    assert len(captured) == 1
    sig = captured[0]
    expected_tp = trigger.close + (2342.0 - 2338.0)    # entry + or_range
    assert sig.suggested_tp == pytest.approx(expected_tp)


@pytest.mark.asyncio
async def test_tp_mode_fixed_r_uses_r_multiple() -> None:
    """tp_mode='fixed_r' → TP = entry + take_profit_r × risk."""
    strat = OpeningRangeBreakout()
    params = _short_params(tp_mode="fixed_r", take_profit_r=2.0)
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(20),
        o=2342, h=2343.5, lo=2342, c=2343,
    )
    confirm_m5 = _m5_confirm_bar(close=2343, t_offset=15)
    ctx, captured = _build_ctx(
        params=params,
        m1_bars=[*_or_window_m1_bars(high=2342.0, low=2338.0), trigger],
        m5_bars=[*_pre_anchor_m5_bars(), confirm_m5],
    )
    await strat.on_bar(ctx, trigger)
    assert len(captured) == 1
    sig = captured[0]
    assert sig.suggested_sl is not None and sig.suggested_tp is not None
    entry = trigger.close
    risk = entry - sig.suggested_sl
    reward = sig.suggested_tp - entry
    assert reward == pytest.approx(2.0 * risk, rel=1e-6)


@pytest.mark.asyncio
async def test_tp_mode_trailing_attaches_manager_and_omits_tp() -> None:
    """tp_mode='trailing' attaches a TrailingStopManager and the emitted
    signal has tp=None (SL serves as initial protection)."""
    strat = OpeningRangeBreakout()
    params = _short_params(tp_mode="trailing")
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(20),
        o=2342, h=2343.5, lo=2342, c=2343,
    )
    confirm_m5 = _m5_confirm_bar(close=2343, t_offset=15)
    ctx, captured = _build_ctx(
        params=params,
        m1_bars=[*_or_window_m1_bars(), trigger],
        m5_bars=[*_pre_anchor_m5_bars(), confirm_m5],
    )
    assert ctx.managers == []
    await strat.on_bar(ctx, trigger)

    assert len(captured) == 1
    sig = captured[0]
    assert sig.suggested_tp is None
    assert sig.suggested_sl is not None
    assert len(ctx.managers) == 1
    assert isinstance(ctx.managers[0], TrailingStopManager)


# ---------------------------------------------------------------------- #
# Tests — RR + time exit                                                   #
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_rejects_setup_when_rr_below_min() -> None:
    strat = OpeningRangeBreakout()
    params = _short_params(min_rr=100.0)
    trigger = _bar(
        tf=Timeframe.M1, t=_ts(20),
        o=2342, h=2343.5, lo=2342, c=2343,
    )
    confirm_m5 = _m5_confirm_bar(close=2343, t_offset=15)
    ctx, captured = _build_ctx(
        params=params,
        m1_bars=[*_or_window_m1_bars(), trigger],
        m5_bars=[*_pre_anchor_m5_bars(), confirm_m5],
    )
    await strat.on_bar(ctx, trigger)
    assert captured == [], "RR below min_rr must skip"


@pytest.mark.asyncio
async def test_time_exit_closes_position_after_max_hold_bars() -> None:
    from stinger_fx.core.events import ClosePositionRequestEvent

    strat = OpeningRangeBreakout()
    params = _short_params(max_hold_bars=3)
    ctx, _ = _build_ctx(params=params)

    closes: list[ClosePositionRequestEvent] = []

    async def collect(evt: ClosePositionRequestEvent) -> None:
        closes.append(evt)

    ctx._bus.subscribe(ClosePositionRequestEvent, collect)  # type: ignore[union-attr]

    # Ticket 7 opened at bar index 0; we sit at index 3 → on_bar
    # increments to 4 → elapsed = 4 ≥ 3 → close.
    strat._entry_bar_by_ticket = {7: 0}
    strat._bar_index = 3

    bar = _bar(
        tf=Timeframe.M1, t=_ts(10), o=2340, h=2340.5, lo=2339.5, c=2340,
    )
    await strat.on_bar(ctx, bar)
    for _ in range(3):
        await asyncio.sleep(0)

    assert len(closes) == 1, f"expected one ClosePositionRequestEvent, got {closes}"
    assert closes[0].ticket == 7
    assert "time_exit" in closes[0].reason
