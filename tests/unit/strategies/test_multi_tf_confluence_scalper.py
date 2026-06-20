"""Unit tests for MultiTimeframeConfluenceScalper.

Same harness as PRS / MBS / OFIS — monkeypatch the indicators so the
*decision logic* (3-TF agreement + M1 RSI cross-up out of oversold) is
what we're verifying, not the indicator math.  ``ema`` / ``rsi`` calls
are routed by series length so the test can drive each timeframe
deterministically without managing call-order brittleness.  The cross
test needs a *previous* RSI, so entry tests seed ``strat._prev_m1_rsi``
directly (the stub supplies only the current bar's RSI).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import structlog

import stinger_fx.strategies.examples.multi_tf_confluence_scalper as mtc_mod
from stinger_fx.core import AsyncEventBus, SimClock
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
from stinger_fx.strategies.examples.multi_tf_confluence_scalper import (
    MultiTFConfluenceScalper,
    MultiTFConfluenceScalperParams,
)

SYMBOL = "XAUUSD"

# History lengths the fixture seeds for each timeframe.  Numbers
# chosen so the indicator stub can dispatch by ``len(closes)`` without
# overlap between feeds.
M1_HIST = 80
M5_HIST = 30
M15_HIST = 15


def _ts(hour: int = 10, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2024, 1, 3, hour, minute, second, tzinfo=UTC)


def _bar(*, tf: Timeframe, t: datetime, close: float = 2340.0,
         high: float | None = None, low: float | None = None) -> Bar:
    return Bar(
        symbol=SYMBOL, timeframe=tf, time=t,
        open=close, high=high if high is not None else close + 0.5,
        low=low if low is not None else close - 0.5, close=close,
        tick_volume=1000, is_closed=True,
    )


def _short_params(**overrides: Any) -> MultiTFConfluenceScalperParams:
    base: dict[str, Any] = dict(
        m5_ema_fast=5,
        m5_ema_slow=10,
        m15_ema_fast=5,
        m15_ema_slow=10,
        m1_rsi_period=5,
        m1_rsi_oversold=30.0,
        m1_rsi_overbought=70.0,
        avoid_hours_utc=[13, 14, 15, 18],
        vol_gate_lookback=5,
        vol_gate_mult=1.8,
        atr_period=5,
        sl_atr_mult=1.0,
        tp_atr_mult=2.0,
        max_hold_bars_m1=10,
        cooldown_bars_m1=2,
    )
    base.update(overrides)
    return MultiTFConfluenceScalperParams(**base)


def _patch_indicators(monkeypatch, *,
                     m1_rsi_value: float = 25.0,        # M1 oversold by default
                     m5_ema_fast: float = 2336.0,
                     m5_ema_slow: float = 2330.0,
                     m15_ema_fast: float = 2340.0,
                     m15_ema_slow: float = 2335.0,
                     atr_value: float = 1.0,
                     tr_series: list[float] | None = None) -> dict[str, list[Any]]:
    """Dispatch indicators by series length.

    Series lengths in the fixture:
      * M1  ≈ 80
      * M5  ≈ 30
      * M15 ≈ 15

    EMA: by closes length → M5 vs M15.  Each timeframe needs two EMA
    answers (fast, slow); we route by an explicit call-counter.
    RSI: called only on M1 closes (length ≈ 80).
    """
    state: dict[str, list[Any]] = {
        "ema_calls": [],     # list of (period, len)
        "rsi_calls": [],
    }

    def _ema_stub(closes, period):  # type: ignore[no-untyped-def]
        n = len(closes)
        state["ema_calls"].append((period, n))
        if n < 20:
            # M15 feed (~15 bars)
            same_tf = [c for c in state["ema_calls"] if c[1] == n]
            return m15_ema_fast if len(same_tf) == 1 else m15_ema_slow
        # M5 feed (~30 bars)
        same_tf = [c for c in state["ema_calls"] if c[1] == n]
        return m5_ema_fast if len(same_tf) == 1 else m5_ema_slow

    def _rsi_stub(closes, period):  # type: ignore[no-untyped-def]
        state["rsi_calls"].append((period, len(closes)))
        return m1_rsi_value

    def _atr_stub(bars, period):  # type: ignore[no-untyped-def]
        return atr_value

    def _tr_stub(bars):  # type: ignore[no-untyped-def]
        if tr_series is not None:
            return list(tr_series)
        return [1.0] * 20

    def _sma_stub(values, period):  # type: ignore[no-untyped-def]
        if period <= 0 or len(values) < period:
            return None
        return sum(values[-period:]) / period

    monkeypatch.setattr(mtc_mod, "ema", _ema_stub)
    monkeypatch.setattr(mtc_mod, "rsi", _rsi_stub)
    monkeypatch.setattr(mtc_mod, "atr", _atr_stub)
    monkeypatch.setattr(mtc_mod, "sma", _sma_stub)
    monkeypatch.setattr(mtc_mod, "_true_range_series", _tr_stub)
    return state


def _build_ctx(*, params: MultiTFConfluenceScalperParams,
              m1_count: int = M1_HIST,
              m5_count: int = M5_HIST,
              m15_count: int = M15_HIST) -> tuple[
                  StrategyContext, list[Signal], AsyncEventBus]:
    """Build a context with M1 + M5 + M15 history.  All bar geometry
    is irrelevant — patched stubs drive every decision; we only need
    the bar *counts* to clear the early-return gates."""
    bus = AsyncEventBus()
    captured: list[Signal] = []

    async def sink(sig: Signal) -> None:
        captured.append(sig)

    ctx = StrategyContext(
        strategy_id="mtc_test",
        symbol=params.symbol,
        timeframe=params.entry_timeframe,
        params=params,
        clock=SimClock(_ts(10)),
        logger=structlog.get_logger("mtc_test"),
        magic=99,
        signal_sink=sink,
        subscriptions=MultiTFConfluenceScalper.subscriptions(params),
        bus=bus,
    )
    m1_view = ctx.history_for(params.symbol, params.entry_timeframe)
    assert m1_view is not None
    for i in range(m1_count):
        m1_view.append_bar(_bar(tf=Timeframe.M1, t=_ts(10) + timedelta(seconds=i)))
    m5_view = ctx.history_for(params.symbol, params.mid_timeframe)
    assert m5_view is not None
    for i in range(m5_count):
        m5_view.append_bar(
            _bar(tf=Timeframe.M5, t=_ts(10) + timedelta(minutes=5 * i)),
        )
    m15_view = ctx.history_for(params.symbol, params.high_timeframe)
    assert m15_view is not None
    for i in range(m15_count):
        m15_view.append_bar(
            _bar(tf=Timeframe.M15, t=_ts(10) + timedelta(minutes=15 * i)),
        )
    return ctx, captured, bus


def _trigger_bar(close: float = 2340.0, hour: int = 10,
                minute: int = 30) -> Bar:
    return _bar(tf=Timeframe.M1, t=_ts(hour, minute, 0), close=close)


# =========================================================================
# Entry tests
# =========================================================================


@pytest.mark.asyncio
async def test_emits_buy_on_rsi_cross_up_in_aligned_uptrend(monkeypatch) -> None:
    """Canonical happy path: M15 up + M5 up + M1 RSI crossing UP out of
    oversold (prev 25 < 30 <= 35 current)."""
    params = _short_params()
    _patch_indicators(
        monkeypatch,
        m1_rsi_value=35.0,                          # current: back above 30
        m5_ema_fast=2336.0, m5_ema_slow=2330.0,     # M5 up
        m15_ema_fast=2340.0, m15_ema_slow=2335.0,   # M15 up
    )
    strat = MultiTFConfluenceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._prev_m1_rsi = 25.0                       # previous: was oversold

    await strat.on_bar(ctx, _trigger_bar(close=2340.0))

    assert len(captured) == 1, f"expected BUY; got {captured}"
    sig = captured[0]
    assert sig.side is Side.BUY
    assert sig.suggested_sl == pytest.approx(2339.0)   # entry - 1 * ATR
    assert sig.suggested_tp == pytest.approx(2342.0)   # entry + 2 * ATR


@pytest.mark.asyncio
async def test_emits_sell_on_rsi_cross_down_in_aligned_downtrend(monkeypatch) -> None:
    """M15 down + M5 down + M1 RSI crossing DOWN out of overbought
    (prev 75 > 70 >= 65 current)."""
    params = _short_params()
    _patch_indicators(
        monkeypatch,
        m1_rsi_value=65.0,                          # current: back below 70
        m5_ema_fast=2330.0, m5_ema_slow=2336.0,     # M5 down
        m15_ema_fast=2335.0, m15_ema_slow=2340.0,   # M15 down
    )
    strat = MultiTFConfluenceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._prev_m1_rsi = 75.0                       # previous: was overbought

    await strat.on_bar(ctx, _trigger_bar(close=2340.0))

    assert len(captured) == 1
    sig = captured[0]
    assert sig.side is Side.SELL
    assert sig.suggested_sl == pytest.approx(2341.0)   # entry + ATR
    assert sig.suggested_tp == pytest.approx(2338.0)   # entry - 2 * ATR


@pytest.mark.asyncio
async def test_no_buy_when_rsi_still_in_oversold_zone(monkeypatch) -> None:
    """The whole point of cross-confirmation: RSI deep in oversold and
    NOT yet recovered must not fire a BUY — no catching the falling
    knife.  prev 18 and current 22 are both < 30 → no cross-up."""
    params = _short_params()
    _patch_indicators(
        monkeypatch,
        m1_rsi_value=22.0,                          # still < 30
        m5_ema_fast=2336.0, m5_ema_slow=2330.0,     # M5 up
        m15_ema_fast=2340.0, m15_ema_slow=2335.0,   # M15 up
    )
    strat = MultiTFConfluenceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._prev_m1_rsi = 18.0                       # prev also oversold
    await strat.on_bar(ctx, _trigger_bar())
    assert captured == []


@pytest.mark.asyncio
async def test_no_sell_when_rsi_still_in_overbought_zone(monkeypatch) -> None:
    """Symmetric falling-knife guard for sells: prev 82 and current 78
    are both > 70 → no cross-down, no SELL."""
    params = _short_params()
    _patch_indicators(
        monkeypatch,
        m1_rsi_value=78.0,                          # still > 70
        m5_ema_fast=2330.0, m5_ema_slow=2336.0,     # M5 down
        m15_ema_fast=2335.0, m15_ema_slow=2340.0,   # M15 down
    )
    strat = MultiTFConfluenceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._prev_m1_rsi = 82.0
    await strat.on_bar(ctx, _trigger_bar())
    assert captured == []


@pytest.mark.asyncio
async def test_no_signal_on_first_bar_without_prev_rsi(monkeypatch) -> None:
    """With no prior RSI there is nothing to cross *from*, so even a
    perfect alignment cannot fire — but the tracker is primed for the
    next bar."""
    params = _short_params()
    _patch_indicators(
        monkeypatch,
        m1_rsi_value=35.0,
        m5_ema_fast=2336.0, m5_ema_slow=2330.0,
        m15_ema_fast=2340.0, m15_ema_slow=2335.0,
    )
    strat = MultiTFConfluenceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    # _prev_m1_rsi is None on a fresh strategy
    await strat.on_bar(ctx, _trigger_bar())
    assert captured == []
    assert strat._prev_m1_rsi == pytest.approx(35.0)   # primed for next bar


@pytest.mark.asyncio
async def test_no_buy_when_m15_disagrees(monkeypatch) -> None:
    """Valid M1 cross-up + M5 up but M15 DOWN → no signal.  Confluence
    requires all three to agree."""
    params = _short_params()
    _patch_indicators(
        monkeypatch,
        m1_rsi_value=35.0,
        m5_ema_fast=2336.0, m5_ema_slow=2330.0,     # M5 up
        m15_ema_fast=2335.0, m15_ema_slow=2340.0,   # M15 down ← blocks
    )
    strat = MultiTFConfluenceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._prev_m1_rsi = 25.0                       # valid cross would fire
    await strat.on_bar(ctx, _trigger_bar())
    assert captured == []


@pytest.mark.asyncio
async def test_no_buy_when_m5_disagrees(monkeypatch) -> None:
    params = _short_params()
    _patch_indicators(
        monkeypatch,
        m1_rsi_value=35.0,
        m5_ema_fast=2330.0, m5_ema_slow=2336.0,     # M5 down ← blocks
        m15_ema_fast=2340.0, m15_ema_slow=2335.0,   # M15 up
    )
    strat = MultiTFConfluenceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._prev_m1_rsi = 25.0
    await strat.on_bar(ctx, _trigger_bar())
    assert captured == []


@pytest.mark.asyncio
async def test_session_avoid_blocks_entry(monkeypatch) -> None:
    params = _short_params()
    _patch_indicators(
        monkeypatch,
        m1_rsi_value=35.0,
        m5_ema_fast=2336.0, m5_ema_slow=2330.0,
        m15_ema_fast=2340.0, m15_ema_slow=2335.0,
    )
    strat = MultiTFConfluenceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._prev_m1_rsi = 25.0                       # valid cross would fire
    # 14:30 UTC is in default avoid list
    await strat.on_bar(ctx, _trigger_bar(hour=14, minute=30))
    assert captured == []


@pytest.mark.asyncio
async def test_vol_gate_blocks_entry_on_spike(monkeypatch) -> None:
    params = _short_params(vol_gate_mult=1.8, vol_gate_lookback=5)
    _patch_indicators(
        monkeypatch,
        m1_rsi_value=35.0,
        m5_ema_fast=2336.0, m5_ema_slow=2330.0,
        m15_ema_fast=2340.0, m15_ema_slow=2335.0,
        tr_series=[1.0, 1.0, 1.0, 1.0, 1.0, 5.0],   # spike on the last bar
    )
    strat = MultiTFConfluenceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._prev_m1_rsi = 25.0                       # valid cross would fire
    await strat.on_bar(ctx, _trigger_bar())
    assert captured == []


@pytest.mark.asyncio
async def test_tp_disabled_when_mult_is_zero(monkeypatch) -> None:
    params = _short_params(tp_atr_mult=0.0)
    _patch_indicators(
        monkeypatch,
        m1_rsi_value=35.0,
        m5_ema_fast=2336.0, m5_ema_slow=2330.0,
        m15_ema_fast=2340.0, m15_ema_slow=2335.0,
    )
    strat = MultiTFConfluenceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._prev_m1_rsi = 25.0
    await strat.on_bar(ctx, _trigger_bar())
    assert len(captured) == 1
    assert captured[0].suggested_tp is None


# =========================================================================
# Exit + lifecycle
# =========================================================================


@pytest.mark.asyncio
async def test_time_stop_closes_position_after_max_hold(monkeypatch) -> None:
    params = _short_params(max_hold_bars_m1=3)
    _patch_indicators(monkeypatch)
    strat = MultiTFConfluenceScalper()
    ctx, _captured, bus = _build_ctx(params=params)
    strat._open_ticket = 42
    strat._open_side = Side.BUY
    strat._open_bars = 2

    close_events: list[Any] = []
    from stinger_fx.core.events import ClosePositionRequestEvent
    bus.subscribe(ClosePositionRequestEvent,
                  lambda e: close_events.append(e),
                  name="probe.close")
    await strat.on_bar(ctx, _trigger_bar())
    import asyncio
    for _ in range(3):
        await asyncio.sleep(0)
    assert len(close_events) == 1


@pytest.mark.asyncio
async def test_on_order_filled_records_open_ticket() -> None:
    strat = MultiTFConfluenceScalper()
    params = _short_params()
    ctx, _captured, _ = _build_ctx(params=params)
    order = Order(
        ticket=101, strategy_id=ctx.strategy_id,
        symbol=SYMBOL, side=Side.BUY,
        type=OrderType.MARKET, volume=0.01,
        status=OrderStatus.FILLED,
        filled_volume=0.01, fill_price=2340.0,
    )
    await strat.on_order_filled(ctx, order)
    assert strat._open_ticket == 101


@pytest.mark.asyncio
async def test_on_position_closed_starts_cooldown() -> None:
    strat = MultiTFConfluenceScalper()
    params = _short_params(cooldown_bars_m1=4)
    ctx, _captured, _ = _build_ctx(params=params)
    strat._open_ticket = 55
    strat._open_side = Side.BUY
    strat._open_bars = 7
    pos = Position(
        ticket=55, symbol=SYMBOL, side=Side.BUY,
        volume=0.01, open_price=2340.0, open_time=_ts(10),
    )
    await strat.on_position_closed(ctx, pos)
    assert strat._open_ticket is None
    assert strat._cooldown_left == 4


@pytest.mark.asyncio
async def test_cooldown_blocks_immediate_re_entry(monkeypatch) -> None:
    params = _short_params(cooldown_bars_m1=2)
    _patch_indicators(
        monkeypatch,
        m1_rsi_value=35.0,                          # current: crossed above 30
        m5_ema_fast=2336.0, m5_ema_slow=2330.0,
        m15_ema_fast=2340.0, m15_ema_slow=2335.0,
    )
    strat = MultiTFConfluenceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._cooldown_left = 2
    strat._prev_m1_rsi = 25.0

    await strat.on_bar(ctx, _trigger_bar())   # cooldown 2 → 1
    assert captured == []
    assert strat._cooldown_left == 1
    await strat.on_bar(ctx, _trigger_bar())   # cooldown 1 → 0
    assert captured == []
    assert strat._cooldown_left == 0
    # RSI dipped back into oversold during the wait, now crosses up:
    strat._prev_m1_rsi = 25.0
    await strat.on_bar(ctx, _trigger_bar())   # fires
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_only_one_open_position_at_a_time(monkeypatch) -> None:
    params = _short_params()
    _patch_indicators(
        monkeypatch,
        m1_rsi_value=25.0,
        m5_ema_fast=2336.0, m5_ema_slow=2330.0,
        m15_ema_fast=2340.0, m15_ema_slow=2335.0,
    )
    strat = MultiTFConfluenceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._open_ticket = 999
    strat._open_side = Side.BUY
    strat._open_bars = 1
    strat._prev_m1_rsi = 25.0                       # a cross would fire if free
    await strat.on_bar(ctx, _trigger_bar())
    assert captured == [], "no new signal while a position is open"


@pytest.mark.asyncio
async def test_prev_rsi_tracks_even_while_position_open(monkeypatch) -> None:
    """RSI must keep updating bar-to-bar even while a trade is live, so
    the cross test is accurate the moment the position frees up.  Here a
    stale 99.0 is replaced by the current bar's 42.0 despite the early
    position-open return."""
    params = _short_params()
    _patch_indicators(monkeypatch, m1_rsi_value=42.0)
    strat = MultiTFConfluenceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._open_ticket = 7
    strat._open_side = Side.BUY
    strat._open_bars = 0
    strat._prev_m1_rsi = 99.0                       # stale value
    await strat.on_bar(ctx, _trigger_bar())
    assert captured == []                           # no entry while open
    assert strat._prev_m1_rsi == pytest.approx(42.0)  # tracker advanced
