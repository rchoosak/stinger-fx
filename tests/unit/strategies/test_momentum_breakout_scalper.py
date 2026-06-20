"""Unit tests for MomentumBreakoutScalper.

Same harness as ``test_pullback_reversal_scalper.py`` — monkeypatch the
indicator functions so the *decision logic* is the unit under test, not
the indicator math (which has its own dedicated tests). Each test sets
the indicator outputs it cares about and asserts on the captured
``Signal`` (or its absence).

The bar ``time`` matters here in a way it didn't for PRS: the
session-avoid gate inspects ``bar.time.hour``. Helpers pin the trigger
bar's clock to a configurable UTC hour.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import structlog

import stinger_fx.strategies.examples.momentum_breakout_scalper as mbs_mod
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
from stinger_fx.strategies.examples.momentum_breakout_scalper import (
    MomentumBreakoutScalper,
    MomentumBreakoutScalperParams,
)
from stinger_fx.strategies.indicators.donchian import DonchianChannels

SYMBOL = "XAUUSD"


def _ts(hour: int = 10, minute: int = 0) -> datetime:
    """A UTC timestamp on a fixed Wednesday so the hour we choose
    isn't accidentally during a weekend gap."""
    return datetime(2024, 1, 3, hour, minute, tzinfo=UTC)


def _bar(
    *,
    tf: Timeframe,
    t: datetime,
    close: float = 2340.0,
    high: float | None = None,
    low: float | None = None,
) -> Bar:
    return Bar(
        symbol=SYMBOL,
        timeframe=tf,
        time=t,
        open=close,
        high=high if high is not None else close + 0.5,
        low=low if low is not None else close - 0.5,
        close=close,
        tick_volume=1000,
        is_closed=True,
    )


def _short_params(**overrides: Any) -> MomentumBreakoutScalperParams:
    """Shrunken indicator windows so the fixture doesn't need 60+ bars
    to clear warmup gates."""
    base: dict[str, Any] = dict(
        donchian_period=5,
        require_m5_trend=True,
        ema_fast_period=5,
        ema_slow_period=10,
        avoid_hours_utc=[13, 14, 15, 18],
        vol_gate_lookback=5,
        vol_gate_mult=1.8,
        atr_period=5,
        sl_atr_mult=1.0,
        tp_atr_mult=1.5,
        max_hold_bars_m1=10,
        cooldown_bars_m1=2,
    )
    base.update(overrides)
    return MomentumBreakoutScalperParams(**base)


def _patch_indicators(
    monkeypatch,
    *,
    donchian_upper: float = 2335.0,
    donchian_lower: float = 2325.0,
    atr_value: float = 1.0,
    ema_fast: float = 2335.0,
    ema_slow: float = 2330.0,
    tr_series: list[float] | None = None,
) -> dict[str, list[Any]]:
    """Stub the indicators used by the strategy.

    Returns the tracking dict so a test can inspect how many calls fired
    and with what arguments.
    """
    state: dict[str, list[Any]] = {
        "ema": [ema_fast, ema_slow],
        "calls_ema": [],
        "calls_donchian": [],
        "calls_atr": [],
        "calls_tr": [],
    }

    def _donchian_stub(bars, period):  # type: ignore[no-untyped-def]
        state["calls_donchian"].append((period, len(bars)))
        return DonchianChannels(
            upper=donchian_upper,
            middle=(donchian_upper + donchian_lower) / 2,
            lower=donchian_lower,
        )

    def _atr_stub(bars, period):  # type: ignore[no-untyped-def]
        state["calls_atr"].append((period, len(bars)))
        return atr_value

    def _ema_stub(closes, period):  # type: ignore[no-untyped-def]
        # EMA-fast is requested first, EMA-slow second (see strategy
        # code). Pop in order so multi-call tests stay deterministic.
        state["calls_ema"].append(period)
        idx = (len(state["calls_ema"]) - 1) % 2
        return state["ema"][idx]

    def _tr_stub(bars):  # type: ignore[no-untyped-def]
        state["calls_tr"].append(len(bars))
        # Default: flat series so the latest TR (==1.0) sits at the SMA
        # baseline (==1.0). vol_gate_mult * baseline > tr_now → gate
        # passes. Override via the ``tr_series`` kwarg.
        if tr_series is not None:
            return list(tr_series)
        # Need vol_gate_lookback + 1 values; use enough length for the
        # shrunken params (lookback=5 → 6 values).
        return [1.0] * 20

    def _sma_stub(values, period):  # type: ignore[no-untyped-def]
        # Use the real arithmetic — it's a 3-line function and we want
        # the assertions to test "spike vs baseline" math, not stub
        # equality. period > len(values) → None (mirrors real sma).
        if period <= 0 or len(values) < period:
            return None
        window = values[-period:]
        return sum(window) / period

    monkeypatch.setattr(mbs_mod, "donchian", _donchian_stub)
    monkeypatch.setattr(mbs_mod, "atr", _atr_stub)
    monkeypatch.setattr(mbs_mod, "ema", _ema_stub)
    monkeypatch.setattr(mbs_mod, "sma", _sma_stub)
    monkeypatch.setattr(mbs_mod, "_true_range_series", _tr_stub)
    return state


def _build_ctx(
    *,
    params: MomentumBreakoutScalperParams,
    m1_count: int = 60,
    m5_count: int | None = None,
    positions: list[Position] | None = None,
) -> tuple[StrategyContext, list[Signal], AsyncEventBus]:
    """Build a context with enough M1 + M5 history to clear warmup.

    Indicator math is stubbed so bar close/high/low geometry doesn't
    drive any decision — we just need the lengths to clear the
    ``len(bars) < need`` early-return in ``on_bar``.
    """
    bus = AsyncEventBus()
    captured: list[Signal] = []

    async def sink(sig: Signal) -> None:
        captured.append(sig)

    ctx = StrategyContext(
        strategy_id="mbs_test",
        symbol=params.symbol,
        timeframe=params.entry_timeframe,
        params=params,
        clock=SimClock(_ts(10)),
        logger=structlog.get_logger("mbs_test"),
        magic=99,
        signal_sink=sink,
        subscriptions=MomentumBreakoutScalper.subscriptions(params),
        bus=bus,
    )
    m1_view = ctx.history_for(params.symbol, params.entry_timeframe)
    assert m1_view is not None
    for i in range(m1_count):
        m1_view.append_bar(_bar(tf=Timeframe.M1, t=_ts(10) + timedelta(seconds=i)))

    m5_need = m5_count if m5_count is not None else (
        max(params.ema_slow_period, params.ema_fast_period) + 5
    )
    m5_view = ctx.history_for(params.symbol, params.structure_timeframe)
    assert m5_view is not None
    for i in range(m5_need):
        m5_view.append_bar(
            _bar(tf=Timeframe.M5, t=_ts(10) + timedelta(minutes=5 * i))
        )
    if positions:
        ctx.position.update([
            pos.model_copy(update={"magic": ctx.magic}) for pos in positions
        ])
    return ctx, captured, bus


def _trigger_bar(close: float = 2340.0, hour: int = 10) -> Bar:
    """Trigger bar — only ``close`` and ``time.hour`` matter for the
    indicator-stubbed tests."""
    return _bar(tf=Timeframe.M1, t=_ts(hour, 30), close=close)


# =========================================================================
# Entry tests
# =========================================================================


@pytest.mark.asyncio
async def test_emits_buy_when_breakout_in_m5_uptrend(monkeypatch) -> None:
    """Canonical BUY: bar.close > prior Donchian upper AND M5 EMA-fast
    > EMA-slow AND no avoid-hour AND TR within vol-gate."""
    params = _short_params()
    _patch_indicators(
        monkeypatch,
        donchian_upper=2335.0,
        donchian_lower=2325.0,
        atr_value=2.0,
        ema_fast=2336.0,
        ema_slow=2330.0,
    )
    strat = MomentumBreakoutScalper()
    ctx, captured, _ = _build_ctx(params=params)

    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10))

    assert len(captured) == 1, f"expected 1 signal; got {captured}"
    sig = captured[0]
    assert sig.side is Side.BUY
    assert sig.suggested_volume == params.volume
    # SL = entry - 1.0 * ATR(2.0) = 2338; TP = entry + 1.5 * 2.0 = 2343
    assert sig.suggested_sl == pytest.approx(2338.0)
    assert sig.suggested_tp == pytest.approx(2343.0)


@pytest.mark.asyncio
async def test_emits_sell_when_breakdown_in_m5_downtrend(monkeypatch) -> None:
    """Symmetric SELL: bar.close < prior Donchian lower AND EMA-fast
    < EMA-slow."""
    params = _short_params()
    _patch_indicators(
        monkeypatch,
        donchian_upper=2350.0,
        donchian_lower=2335.0,
        atr_value=2.0,
        ema_fast=2330.0,
        ema_slow=2340.0,    # fast < slow → downtrend
    )
    strat = MomentumBreakoutScalper()
    ctx, captured, _ = _build_ctx(params=params)

    await strat.on_bar(ctx, _trigger_bar(close=2330.0, hour=10))

    assert len(captured) == 1
    sig = captured[0]
    assert sig.side is Side.SELL
    # SL = entry + ATR = 2332; TP = entry - 1.5 * ATR = 2327
    assert sig.suggested_sl == pytest.approx(2332.0)
    assert sig.suggested_tp == pytest.approx(2327.0)


@pytest.mark.asyncio
async def test_no_buy_when_close_does_not_exceed_donchian_upper(monkeypatch) -> None:
    """Price still inside the prior range → no breakout."""
    params = _short_params()
    _patch_indicators(
        monkeypatch,
        donchian_upper=2335.0,
        donchian_lower=2325.0,
        ema_fast=2335.0,
        ema_slow=2330.0,
    )
    strat = MomentumBreakoutScalper()
    ctx, captured, _ = _build_ctx(params=params)

    # close == upper → must be strictly > to fire
    await strat.on_bar(ctx, _trigger_bar(close=2335.0, hour=10))

    assert captured == []


@pytest.mark.asyncio
async def test_session_avoid_blocks_entry(monkeypatch) -> None:
    """avoid_hours_utc default includes 13–15, 18. A bar timestamped
    at 14:30 must be skipped even with all signals aligned."""
    params = _short_params()
    _patch_indicators(monkeypatch, ema_fast=2336.0, ema_slow=2330.0)
    strat = MomentumBreakoutScalper()
    ctx, captured, _ = _build_ctx(params=params)

    # 14:30 UTC → hour 14 is in default avoid list
    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=14))

    assert captured == []


@pytest.mark.asyncio
async def test_empty_avoid_hours_allows_any_hour(monkeypatch) -> None:
    """avoid_hours_utc=[] disables the session filter."""
    params = _short_params(avoid_hours_utc=[])
    _patch_indicators(monkeypatch, ema_fast=2336.0, ema_slow=2330.0)
    strat = MomentumBreakoutScalper()
    ctx, captured, _ = _build_ctx(params=params)

    # 14:30 UTC — would normally be blocked
    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=14))

    assert len(captured) == 1
    assert captured[0].side is Side.BUY


@pytest.mark.asyncio
async def test_vol_gate_blocks_entry_on_spike(monkeypatch) -> None:
    """When the latest TR exceeds vol_gate_mult × SMA(TR, lookback)
    the strategy must skip the entry — that's the spike-candle filter."""
    params = _short_params(vol_gate_mult=1.8, vol_gate_lookback=5)
    # 5 baseline TRs of 1.0 (SMA = 1.0); current TR = 5.0 (>> 1.8).
    _patch_indicators(
        monkeypatch,
        ema_fast=2336.0,
        ema_slow=2330.0,
        tr_series=[1.0, 1.0, 1.0, 1.0, 1.0, 5.0],
    )
    strat = MomentumBreakoutScalper()
    ctx, captured, _ = _build_ctx(params=params)

    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10))

    assert captured == []


@pytest.mark.asyncio
async def test_vol_gate_allows_entry_when_current_tr_under_threshold(monkeypatch) -> None:
    """Current TR is 1.7 × baseline (under the 1.8 cutoff) → entry fires."""
    params = _short_params(vol_gate_mult=1.8, vol_gate_lookback=5)
    _patch_indicators(
        monkeypatch,
        ema_fast=2336.0,
        ema_slow=2330.0,
        tr_series=[1.0, 1.0, 1.0, 1.0, 1.0, 1.7],
    )
    strat = MomentumBreakoutScalper()
    ctx, captured, _ = _build_ctx(params=params)

    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10))

    assert len(captured) == 1


@pytest.mark.asyncio
async def test_no_buy_when_m5_filter_on_and_trend_down(monkeypatch) -> None:
    """require_m5_trend=True (default) — EMA-fast < EMA-slow blocks BUY
    even with a fresh breakout."""
    params = _short_params()
    _patch_indicators(
        monkeypatch,
        donchian_upper=2335.0,
        ema_fast=2330.0, ema_slow=2336.0,   # fast < slow → downtrend
    )
    strat = MomentumBreakoutScalper()
    ctx, captured, _ = _build_ctx(params=params)
    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10))
    assert captured == []


@pytest.mark.asyncio
async def test_m5_filter_off_lets_buy_fire_without_trend(monkeypatch) -> None:
    """require_m5_trend=False → strategy fires on M1 breakout alone."""
    params = _short_params(require_m5_trend=False)
    _patch_indicators(
        monkeypatch,
        donchian_upper=2335.0,
        # EMA stubs won't be consulted; supply anyway for safety.
        ema_fast=2330.0, ema_slow=2336.0,
    )
    strat = MomentumBreakoutScalper()
    ctx, captured, _ = _build_ctx(params=params)
    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10))
    assert len(captured) == 1
    assert captured[0].side is Side.BUY


@pytest.mark.asyncio
async def test_tp_disabled_when_mult_is_zero(monkeypatch) -> None:
    """tp_atr_mult=0 → no TP on the order (rely on SL / time stop /
    trailing)."""
    params = _short_params(tp_atr_mult=0.0)
    _patch_indicators(monkeypatch, ema_fast=2336.0, ema_slow=2330.0)
    strat = MomentumBreakoutScalper()
    ctx, captured, _ = _build_ctx(params=params)
    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10))
    assert len(captured) == 1
    assert captured[0].suggested_tp is None


# =========================================================================
# Exit tests
# =========================================================================


@pytest.mark.asyncio
async def test_time_stop_closes_position_after_max_hold(monkeypatch) -> None:
    """The strategy delegates SL/TP to the broker but still enforces a
    time stop so the runner can't camp on a stale position."""
    params = _short_params(max_hold_bars_m1=3)
    _patch_indicators(monkeypatch)
    strat = MomentumBreakoutScalper()
    ctx, _captured, bus = _build_ctx(params=params)
    strat._open_ticket = 42
    strat._open_side = Side.BUY
    strat._open_bars = 2   # this on_bar tick → 3 → triggers

    close_events: list[Any] = []
    from stinger_fx.core.events import ClosePositionRequestEvent
    bus.subscribe(
        ClosePositionRequestEvent,
        lambda e: close_events.append(e),
        name="probe.close",
    )

    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10))
    # Drain the bus.
    import asyncio
    for _ in range(3):
        await asyncio.sleep(0)

    assert len(close_events) == 1
    assert close_events[0].ticket == 42


@pytest.mark.asyncio
async def test_time_stop_does_not_fire_before_max_hold(monkeypatch) -> None:
    """Time stop must wait until the bar count actually crosses the
    threshold — no premature exit."""
    params = _short_params(max_hold_bars_m1=10)
    _patch_indicators(monkeypatch)
    strat = MomentumBreakoutScalper()
    ctx, _captured, bus = _build_ctx(params=params)
    strat._open_ticket = 7
    strat._open_side = Side.BUY
    strat._open_bars = 3

    close_events: list[Any] = []
    from stinger_fx.core.events import ClosePositionRequestEvent
    bus.subscribe(
        ClosePositionRequestEvent,
        lambda e: close_events.append(e),
        name="probe.close",
    )

    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10))
    import asyncio
    for _ in range(3):
        await asyncio.sleep(0)
    assert close_events == []
    # bar counter advanced
    assert strat._open_bars == 4


# =========================================================================
# Position-tracking lifecycle
# =========================================================================


@pytest.mark.asyncio
async def test_on_order_filled_records_open_ticket() -> None:
    strat = MomentumBreakoutScalper()
    params = _short_params()
    ctx, _captured, _ = _build_ctx(params=params)
    order = Order(
        ticket=101,
        strategy_id=ctx.strategy_id,
        symbol=SYMBOL,
        side=Side.BUY,
        type=OrderType.MARKET,
        volume=0.01,
        status=OrderStatus.FILLED,
        filled_volume=0.01,
        fill_price=2340.0,
    )
    await strat.on_order_filled(ctx, order)
    assert strat._open_ticket == 101
    assert strat._open_side is Side.BUY
    assert strat._open_bars == 0


@pytest.mark.asyncio
async def test_on_order_filled_ignores_other_strategies_orders() -> None:
    strat = MomentumBreakoutScalper()
    params = _short_params()
    ctx, _captured, _ = _build_ctx(params=params)
    order = Order(
        ticket=99,
        strategy_id="other_strategy",
        symbol=SYMBOL,
        side=Side.BUY,
        type=OrderType.MARKET,
        volume=0.01,
        status=OrderStatus.FILLED,
        filled_volume=0.01,
        fill_price=2340.0,
    )
    await strat.on_order_filled(ctx, order)
    assert strat._open_ticket is None


@pytest.mark.asyncio
async def test_on_position_closed_starts_cooldown() -> None:
    strat = MomentumBreakoutScalper()
    params = _short_params(cooldown_bars_m1=4)
    ctx, _captured, _ = _build_ctx(params=params)
    strat._open_ticket = 55
    strat._open_side = Side.BUY
    strat._open_bars = 7
    pos = Position(
        ticket=55,
        symbol=SYMBOL,
        side=Side.BUY,
        volume=0.01,
        open_price=2340.0,
        open_time=_ts(10),
    )
    await strat.on_position_closed(ctx, pos)
    assert strat._open_ticket is None
    assert strat._open_side is None
    assert strat._open_bars == 0
    assert strat._cooldown_left == 4


@pytest.mark.asyncio
async def test_cooldown_blocks_immediate_re_entry(monkeypatch) -> None:
    """After a close, the strategy must wait ``cooldown_bars_m1`` bars
    before sending another signal even when conditions are perfect."""
    params = _short_params(cooldown_bars_m1=2)
    _patch_indicators(monkeypatch, ema_fast=2336.0, ema_slow=2330.0)
    strat = MomentumBreakoutScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._cooldown_left = 2

    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10))
    assert captured == []
    assert strat._cooldown_left == 1
    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10))
    assert captured == []
    assert strat._cooldown_left == 0
    # Third bar — cooldown elapsed → entry now allowed.
    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10))
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_only_one_open_position_at_a_time(monkeypatch) -> None:
    """While a position is open, no new entry signal can fire."""
    params = _short_params()
    _patch_indicators(monkeypatch, ema_fast=2336.0, ema_slow=2330.0)
    strat = MomentumBreakoutScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._open_ticket = 999
    strat._open_side = Side.BUY
    strat._open_bars = 1

    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10))
    assert captured == [], "no new signal while a position is open"


# =========================================================================
# Trailing stop loss (toggle)
# =========================================================================


@pytest.mark.asyncio
async def test_trailing_manager_attached_when_flag_on(monkeypatch) -> None:
    """enable_trailing=True → after BUY emission, a TrailingStopManager
    is appended to ``ctx._managers``."""
    from stinger_fx.strategies.managers.trailing import TrailingStopManager

    params = _short_params(
        enable_trailing=True,
        trailing_distance_atr=0.8,
        trailing_activate_atr=0.5,
        trailing_point=0.01,
    )
    _patch_indicators(
        monkeypatch,
        ema_fast=2336.0, ema_slow=2330.0,
        atr_value=2.0,
    )
    strat = MomentumBreakoutScalper()
    ctx, captured, _ = _build_ctx(params=params)
    before = len(ctx._managers)

    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10))

    assert len(captured) == 1
    assert len(ctx._managers) == before + 1
    mgr = ctx._managers[-1]
    assert isinstance(mgr, TrailingStopManager)
    # distance = 0.8 ATR / point × point = 0.8 × 2.0 = 1.6
    assert mgr._distance == pytest.approx(1.6)
    assert mgr._activate == pytest.approx(1.0)   # 0.5 × 2.0
    assert mgr._symbol == params.symbol


@pytest.mark.asyncio
async def test_no_trailing_manager_when_flag_off(monkeypatch) -> None:
    params = _short_params(enable_trailing=False)
    _patch_indicators(monkeypatch, ema_fast=2336.0, ema_slow=2330.0)
    strat = MomentumBreakoutScalper()
    ctx, captured, _ = _build_ctx(params=params)
    before = len(ctx._managers)

    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10))

    assert len(captured) == 1
    assert len(ctx._managers) == before
