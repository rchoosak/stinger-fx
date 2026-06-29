"""Unit tests for PullbackReversalScalper.

The strategy logic depends on six indicator readings per bar
(EMA fast, EMA slow, M5 RSI, M1 RSI, Stoch RSI K/D, ATR). Crafting
real bar series to drive all six to specific values is brittle and
distracts from what we actually want to verify — the *decision logic*,
not the indicator math. We test indicators separately in
``tests/unit/strategies/test_indicators*.py``.

So these tests **monkeypatch the four indicator functions** inside the
strategy module to deterministic stubs. Each test sets the values it
needs and asserts on the signal that comes out (or doesn't).

Cross-detection state (``_prev_k`` / ``_prev_d``) is set directly on
the strategy instance for entry tests — that lets each test exercise
a single bar instead of running two-bar fixtures everywhere.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import structlog

import stinger_fx.strategies.examples.pullback_reversal_scalper as prs_mod
from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.domain import Bar, Order, OrderStatus, OrderType, Position, Side, Signal, Timeframe
from stinger_fx.strategies.context import HistoryView, StrategyContext
from stinger_fx.strategies.examples.pullback_reversal_scalper import (
    PullbackReversalScalper,
    PullbackReversalScalperParams,
)

SYMBOL = "XAUUSD"


def _ts(seconds: int = 0) -> datetime:
    return datetime(2024, 1, 1, 12, 0, tzinfo=UTC) + timedelta(seconds=seconds)


def _bar(*, tf: Timeframe, t: datetime, close: float = 2340.0, high: float | None = None,
         low: float | None = None) -> Bar:
    return Bar(
        symbol=SYMBOL, timeframe=tf, time=t,
        open=close, high=high if high is not None else close + 0.5,
        low=low if low is not None else close - 0.5, close=close,
        tick_volume=1000, is_closed=True,
    )


def _short_params(**overrides: Any) -> PullbackReversalScalperParams:
    base: dict[str, Any] = dict(
        ema_fast_period=5,
        ema_slow_period=10,
        m5_rsi_period=5,
        m1_rsi_period=5,
        stoch_rsi_period=5,
        stoch_rsi_k_smooth=2,
        stoch_rsi_d_smooth=2,
        atr_period=5,
        swing_lookback=3,
        sl_atr_mult=1.0,
        tp_atr_mult=0.0,
        max_hold_bars_m1=10,
        cooldown_bars_m1=2,
    )
    base.update(overrides)
    return PullbackReversalScalperParams(**base)


def _patch_indicators(
    monkeypatch,
    *,
    m5_rsi: float = 45.0,
    m1_rsi: float = 15.0,
    ema_fast: float = 2335.0,
    ema_slow: float = 2330.0,
    stoch_k: float | list[float] = 18.0,
    stoch_d: float | list[float] = 12.0,
    atr_value: float = 1.0,
) -> dict[str, list[Any]]:
    """Stub the 4 indicator functions in the strategy module.

    ``stoch_k`` / ``stoch_d`` accept either a constant (returned every
    call) or a list — list values are popped from the front on each
    call so the same test can drive entry then exit through one stub.

    EMA calls are ordered (fast first, slow second) — same for RSI
    (M5 first, M1 second). The strategy code calls them in that exact
    order inside ``on_bar``.
    """
    state: dict[str, list[Any]] = {
        "ema": [ema_fast, ema_slow],
        "rsi": [m5_rsi, m1_rsi],
        "stoch_k": [stoch_k] if isinstance(stoch_k, float | int) else list(stoch_k),
        "stoch_d": [stoch_d] if isinstance(stoch_d, float | int) else list(stoch_d),
        "calls_ema": [],
        "calls_rsi": [],
        "calls_stoch": [],
    }

    def _ema_stub(closes, period):  # type: ignore[no-untyped-def]
        state["calls_ema"].append(period)
        idx = (len(state["calls_ema"]) - 1) % 2
        return state["ema"][idx]

    # M5 RSI is still a module call (the M5 trend filter); M1 RSI / Stoch RSI /
    # ATR now stream off the HistoryView, so they're stubbed on the *view*
    # methods below. The module rsi stub therefore only ever serves M5.
    def _rsi_stub(closes, period):  # type: ignore[no-untyped-def]
        state["calls_rsi"].append((period, len(closes)))
        return state["rsi"][0]

    def _hv_rsi(self, period=14):  # type: ignore[no-untyped-def]
        return state["rsi"][1]

    def _hv_stoch(self, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):  # type: ignore[no-untyped-def]
        state["calls_stoch"].append(rsi_period)
        k = state["stoch_k"].pop(0) if len(state["stoch_k"]) > 1 else state["stoch_k"][0]
        d = state["stoch_d"].pop(0) if len(state["stoch_d"]) > 1 else state["stoch_d"][0]
        return (float(k), float(d))

    def _hv_atr(self, period=14):  # type: ignore[no-untyped-def]
        return atr_value

    monkeypatch.setattr(prs_mod, "ema", _ema_stub)
    monkeypatch.setattr(prs_mod, "rsi", _rsi_stub)
    monkeypatch.setattr(HistoryView, "rsi", _hv_rsi)
    monkeypatch.setattr(HistoryView, "stoch_rsi", _hv_stoch)
    monkeypatch.setattr(HistoryView, "atr", _hv_atr)
    return state


def _build_ctx(
    *, params: PullbackReversalScalperParams,
    m1_count: int = 60,
    m5_uptrend: bool = True,
    positions: list[Position] | None = None,
) -> tuple[StrategyContext, list[Signal], AsyncEventBus]:
    """Build a context with hand-shaped M1 + M5 histories.

    Enough bars to clear all warmup gates with the SHRUNK params from
    ``_short_params`` — actual indicator values come from the patched
    stubs, the bars only need to satisfy "have at least N entries".

    ``m5_uptrend`` controls the M5 HH/HL geometry:
      * True  → strictly rising highs + rising lows
      * False → strictly falling highs + falling lows
    """
    bus = AsyncEventBus()
    captured: list[Signal] = []

    async def sink(sig: Signal) -> None:
        captured.append(sig)

    ctx = StrategyContext(
        strategy_id="prs_test",
        symbol=params.symbol,
        timeframe=params.entry_timeframe,
        params=params,
        clock=SimClock(_ts(0)),
        logger=structlog.get_logger("prs_test"),
        magic=99,
        signal_sink=sink,
        subscriptions=PullbackReversalScalper.subscriptions(params),
        bus=bus,
    )
    # Fill M1 history with enough bars (close geometry irrelevant — patched stubs).
    m1_view = ctx.history_for(params.symbol, params.entry_timeframe)
    assert m1_view is not None
    for i in range(m1_count):
        m1_view.append_bar(_bar(tf=Timeframe.M1, t=_ts(i)))
    # M5 history — shape highs/lows so HH/HL check returns the desired bool.
    m5_view = ctx.history_for(params.symbol, params.structure_timeframe)
    assert m5_view is not None
    m5_count = max(params.ema_slow_period, params.m5_rsi_period,
                   params.swing_lookback + 1) + 5
    for i in range(m5_count):
        # Each new bar's high/low is higher (or lower) than the previous one.
        delta = i * 0.5 if m5_uptrend else -i * 0.5
        close = 2340.0 + delta
        m5_view.append_bar(
            _bar(tf=Timeframe.M5, t=_ts(60 * i), close=close,
                 high=close + 0.5, low=close - 0.5),
        )
    if positions:
        ctx.position.update([
            pos.model_copy(update={"magic": ctx.magic}) for pos in positions
        ])
    return ctx, captured, bus


def _trigger_bar(close: float = 2336.0) -> Bar:
    """A single M1 bar to drive on_bar. Time/geometry don't drive any
    decision because we patch indicators — only `close` matters (used
    as the entry price for SL/TP calculation)."""
    return _bar(tf=Timeframe.M1, t=_ts(120), close=close)


# =========================================================================
# Entry tests
# =========================================================================


@pytest.mark.asyncio
async def test_emits_buy_when_all_conditions_met(monkeypatch) -> None:
    """The canonical happy path: M5 uptrend + M5 RSI < 50 + M1 oversold
    + Stoch K cross up D in the OS zone."""
    params = _short_params()
    _patch_indicators(
        monkeypatch,
        m5_rsi=45.0,         # below buy max (50) → pullback in uptrend
        m1_rsi=15.0,         # below 20 → M1 OS
        ema_fast=2335.0,
        ema_slow=2330.0,     # fast > slow → uptrend OK
        stoch_k=18.0,        # K < 20 (OS zone)
        stoch_d=12.0,        # current K=18 > D=12 (after the cross)
    )
    strat = PullbackReversalScalper()
    ctx, captured, _bus = _build_ctx(params=params, m5_uptrend=True)
    # Seed prev so the cross detector sees K rising through D.
    strat._prev_k = 10.0
    strat._prev_d = 13.0

    await strat.on_bar(ctx, _trigger_bar(close=2336.0))

    assert len(captured) == 1, f"expected 1 signal; got {[s.side for s in captured]}"
    sig = captured[0]
    assert sig.side is Side.BUY
    assert sig.suggested_volume == params.volume
    # SL = entry - 1.0 * ATR(1.0) = 2335
    assert sig.suggested_sl == pytest.approx(2335.0)
    assert sig.suggested_tp is None  # tp_atr_mult=0


@pytest.mark.asyncio
async def test_emits_sell_when_all_conditions_met(monkeypatch) -> None:
    """Symmetric SELL case.

    In the downtrend fixture the latest M5 close ends up near 2333
    (2340 - 14 * 0.5). For "close < EMA50" to hold, the slow stub
    must sit ABOVE that value — use 2345.
    """
    params = _short_params()
    _patch_indicators(
        monkeypatch,
        m5_rsi=55.0,         # above sell min (50) → bounce in downtrend
        m1_rsi=85.0,         # > 80 → M1 OB
        ema_fast=2335.0,
        ema_slow=2345.0,     # fast < slow AND close (~2333) < slow → downtrend
        stoch_k=82.0,        # OB zone
        stoch_d=88.0,        # K=82 < D=88 (after cross down)
    )
    strat = PullbackReversalScalper()
    ctx, captured, _bus = _build_ctx(params=params, m5_uptrend=False)
    strat._prev_k = 90.0
    strat._prev_d = 87.0   # K was above D → cross down

    await strat.on_bar(ctx, _trigger_bar(close=2344.0))

    assert len(captured) == 1
    sig = captured[0]
    assert sig.side is Side.SELL
    assert sig.suggested_sl == pytest.approx(2345.0)  # entry + ATR


@pytest.mark.asyncio
async def test_no_buy_when_m5_rsi_above_threshold(monkeypatch) -> None:
    """M5 uptrend is in place, but RSI hasn't pulled back below 50 —
    no entry (we don't chase strength). Requires ``use_m5_filter=True``
    since the filter is OFF by default."""
    params = _short_params(use_m5_filter=True)
    _patch_indicators(monkeypatch, m5_rsi=60.0)   # too strong
    strat = PullbackReversalScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._prev_k, strat._prev_d = 10.0, 13.0
    await strat.on_bar(ctx, _trigger_bar())
    assert captured == []


@pytest.mark.asyncio
async def test_no_buy_when_m5_structure_is_downtrend(monkeypatch) -> None:
    """Even with everything else BUY-ish, M5 not in uptrend → no trade
    when the filter is ON."""
    params = _short_params(use_m5_filter=True)
    _patch_indicators(
        monkeypatch,
        m5_rsi=45.0, m1_rsi=15.0,
        ema_fast=2325.0, ema_slow=2330.0,    # fast < slow = downtrend
        stoch_k=18.0, stoch_d=12.0,
    )
    strat = PullbackReversalScalper()
    ctx, captured, _ = _build_ctx(params=params, m5_uptrend=False)
    strat._prev_k, strat._prev_d = 10.0, 13.0
    await strat.on_bar(ctx, _trigger_bar())
    assert captured == []


@pytest.mark.asyncio
async def test_buy_fires_when_m5_filter_off_even_with_m5_downtrend(monkeypatch) -> None:
    """With ``use_m5_filter=False`` (the new default), the strategy
    ignores M5 entirely. Even when M5 looks like a downtrend AND M5
    RSI is bullish (both of which would BLOCK a BUY under the filter),
    a BUY fires as long as M1 conditions are met."""
    params = _short_params(use_m5_filter=False)
    _patch_indicators(
        monkeypatch,
        m5_rsi=80.0,                         # strongly bullish — would block BUY
        m1_rsi=15.0,                         # M1 OS
        ema_fast=2325.0, ema_slow=2330.0,    # fast < slow → would block BUY
        stoch_k=18.0, stoch_d=12.0,
    )
    strat = PullbackReversalScalper()
    ctx, captured, _ = _build_ctx(params=params, m5_uptrend=False)
    strat._prev_k, strat._prev_d = 10.0, 13.0
    await strat.on_bar(ctx, _trigger_bar())
    assert len(captured) == 1
    assert captured[0].side is Side.BUY


@pytest.mark.asyncio
async def test_sell_fires_when_m5_filter_off_even_with_m5_uptrend(monkeypatch) -> None:
    """Symmetric: with the filter off, SELL ignores M5 entirely."""
    params = _short_params(use_m5_filter=False)
    _patch_indicators(
        monkeypatch,
        m5_rsi=20.0,                         # bearish — would block SELL
        m1_rsi=85.0,                         # M1 OB
        ema_fast=2335.0, ema_slow=2330.0,    # fast > slow → would block SELL
        stoch_k=82.0, stoch_d=88.0,
    )
    strat = PullbackReversalScalper()
    ctx, captured, _ = _build_ctx(params=params, m5_uptrend=True)
    strat._prev_k, strat._prev_d = 90.0, 87.0
    await strat.on_bar(ctx, _trigger_bar(close=2344.0))
    assert len(captured) == 1
    assert captured[0].side is Side.SELL


@pytest.mark.asyncio
async def test_no_buy_when_m1_rsi_not_oversold(monkeypatch) -> None:
    params = _short_params()
    _patch_indicators(monkeypatch, m1_rsi=40.0)   # not OS
    strat = PullbackReversalScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._prev_k, strat._prev_d = 10.0, 13.0
    await strat.on_bar(ctx, _trigger_bar())
    assert captured == []


@pytest.mark.asyncio
async def test_no_buy_when_stoch_cross_above_oversold_zone(monkeypatch) -> None:
    """Stoch K crosses up but K=25 — above the 20 OS gate. Must skip."""
    params = _short_params()
    _patch_indicators(monkeypatch, stoch_k=25.0, stoch_d=20.0)
    strat = PullbackReversalScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._prev_k, strat._prev_d = 18.0, 22.0  # crossed up but K already > 20
    await strat.on_bar(ctx, _trigger_bar())
    assert captured == []


@pytest.mark.asyncio
async def test_no_buy_when_no_stoch_cross(monkeypatch) -> None:
    """Stoch K is in OS, but no cross happened — current K still below D."""
    params = _short_params()
    _patch_indicators(monkeypatch, stoch_k=15.0, stoch_d=18.0)
    strat = PullbackReversalScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._prev_k, strat._prev_d = 12.0, 17.0  # still below D, no cross
    await strat.on_bar(ctx, _trigger_bar())
    assert captured == []


# =========================================================================
# Exit tests
# =========================================================================


@pytest.mark.asyncio
async def test_closes_buy_when_stoch_k_exceeds_exit_long(monkeypatch) -> None:
    """While a BUY is open, when M1 Stoch K crosses above 80 we send
    a close request through the bus."""
    params = _short_params()
    _patch_indicators(monkeypatch, stoch_k=85.0, stoch_d=70.0)
    strat = PullbackReversalScalper()
    ctx, _captured, bus = _build_ctx(params=params)

    # Simulate an open BUY.
    strat._open_ticket = 42
    strat._open_side = Side.BUY
    strat._open_bars = 5

    close_events: list[Any] = []
    from stinger_fx.core.events import ClosePositionRequestEvent
    bus.subscribe(
        ClosePositionRequestEvent,
        lambda e: close_events.append(e),
        name="probe.close",
    )

    await strat.on_bar(ctx, _trigger_bar())
    # Drain bus
    import asyncio
    for _ in range(3):
        await asyncio.sleep(0)

    assert len(close_events) == 1
    assert close_events[0].ticket == 42


@pytest.mark.asyncio
async def test_closes_sell_when_stoch_k_below_exit_short(monkeypatch) -> None:
    params = _short_params()
    _patch_indicators(monkeypatch, stoch_k=15.0, stoch_d=30.0)
    strat = PullbackReversalScalper()
    ctx, _captured, bus = _build_ctx(params=params)
    strat._open_ticket = 7
    strat._open_side = Side.SELL
    strat._open_bars = 3

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
    assert close_events[0].ticket == 7


@pytest.mark.asyncio
async def test_max_hold_bars_force_exit(monkeypatch) -> None:
    """Stoch never reaches the exit, but max_hold_bars fires the time-stop."""
    params = _short_params(max_hold_bars_m1=3)
    _patch_indicators(monkeypatch, stoch_k=50.0, stoch_d=50.0)  # mid-zone, no Stoch exit
    strat = PullbackReversalScalper()
    ctx, _captured, bus = _build_ctx(params=params)
    strat._open_ticket = 13
    strat._open_side = Side.BUY
    strat._open_bars = 2   # this on_bar will tick to 3 → triggers

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


# =========================================================================
# Position-tracking lifecycle
# =========================================================================


@pytest.mark.asyncio
async def test_on_order_filled_records_open_ticket() -> None:
    """When our own strategy's order fills, the strategy must record
    the ticket + side so the next exit check can find it."""
    strat = PullbackReversalScalper()
    params = _short_params()
    ctx, _captured, _ = _build_ctx(params=params)
    order = Order(
        ticket=101,
        strategy_id=ctx.strategy_id,
        symbol=SYMBOL, side=Side.BUY,
        type=OrderType.MARKET, volume=0.01,
        status=OrderStatus.FILLED,
        filled_volume=0.01, fill_price=2340.0,
    )
    await strat.on_order_filled(ctx, order)
    assert strat._open_ticket == 101
    assert strat._open_side is Side.BUY
    assert strat._open_bars == 0


@pytest.mark.asyncio
async def test_on_order_filled_ignores_other_strategies_orders() -> None:
    """Multi-strategy live setups share a broker — an order filled by
    a different strategy must not steal our slot."""
    strat = PullbackReversalScalper()
    params = _short_params()
    ctx, _captured, _ = _build_ctx(params=params)
    order = Order(
        ticket=99, strategy_id="other_strategy",
        symbol=SYMBOL, side=Side.BUY,
        type=OrderType.MARKET, volume=0.01,
        status=OrderStatus.FILLED,
        filled_volume=0.01, fill_price=2340.0,
    )
    await strat.on_order_filled(ctx, order)
    assert strat._open_ticket is None


@pytest.mark.asyncio
async def test_on_position_closed_starts_cooldown() -> None:
    """Closing a position must reset state + start the cooldown counter
    so we don't immediately re-enter on the next bar."""
    strat = PullbackReversalScalper()
    params = _short_params(cooldown_bars_m1=4)
    ctx, _captured, _ = _build_ctx(params=params)
    strat._open_ticket = 55
    strat._open_side = Side.BUY
    strat._open_bars = 7
    pos = Position(
        ticket=55, symbol=SYMBOL, side=Side.BUY,
        volume=0.01, open_price=2340.0, open_time=_ts(),
    )
    await strat.on_position_closed(ctx, pos)
    assert strat._open_ticket is None
    assert strat._open_side is None
    assert strat._open_bars == 0
    assert strat._cooldown_left == 4


@pytest.mark.asyncio
async def test_cooldown_blocks_immediate_re_entry(monkeypatch) -> None:
    """After a close, even with all BUY conditions met, the strategy
    must wait ``cooldown_bars_m1`` bars before sending another signal."""
    params = _short_params(cooldown_bars_m1=2)
    _patch_indicators(monkeypatch)
    strat = PullbackReversalScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._prev_k, strat._prev_d = 10.0, 13.0
    strat._cooldown_left = 2

    await strat.on_bar(ctx, _trigger_bar())
    assert captured == []
    assert strat._cooldown_left == 1
    await strat.on_bar(ctx, _trigger_bar())
    assert captured == []
    assert strat._cooldown_left == 0
    # 3rd bar — cooldown elapsed; but prev_k/prev_d weren't tracked while in
    # cooldown, so we need to re-seed.
    strat._prev_k, strat._prev_d = 10.0, 13.0
    await strat.on_bar(ctx, _trigger_bar())
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_only_one_open_position_at_a_time(monkeypatch) -> None:
    """While a position is open, no new entry signal can fire even if
    the conditions are perfect for one."""
    params = _short_params()
    _patch_indicators(monkeypatch, stoch_k=50.0, stoch_d=50.0)  # also no exit
    strat = PullbackReversalScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._open_ticket = 999
    strat._open_side = Side.BUY
    strat._open_bars = 1
    strat._prev_k, strat._prev_d = 10.0, 13.0

    await strat.on_bar(ctx, _trigger_bar())
    assert captured == [], "no new signal while a position is open"


# =========================================================================
# Trailing stop loss (toggle)
# =========================================================================


@pytest.mark.asyncio
async def test_trailing_manager_attached_when_flag_on(monkeypatch) -> None:
    """enable_trailing=True → after BUY emission, a TrailingStopManager
    is appended to ``ctx._managers`` with distance + activation derived
    from entry-time ATR."""
    from stinger_fx.strategies.managers.trailing import TrailingStopManager

    params = _short_params(
        enable_trailing=True,
        trailing_distance_atr=0.8,
        trailing_activate_atr=0.5,
        trailing_point=0.01,
    )
    _patch_indicators(monkeypatch, atr_value=2.0)
    strat = PullbackReversalScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._prev_k, strat._prev_d = 10.0, 13.0
    before = len(ctx._managers)

    await strat.on_bar(ctx, _trigger_bar())

    assert len(captured) == 1                       # BUY still emitted
    assert len(ctx._managers) == before + 1
    mgr = ctx._managers[-1]
    assert isinstance(mgr, TrailingStopManager)
    # `mgr._distance` is stored as price units = pips × point.
    # = (0.8 * 2.0 / 0.01) × 0.01 = 0.8 * 2.0 = 1.6
    assert mgr._distance == pytest.approx(1.6)
    assert mgr._activate == pytest.approx(1.0)      # 0.5 * 2.0
    assert mgr._symbol == params.symbol


@pytest.mark.asyncio
async def test_no_trailing_manager_when_flag_off(monkeypatch) -> None:
    """enable_trailing=False (default) → no manager attached, behavior
    matches the pre-trailing baseline. Verifies the flag actually gates
    the attach."""
    params = _short_params(enable_trailing=False)
    _patch_indicators(monkeypatch)
    strat = PullbackReversalScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._prev_k, strat._prev_d = 10.0, 13.0
    before = len(ctx._managers)

    await strat.on_bar(ctx, _trigger_bar())

    assert len(captured) == 1                       # BUY emitted
    assert len(ctx._managers) == before             # but no manager added


@pytest.mark.asyncio
async def test_trailing_manager_attached_on_sell_entry(monkeypatch) -> None:
    """The SELL branch must also attach a trailing manager when the
    flag is on — symmetric with the BUY path."""
    from stinger_fx.strategies.managers.trailing import TrailingStopManager

    params = _short_params(enable_trailing=True)
    _patch_indicators(
        monkeypatch,
        m5_rsi=55.0, m1_rsi=85.0,
        ema_fast=2335.0, ema_slow=2345.0,
        stoch_k=82.0, stoch_d=88.0,
        atr_value=1.5,
    )
    strat = PullbackReversalScalper()
    ctx, captured, _ = _build_ctx(params=params, m5_uptrend=False)
    strat._prev_k, strat._prev_d = 90.0, 87.0
    before = len(ctx._managers)

    await strat.on_bar(ctx, _trigger_bar(close=2344.0))

    assert len(captured) == 1
    assert captured[0].side is Side.SELL
    assert len(ctx._managers) == before + 1
    assert isinstance(ctx._managers[-1], TrailingStopManager)


# =========================================================================
# Higher-timeframe trend filter
# =========================================================================


def _m1(close: float, t: datetime, high: float | None = None,
        low: float | None = None, open_: float | None = None) -> Bar:
    return Bar(
        symbol=SYMBOL, timeframe=Timeframe.M1, time=t,
        open=open_ if open_ is not None else close,
        high=high if high is not None else close,
        low=low if low is not None else close,
        close=close, tick_volume=10, is_closed=True,
    )


def _fill_trend(strat: PullbackReversalScalper, closes: list[float]) -> None:
    """Push a list of *completed* higher-TF closes into the fold deque
    (bypassing _fold_trend) so _trend_ok can be tested directly."""
    for c in closes:
        strat._trend_bars.append(_m1(c, _ts(0)))


def test_trend_ok_off_by_default_is_always_true() -> None:
    """Filter off (default) → no fold history needed, every side allowed."""
    strat = PullbackReversalScalper()
    params = _short_params()  # trend_filter_timeframe=None
    assert strat._trend_ok(Side.BUY, params) is True
    assert strat._trend_ok(Side.SELL, params) is True


def test_trend_ok_blocks_against_trend() -> None:
    """With the filter on, only the side aligned with the HTF EMA is allowed:
    a rising series (close above EMA) permits BUY and blocks SELL."""
    params = _short_params(
        trend_filter_timeframe=Timeframe.H1, trend_ema_period=5, trend_adx_period=3
    )
    strat = PullbackReversalScalper()
    _fill_trend(strat, [float(x) for x in range(100, 130)])  # steadily rising
    assert strat._trend_ok(Side.BUY, params) is True
    assert strat._trend_ok(Side.SELL, params) is False

    strat2 = PullbackReversalScalper()
    _fill_trend(strat2, [float(x) for x in range(130, 100, -1)])  # falling
    assert strat2._trend_ok(Side.SELL, params) is True
    assert strat2._trend_ok(Side.BUY, params) is False


def test_trend_ok_not_warm_returns_false() -> None:
    """Filter on but too few completed buckets → hold fire (False), never
    trade on an unwarmed trend read."""
    params = _short_params(
        trend_filter_timeframe=Timeframe.H1, trend_ema_period=5, trend_adx_period=3
    )
    strat = PullbackReversalScalper()
    _fill_trend(strat, [100.0, 101.0, 102.0])  # < need (max(6, 6))
    assert strat._trend_ok(Side.BUY, params) is False
    assert strat._trend_ok(Side.SELL, params) is False


def test_trend_ok_adx_band_gates(monkeypatch) -> None:
    """When the ADX band is non-trivial, a HTF ADX outside [min, max] blocks
    even a correctly-aligned trade; the default 0..100 band is a no-op."""
    from stinger_fx.strategies.indicators.adx import ADXResult

    monkeypatch.setattr(prs_mod, "adx", lambda bars, period: ADXResult(
        adx=40.0, plus_di=30.0, minus_di=10.0))
    strat = PullbackReversalScalper()
    _fill_trend(strat, [float(x) for x in range(100, 130)])  # rising → BUY aligned

    # ADX 40 is above max 35 → blocked despite alignment.
    p_band = _short_params(trend_filter_timeframe=Timeframe.H1, trend_ema_period=5,
                           trend_adx_period=3, trend_adx_min=18.0, trend_adx_max=35.0)
    assert strat._trend_ok(Side.BUY, p_band) is False

    # Same series, band wide open (default) → ADX never consulted, allowed.
    p_wide = _short_params(trend_filter_timeframe=Timeframe.H1, trend_ema_period=5,
                           trend_adx_period=3)
    assert strat._trend_ok(Side.BUY, p_wide) is True


def test_fold_trend_emit_on_next_bucket() -> None:
    """The fold emits a higher-TF bucket only once the *next* bucket opens,
    aggregates OHLC correctly, and never exposes the in-progress bucket."""
    params = _short_params(trend_filter_timeframe=Timeframe.H1)
    strat = PullbackReversalScalper()
    base = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)

    def at(minutes: int) -> datetime:
        return base + timedelta(minutes=minutes)

    # Hour 12 bucket: 4 bars. open=100 (first), high=115, low=90, close=105.
    strat._fold_trend(_m1(100.0, at(0), high=100.0, low=100.0, open_=100.0), params)
    strat._fold_trend(_m1(110.0, at(15), high=115.0, low=108.0), params)
    strat._fold_trend(_m1(95.0, at(30), high=96.0, low=90.0), params)
    strat._fold_trend(_m1(105.0, at(45), high=106.0, low=104.0), params)
    # Still in-progress (hour 12 not closed): nothing emitted yet.
    assert len(strat._trend_bars) == 0

    # Hour 13 opens → hour 12 finalised + emitted.
    strat._fold_trend(_m1(120.0, at(60), high=121.0, low=119.0), params)
    assert len(strat._trend_bars) == 1
    b12 = strat._trend_bars[-1]
    assert b12.open == 100.0 and b12.high == 115.0
    assert b12.low == 90.0 and b12.close == 105.0
    assert b12.timeframe is Timeframe.H1

    # Hour 14 opens → hour 13 emitted; hour 14 stays in-progress (not in deque).
    strat._fold_trend(_m1(130.0, at(120), high=131.0, low=129.0), params)
    assert len(strat._trend_bars) == 2
    assert strat._trend_bars[-1].close == 120.0  # hour 13's close


@pytest.mark.asyncio
async def test_on_bar_trend_filter_blocks_entry(monkeypatch) -> None:
    """on_bar consults _trend_ok: when it vetoes the side, a BUY that meets
    every M1 condition is suppressed."""
    params = _short_params(trend_filter_timeframe=Timeframe.H1, trend_ema_period=5)
    _patch_indicators(monkeypatch)  # M1 conditions all BUY-ready
    strat = PullbackReversalScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._prev_k, strat._prev_d = 10.0, 13.0
    # Force the trend gate to veto regardless of fold state.
    strat._trend_ok = lambda side, p: False  # type: ignore[method-assign]
    await strat.on_bar(ctx, _trigger_bar())
    assert captured == []


@pytest.mark.asyncio
async def test_on_bar_trend_filter_allows_entry(monkeypatch) -> None:
    """Mirror: when _trend_ok permits the side, the BUY fires as usual —
    the gate doesn't suppress an aligned trade."""
    params = _short_params(trend_filter_timeframe=Timeframe.H1, trend_ema_period=5)
    _patch_indicators(monkeypatch)
    strat = PullbackReversalScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._prev_k, strat._prev_d = 10.0, 13.0
    strat._trend_ok = lambda side, p: True  # type: ignore[method-assign]
    await strat.on_bar(ctx, _trigger_bar())
    assert len(captured) == 1
    assert captured[0].side is Side.BUY


@pytest.mark.parametrize(
    "overrides",
    [
        dict(trend_filter_timeframe=Timeframe.TICK),
        dict(trend_filter_timeframe=Timeframe.W1),
        dict(trend_filter_timeframe=Timeframe.MN1),
        dict(trend_filter_timeframe=Timeframe.M1),   # <= entry (M1)
        dict(trend_adx_min=40.0, trend_adx_max=30.0),  # max < min
    ],
)
def test_trend_filter_param_validation_rejects(overrides) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _short_params(**overrides)


# =========================================================================
# Stop-distance floor (lot-explosion guard)
# =========================================================================


@pytest.mark.asyncio
async def test_min_stop_distance_floors_the_stop(monkeypatch) -> None:
    """When ATR is tiny, ``min_stop_distance`` widens the stop so the
    risk engine can't size into a tens-of-lots position. The emitted SL
    sits at the floor, not the (much closer) ATR-based level."""
    params = _short_params(sl_atr_mult=1.0, min_stop_distance=3.0)
    _patch_indicators(monkeypatch, atr_value=0.4)   # ATR-only stop = 0.4
    strat = PullbackReversalScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._prev_k, strat._prev_d = 10.0, 13.0

    await strat.on_bar(ctx, _trigger_bar(close=2336.0))

    assert len(captured) == 1
    # SL = entry(2336) - max(0.4, 3.0) = 2333.0, NOT 2335.6.
    assert captured[0].suggested_sl == pytest.approx(2333.0)


@pytest.mark.asyncio
async def test_min_stop_distance_noop_when_atr_wider(monkeypatch) -> None:
    """When the ATR-based stop already exceeds the floor, the floor is a
    no-op — the ATR stop wins (default 0.0 floor keeps old behaviour)."""
    params = _short_params(sl_atr_mult=1.0, min_stop_distance=1.0)
    _patch_indicators(monkeypatch, atr_value=2.5)   # ATR stop 2.5 > floor 1.0
    strat = PullbackReversalScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._prev_k, strat._prev_d = 10.0, 13.0

    await strat.on_bar(ctx, _trigger_bar(close=2336.0))

    assert len(captured) == 1
    # SL = entry - max(2.5, 1.0) = 2333.5
    assert captured[0].suggested_sl == pytest.approx(2333.5)
