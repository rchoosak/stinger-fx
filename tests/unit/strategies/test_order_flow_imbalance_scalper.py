"""Unit tests for OrderFlowImbalanceScalper.

Same harness as MBS — monkeypatch ``atr`` / ``ema`` / ``sma`` and the
inline ``_true_range_series`` helper so the *decision logic* is the
unit under test, not the indicator math.  The novel piece is the
``on_tick`` accumulator + signed-volume deque, which we exercise by
calling ``on_tick`` directly with synthetic ``Tick`` objects before
firing ``on_bar``.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import structlog

import stinger_fx.strategies.examples.order_flow_imbalance_scalper as ofis_mod
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
from stinger_fx.domain.ticks import Tick
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.examples.order_flow_imbalance_scalper import (
    OrderFlowImbalanceScalper,
    OrderFlowImbalanceScalperParams,
)

SYMBOL = "XAUUSD"


def _ts(hour: int = 10, minute: int = 0, second: int = 0) -> datetime:
    """UTC timestamp on a fixed Wednesday so weekday-aware code paths
    (none in OFIS today, but matches the MBS test fixture) won't trip."""
    return datetime(2024, 1, 3, hour, minute, second, tzinfo=UTC)


def _bar(*, tf: Timeframe, t: datetime, close: float = 2340.0,
         high: float | None = None, low: float | None = None) -> Bar:
    return Bar(
        symbol=SYMBOL, timeframe=tf, time=t,
        open=close, high=high if high is not None else close + 0.5,
        low=low if low is not None else close - 0.5, close=close,
        tick_volume=1000, is_closed=True,
    )


def _tick(*, t: datetime, bid: float, ask: float, volume: int = 1,
          symbol: str = SYMBOL) -> Tick:
    return Tick(symbol=symbol, time=t, bid=bid, ask=ask, volume=volume)


def _short_params(**overrides: Any) -> OrderFlowImbalanceScalperParams:
    base: dict[str, Any] = dict(
        ofi_lookback_ticks=50,
        ofi_window_seconds=60,
        ofi_threshold=10.0,        # easy threshold for tests
        require_m5_trend=True,
        ema_fast_period=5,
        ema_slow_period=10,
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
    return OrderFlowImbalanceScalperParams(**base)


def _patch_indicators(monkeypatch, *,
                     atr_value: float = 1.0,
                     ema_fast: float = 2336.0,
                     ema_slow: float = 2330.0,
                     tr_series: list[float] | None = None) -> dict[str, list[Any]]:
    state: dict[str, list[Any]] = {
        "ema": [ema_fast, ema_slow],
        "calls_ema": [],
    }

    def _atr_stub(bars, period):  # type: ignore[no-untyped-def]
        return atr_value

    def _ema_stub(closes, period):  # type: ignore[no-untyped-def]
        state["calls_ema"].append(period)
        idx = (len(state["calls_ema"]) - 1) % 2
        return state["ema"][idx]

    def _tr_stub(bars):  # type: ignore[no-untyped-def]
        if tr_series is not None:
            return list(tr_series)
        return [1.0] * 20

    def _sma_stub(values, period):  # type: ignore[no-untyped-def]
        if period <= 0 or len(values) < period:
            return None
        return sum(values[-period:]) / period

    monkeypatch.setattr(ofis_mod, "atr", _atr_stub)
    monkeypatch.setattr(ofis_mod, "ema", _ema_stub)
    monkeypatch.setattr(ofis_mod, "sma", _sma_stub)
    monkeypatch.setattr(ofis_mod, "_true_range_series", _tr_stub)
    return state


def _build_ctx(*, params: OrderFlowImbalanceScalperParams,
              m1_count: int = 60) -> tuple[StrategyContext, list[Signal], AsyncEventBus]:
    bus = AsyncEventBus()
    captured: list[Signal] = []

    async def sink(sig: Signal) -> None:
        captured.append(sig)

    ctx = StrategyContext(
        strategy_id="ofis_test",
        symbol=params.symbol,
        timeframe=params.entry_timeframe,
        params=params,
        clock=SimClock(_ts(10)),
        logger=structlog.get_logger("ofis_test"),
        magic=99,
        signal_sink=sink,
        subscriptions=OrderFlowImbalanceScalper.subscriptions(params),
        bus=bus,
    )
    m1_view = ctx.history_for(params.symbol, params.entry_timeframe)
    assert m1_view is not None
    for i in range(m1_count):
        m1_view.append_bar(_bar(tf=Timeframe.M1, t=_ts(10) + timedelta(seconds=i)))
    m5_need = max(params.ema_slow_period, params.ema_fast_period) + 5
    m5_view = ctx.history_for(params.symbol, params.structure_timeframe)
    assert m5_view is not None
    for i in range(m5_need):
        m5_view.append_bar(
            _bar(tf=Timeframe.M5, t=_ts(10) + timedelta(minutes=5 * i)),
        )
    return ctx, captured, bus


def _trigger_bar(close: float = 2340.0, hour: int = 10,
                 minute: int = 30, second: int = 0) -> Bar:
    return _bar(tf=Timeframe.M1, t=_ts(hour, minute, second), close=close)


async def _feed_ticks(strat, ctx, ticks: list[Tick]) -> None:
    for tk in ticks:
        await strat.on_tick(ctx, tk)


def _make_upticks(count: int, *, start: datetime,
                 volume_each: int = 5, step: float = 0.05) -> list[Tick]:
    """Sequence of strictly upward mid-shifts.  ``start`` is the time
    of the FIRST tick (used as seed — produces no signed entry); each
    subsequent tick is 1 second later with mid increased by ``step``.
    """
    out: list[Tick] = []
    bid0 = 2340.0
    for i in range(count):
        bid = bid0 + i * step
        ask = bid + 0.20
        out.append(_tick(t=start + timedelta(seconds=i),
                         bid=bid, ask=ask, volume=volume_each))
    return out


def _make_downticks(count: int, *, start: datetime,
                   volume_each: int = 5, step: float = 0.05) -> list[Tick]:
    out: list[Tick] = []
    bid0 = 2340.0
    for i in range(count):
        bid = bid0 - i * step
        ask = bid + 0.20
        out.append(_tick(t=start + timedelta(seconds=i),
                         bid=bid, ask=ask, volume=volume_each))
    return out


# =========================================================================
# OFI accumulator tests
# =========================================================================


@pytest.mark.asyncio
async def test_first_tick_seeds_prev_mid_no_entry_yet() -> None:
    """The first tick has no prior mid to compare against — its sign is
    undefined, so it must NOT be appended to the deque."""
    strat = OrderFlowImbalanceScalper()
    params = _short_params()
    ctx, _captured, _ = _build_ctx(params=params)
    await strat.on_tick(ctx, _tick(t=_ts(10, 0, 0), bid=2340.0, ask=2340.2, volume=5))
    assert len(strat._ofi_deque) == 0
    assert strat._prev_mid == pytest.approx(2340.1)


@pytest.mark.asyncio
async def test_upticks_produce_positive_signed_volume() -> None:
    """A sequence of upward mid-shifts must all land in the deque with
    sign = +1."""
    strat = OrderFlowImbalanceScalper()
    params = _short_params()
    ctx, _captured, _ = _build_ctx(params=params)
    ticks = _make_upticks(5, start=_ts(10, 0, 0), volume_each=4)
    await _feed_ticks(strat, ctx, ticks)
    # 5 ticks fed → first seeds, 4 produce signed entries (+4 each)
    assert len(strat._ofi_deque) == 4
    assert sum(v for _, v in strat._ofi_deque) == 16


@pytest.mark.asyncio
async def test_downticks_produce_negative_signed_volume() -> None:
    strat = OrderFlowImbalanceScalper()
    params = _short_params()
    ctx, _captured, _ = _build_ctx(params=params)
    ticks = _make_downticks(5, start=_ts(10, 0, 0), volume_each=4)
    await _feed_ticks(strat, ctx, ticks)
    assert len(strat._ofi_deque) == 4
    assert sum(v for _, v in strat._ofi_deque) == -16


@pytest.mark.asyncio
async def test_equal_mids_contribute_zero() -> None:
    """If two consecutive mids are equal, sign=0 and the tick lands in
    the deque with 0 contribution (so the count grows but the sum
    doesn't move)."""
    strat = OrderFlowImbalanceScalper()
    params = _short_params()
    ctx, _captured, _ = _build_ctx(params=params)
    t0 = _ts(10, 0, 0)
    ticks = [
        _tick(t=t0, bid=2340.0, ask=2340.2, volume=5),                # seed
        _tick(t=t0 + timedelta(seconds=1), bid=2340.0, ask=2340.2, volume=5),  # equal
        _tick(t=t0 + timedelta(seconds=2), bid=2340.0, ask=2340.2, volume=5),  # equal
    ]
    await _feed_ticks(strat, ctx, ticks)
    assert len(strat._ofi_deque) == 2
    assert sum(v for _, v in strat._ofi_deque) == 0


@pytest.mark.asyncio
async def test_other_symbol_ticks_are_ignored() -> None:
    """Multi-symbol live setups feed ticks for every subscribed symbol
    through the same handler — OFIS must accumulate only its own."""
    strat = OrderFlowImbalanceScalper()
    params = _short_params()
    ctx, _captured, _ = _build_ctx(params=params)
    await strat.on_tick(ctx, _tick(t=_ts(10, 0, 0), bid=1.0, ask=1.001,
                                   volume=5, symbol="EURUSD"))
    await strat.on_tick(ctx, _tick(t=_ts(10, 0, 1), bid=1.001, ask=1.002,
                                   volume=5, symbol="EURUSD"))
    assert len(strat._ofi_deque) == 0
    assert strat._prev_mid is None    # never seeded


@pytest.mark.asyncio
async def test_window_seconds_prunes_stale_samples() -> None:
    """Entries older than ``ofi_window_seconds`` before the OFI-read
    time must be pruned on the next read."""
    strat = OrderFlowImbalanceScalper()
    params = _short_params(ofi_window_seconds=30)
    ctx, _captured, _ = _build_ctx(params=params)
    # Feed 5 upticks starting at 10:00:00; deque ends with 4 entries
    # spanning 10:00:01 to 10:00:04.
    await _feed_ticks(strat, ctx, _make_upticks(5, start=_ts(10, 0, 0)))
    # Now is 10:01:00 → cutoff is 10:00:30 → all entries stale.
    s = strat._compute_ofi_sum(_ts(10, 1, 0), params)
    assert s == 0
    assert len(strat._ofi_deque) == 0


@pytest.mark.asyncio
async def test_lookback_ticks_caps_deque_length() -> None:
    """``ofi_lookback_ticks`` enforces a hard cap so a flood of ticks
    can't blow memory."""
    strat = OrderFlowImbalanceScalper()
    params = _short_params(ofi_lookback_ticks=10)
    ctx, _captured, _ = _build_ctx(params=params)
    await _feed_ticks(strat, ctx, _make_upticks(30, start=_ts(10, 0, 0)))
    assert len(strat._ofi_deque) == 10     # 29 signed entries → capped at 10


# =========================================================================
# Entry tests (on_bar)
# =========================================================================


@pytest.mark.asyncio
async def test_emits_buy_when_ofi_strongly_positive(monkeypatch) -> None:
    """Build up a strong positive OFI signal via upticks, then fire a
    bar — BUY must emit."""
    params = _short_params(ofi_threshold=10.0)
    _patch_indicators(monkeypatch)
    strat = OrderFlowImbalanceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    # 5 upticks @ volume 5 each → +20 signed (>= threshold 10)
    await _feed_ticks(strat, ctx, _make_upticks(5, start=_ts(10, 30, 0),
                                                 volume_each=5))
    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10, minute=30,
                                          second=5))

    assert len(captured) == 1, f"expected BUY signal; got {captured}"
    sig = captured[0]
    assert sig.side is Side.BUY
    # SL = entry - 1.0 * ATR(1.0) = 2339; TP = entry + 2.0 * 1.0 = 2342
    assert sig.suggested_sl == pytest.approx(2339.0)
    assert sig.suggested_tp == pytest.approx(2342.0)


@pytest.mark.asyncio
async def test_emits_sell_when_ofi_strongly_negative(monkeypatch) -> None:
    params = _short_params(ofi_threshold=10.0)
    _patch_indicators(monkeypatch, ema_fast=2330.0, ema_slow=2336.0)
    strat = OrderFlowImbalanceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    await _feed_ticks(strat, ctx, _make_downticks(5, start=_ts(10, 30, 0),
                                                    volume_each=5))
    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10, minute=30,
                                          second=5))

    assert len(captured) == 1
    sig = captured[0]
    assert sig.side is Side.SELL
    # SL = entry + ATR = 2341; TP = entry - 2 * ATR = 2338
    assert sig.suggested_sl == pytest.approx(2341.0)
    assert sig.suggested_tp == pytest.approx(2338.0)


@pytest.mark.asyncio
async def test_no_entry_when_ofi_below_threshold(monkeypatch) -> None:
    """|OFI| < threshold → no signal."""
    params = _short_params(ofi_threshold=100.0)   # need much more flow
    _patch_indicators(monkeypatch)
    strat = OrderFlowImbalanceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    # 4 ticks @ vol 5 = +20 (well under 100)
    await _feed_ticks(strat, ctx, _make_upticks(5, start=_ts(10, 30, 0),
                                                 volume_each=5))
    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10, minute=30,
                                          second=5))
    assert captured == []


@pytest.mark.asyncio
async def test_empty_deque_means_no_entry(monkeypatch) -> None:
    """First-ever bar with no prior ticks → OFI sum is 0 → no entry."""
    params = _short_params(ofi_threshold=10.0)
    _patch_indicators(monkeypatch)
    strat = OrderFlowImbalanceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10))
    assert captured == []


@pytest.mark.asyncio
async def test_session_avoid_blocks_entry(monkeypatch) -> None:
    params = _short_params(ofi_threshold=10.0)
    _patch_indicators(monkeypatch)
    strat = OrderFlowImbalanceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    await _feed_ticks(strat, ctx, _make_upticks(5, start=_ts(14, 0, 0),
                                                 volume_each=5))
    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=14, minute=30))
    assert captured == []


@pytest.mark.asyncio
async def test_vol_gate_blocks_entry_on_spike(monkeypatch) -> None:
    params = _short_params(ofi_threshold=10.0, vol_gate_mult=1.8,
                           vol_gate_lookback=5)
    _patch_indicators(monkeypatch, tr_series=[1.0, 1.0, 1.0, 1.0, 1.0, 5.0])
    strat = OrderFlowImbalanceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    await _feed_ticks(strat, ctx, _make_upticks(5, start=_ts(10, 30, 0),
                                                 volume_each=5))
    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10, minute=30,
                                          second=5))
    assert captured == []


@pytest.mark.asyncio
async def test_m5_filter_off_lets_buy_fire_without_trend(monkeypatch) -> None:
    """``require_m5_trend=False`` skips the EMA check entirely."""
    params = _short_params(ofi_threshold=10.0, require_m5_trend=False)
    _patch_indicators(monkeypatch, ema_fast=2330.0, ema_slow=2336.0)
    strat = OrderFlowImbalanceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    await _feed_ticks(strat, ctx, _make_upticks(5, start=_ts(10, 30, 0),
                                                 volume_each=5))
    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10, minute=30,
                                          second=5))
    assert len(captured) == 1
    assert captured[0].side is Side.BUY


# =========================================================================
# Exit + lifecycle
# =========================================================================


@pytest.mark.asyncio
async def test_time_stop_closes_position_after_max_hold(monkeypatch) -> None:
    params = _short_params(max_hold_bars_m1=3)
    _patch_indicators(monkeypatch)
    strat = OrderFlowImbalanceScalper()
    ctx, _captured, bus = _build_ctx(params=params)
    strat._open_ticket = 42
    strat._open_side = Side.BUY
    strat._open_bars = 2

    close_events: list[Any] = []
    from stinger_fx.core.events import ClosePositionRequestEvent
    bus.subscribe(ClosePositionRequestEvent,
                  lambda e: close_events.append(e),
                  name="probe.close")
    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10))
    import asyncio
    for _ in range(3):
        await asyncio.sleep(0)
    assert len(close_events) == 1


@pytest.mark.asyncio
async def test_on_order_filled_records_open_ticket() -> None:
    strat = OrderFlowImbalanceScalper()
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
    assert strat._open_side is Side.BUY


@pytest.mark.asyncio
async def test_on_position_closed_starts_cooldown() -> None:
    strat = OrderFlowImbalanceScalper()
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
    params = _short_params(ofi_threshold=10.0, cooldown_bars_m1=2)
    _patch_indicators(monkeypatch)
    strat = OrderFlowImbalanceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    await _feed_ticks(strat, ctx, _make_upticks(5, start=_ts(10, 30, 0),
                                                 volume_each=5))
    strat._cooldown_left = 2

    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10, minute=30,
                                          second=5))
    assert captured == []
    assert strat._cooldown_left == 1
    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10, minute=30,
                                          second=6))
    assert captured == []
    assert strat._cooldown_left == 0
    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10, minute=30,
                                          second=7))
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_only_one_open_position_at_a_time(monkeypatch) -> None:
    params = _short_params(ofi_threshold=10.0)
    _patch_indicators(monkeypatch)
    strat = OrderFlowImbalanceScalper()
    ctx, captured, _ = _build_ctx(params=params)
    strat._open_ticket = 999
    strat._open_side = Side.BUY
    strat._open_bars = 1
    await _feed_ticks(strat, ctx, _make_upticks(5, start=_ts(10, 30, 0),
                                                 volume_each=5))
    await strat.on_bar(ctx, _trigger_bar(close=2340.0, hour=10, minute=30,
                                          second=5))
    assert captured == [], "no new signal while a position is open"
