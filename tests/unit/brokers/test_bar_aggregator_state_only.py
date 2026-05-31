"""Unit tests for BarAggregator's ``emit_bars=False`` warmup mode + the
``copy_state_into()`` state-transfer helper (Plan A2).

Both are used by the live runtime's startup tick-backfill flow:
historical ticks are replayed through the live aggregator with
``emit_bars=False`` so its OHLC state warms up without publishing any
backdated BarEvents.  ``copy_state_into()`` is the safety-net pattern
for tests that need a separate temp aggregator and then have to
hand the state off.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from stinger_fx.brokers.bar_aggregator import BarAggregator
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import BarEvent, TickEvent
from stinger_fx.domain import Tick, Timeframe

SYMBOL = "XAUUSD"


def _tick(t: datetime, bid: float) -> Tick:
    return Tick(symbol=SYMBOL, time=t, bid=bid, ask=bid + 0.2)


@pytest.mark.asyncio
async def test_emit_bars_false_suppresses_publish_at_boundary_crossing() -> None:
    """An aggregator built with ``emit_bars=False`` accumulates OHLC
    state correctly but never publishes BarEvents — even when ticks
    cross a bar boundary."""
    bus = AsyncEventBus()
    captured: list[BarEvent] = []

    async def collect(evt: BarEvent) -> None:
        captured.append(evt)

    bus.subscribe(BarEvent, collect, name="probe.bar")

    agg = BarAggregator(SYMBOL, Timeframe.M1, bus, emit_bars=False)

    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    # 3 ticks in the [12:00, 12:01) bar
    await agg.on_tick(TickEvent(tick=_tick(base + timedelta(seconds=10), 1900.0)))
    await agg.on_tick(TickEvent(tick=_tick(base + timedelta(seconds=30), 1905.0)))
    await agg.on_tick(TickEvent(tick=_tick(base + timedelta(seconds=55), 1902.0)))
    # Crossing tick into [12:01, 12:02) — would normally fire emit
    await agg.on_tick(TickEvent(tick=_tick(base + timedelta(seconds=65), 1903.0)))

    # Drain the bus — proves the lack of events is real (not just
    # un-drained tasks).
    import asyncio
    for _ in range(3):
        await asyncio.sleep(0)

    # No BarEvent reached the bus.
    assert captured == []

    # But state advanced: aggregator is now sitting on the 12:01 bar
    # with the crossing tick as both open and close.
    assert agg._open_time == base + timedelta(minutes=1)
    assert agg._o == pytest.approx(1903.0)
    assert agg._c == pytest.approx(1903.0)
    await bus.close()


@pytest.mark.asyncio
async def test_emit_bars_true_default_publishes_at_boundary() -> None:
    """Sanity: pre-A2 default behaviour is unchanged — bar events still
    flow with ``emit_bars=True`` (the implicit default)."""
    bus = AsyncEventBus()
    captured: list[BarEvent] = []

    async def collect(evt: BarEvent) -> None:
        captured.append(evt)

    bus.subscribe(BarEvent, collect, name="probe.bar")

    agg = BarAggregator(SYMBOL, Timeframe.M1, bus)  # emit_bars default = True

    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    await agg.on_tick(TickEvent(tick=_tick(base + timedelta(seconds=10), 1900.0)))
    await agg.on_tick(TickEvent(tick=_tick(base + timedelta(seconds=65), 1903.0)))

    # Drain the bus — publish dispatches via asyncio tasks.
    import asyncio
    for _ in range(3):
        await asyncio.sleep(0)

    assert len(captured) == 1
    assert captured[0].bar.is_closed is True
    await bus.close()


@pytest.mark.asyncio
async def test_copy_state_into_transfers_ohlc_to_live_aggregator() -> None:
    """The warmup pattern: drive a throwaway ``emit_bars=False``
    aggregator across historical ticks, then ``copy_state_into()`` the
    live aggregator.  The live aggregator should look exactly as if it
    had been running the whole time."""
    bus = AsyncEventBus()
    warmup = BarAggregator(SYMBOL, Timeframe.M1, bus, emit_bars=False)
    live = BarAggregator(SYMBOL, Timeframe.M1, bus)  # emit_bars=True

    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    # Replay 3 ticks into warmup
    await warmup.on_tick(TickEvent(tick=_tick(base + timedelta(seconds=10), 1900.0)))
    await warmup.on_tick(TickEvent(tick=_tick(base + timedelta(seconds=30), 1908.0)))
    await warmup.on_tick(TickEvent(tick=_tick(base + timedelta(seconds=55), 1903.0)))

    # Transfer state to live aggregator
    warmup.copy_state_into(live)

    assert live._open_time == warmup._open_time
    assert live._o == pytest.approx(1900.0)
    assert live._h == pytest.approx(1908.0)
    assert live._l == pytest.approx(1900.0)
    assert live._c == pytest.approx(1903.0)
    assert live._tick_count == 3
    await bus.close()


@pytest.mark.asyncio
async def test_copy_state_into_rejects_mismatched_aggregators() -> None:
    """Safety: refuse to copy state between aggregators that disagree
    on symbol or timeframe — would silently corrupt OHLC."""
    bus = AsyncEventBus()
    src = BarAggregator("XAUUSD", Timeframe.M1, bus)
    dst_wrong_symbol = BarAggregator("EURUSD", Timeframe.M1, bus)
    dst_wrong_tf = BarAggregator("XAUUSD", Timeframe.M5, bus)

    with pytest.raises(ValueError, match="mismatched"):
        src.copy_state_into(dst_wrong_symbol)
    with pytest.raises(ValueError, match="mismatched"):
        src.copy_state_into(dst_wrong_tf)
    await bus.close()
