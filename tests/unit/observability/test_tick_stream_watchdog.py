"""Unit tests for ``MetricsCollector``'s tick-stream watchdog (Plan A3).

The watchdog is a periodic task that updates
``stinger_tick_stream_seconds_since_last{symbol}`` every
``_WATCHDOG_INTERVAL_SECONDS``.  Unlike ``tick_pump_lag_seconds`` (only
refreshed by ``_on_tick``), this gauge climbs continuously while no
ticks arrive — the canonical "is the stream alive?" alerting signal.

Pins:
  * The gauge resets to ~0 on tick arrival (immediate refresh in
    ``_on_tick`` to avoid waiting up to one watchdog cycle).
  * The gauge climbs above 0 between watchdog ticks while no tick
    arrives.
  * The watchdog task starts on ``start()`` and is cancelled on
    ``stop()``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from prometheus_client import CollectorRegistry

from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import TickEvent
from stinger_fx.domain import Tick
from stinger_fx.observability.metrics import MetricsCollector, make_metrics


def _isolated_collector(bus: AsyncEventBus) -> MetricsCollector:
    """Use a dedicated ``CollectorRegistry`` so tests don't collide on
    the prometheus_client global registry."""
    return MetricsCollector(bus, metrics=make_metrics(CollectorRegistry()))

SYMBOL = "XAUUSD"


def _tick(t: datetime, bid: float = 1900.0) -> Tick:
    return Tick(symbol=SYMBOL, time=t, bid=bid, ask=bid + 0.2)


@pytest.mark.asyncio
async def test_watchdog_task_starts_and_stops_cleanly() -> None:
    """start() creates the watchdog task; stop() cancels it without
    raising."""
    bus = AsyncEventBus()
    collector = _isolated_collector(bus)
    try:
        await collector.start()
        assert collector._watchdog_task is not None
        assert not collector._watchdog_task.done()
    finally:
        await collector.stop()
        assert collector._watchdog_task is None
    await bus.close()


@pytest.mark.asyncio
async def test_gauge_zeroed_immediately_on_tick_arrival() -> None:
    """``_on_tick`` records the tick's wall-clock arrival AND sets the
    watchdog gauge to 0 right away — consumers don't have to wait up
    to ``_WATCHDOG_INTERVAL_SECONDS`` to see a healthy stream."""
    bus = AsyncEventBus()
    collector = _isolated_collector(bus)
    try:
        await collector.start()
        # Publish a tick — _on_tick will record + zero the gauge.
        await bus.publish(TickEvent(tick=_tick(datetime.now(UTC))))
        for _ in range(3):
            await asyncio.sleep(0)
        # The watchdog gauge for our symbol must be at ~0 (just got a tick).
        gauge_val = collector.metrics[
            "tick_stream_seconds_since_last"
        ].labels(symbol=SYMBOL)._value.get()
        assert gauge_val == pytest.approx(0.0, abs=0.1)
        # And the collector now tracks our symbol's last-seen time.
        assert SYMBOL in collector._last_tick_time
    finally:
        await collector.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_gauge_climbs_when_no_ticks_arrive() -> None:
    """Set a stale ``_last_tick_time`` manually (simulating a tick that
    arrived a long time ago) and drive one watchdog cycle — the gauge
    must reflect the staleness in seconds."""
    bus = AsyncEventBus()
    collector = _isolated_collector(bus)
    # Speed up so we don't wait 5 real seconds for the watchdog tick.
    collector._WATCHDOG_INTERVAL_SECONDS = 0.05  # type: ignore[misc]
    try:
        await collector.start()
        # Tick was "seen" 10 seconds ago.
        ten_seconds_ago = datetime.now(UTC) - timedelta(seconds=10)
        collector._last_tick_time[SYMBOL] = ten_seconds_ago

        # Wait for at least one watchdog tick.
        await asyncio.sleep(0.15)

        gauge_val = collector.metrics[
            "tick_stream_seconds_since_last"
        ].labels(symbol=SYMBOL)._value.get()
        assert gauge_val >= 9.5, (
            f"watchdog must observe ~10s staleness; got {gauge_val}"
        )
    finally:
        await collector.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_watchdog_handles_no_subscribed_symbols() -> None:
    """No symbols → watchdog loop is a no-op (doesn't raise)."""
    bus = AsyncEventBus()
    collector = _isolated_collector(bus)
    collector._WATCHDOG_INTERVAL_SECONDS = 0.02  # type: ignore[misc]
    try:
        await collector.start()
        # No ticks ever published → _last_tick_time is empty.
        await asyncio.sleep(0.1)
        # Watchdog still running, no errors.
        assert collector._watchdog_task is not None
        assert not collector._watchdog_task.done(), (
            f"watchdog exited unexpectedly: {collector._watchdog_task.exception()}"
        )
    finally:
        await collector.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_per_symbol_isolation() -> None:
    """A stale EURUSD must not zero the XAUUSD gauge.  Each symbol's
    last-seen time is tracked independently."""
    bus = AsyncEventBus()
    collector = _isolated_collector(bus)
    collector._WATCHDOG_INTERVAL_SECONDS = 0.05  # type: ignore[misc]
    try:
        await collector.start()
        # EURUSD: 20s stale; XAUUSD: just arrived.
        collector._last_tick_time["EURUSD"] = (
            datetime.now(UTC) - timedelta(seconds=20)
        )
        await bus.publish(TickEvent(tick=_tick(datetime.now(UTC))))
        for _ in range(3):
            await asyncio.sleep(0)
        # Let the watchdog refresh.
        await asyncio.sleep(0.15)

        xau_val = collector.metrics[
            "tick_stream_seconds_since_last"
        ].labels(symbol="XAUUSD")._value.get()
        eur_val = collector.metrics[
            "tick_stream_seconds_since_last"
        ].labels(symbol="EURUSD")._value.get()

        assert xau_val < 2.0, f"XAUUSD should be fresh; got {xau_val}"
        assert eur_val >= 19.0, f"EURUSD should be stale; got {eur_val}"
    finally:
        await collector.stop()
    await bus.close()
