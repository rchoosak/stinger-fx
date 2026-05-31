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
from stinger_fx.core.events import TickEvent, TickStreamUnsubscribedEvent
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


# --- Code review #2 / #3 — replay tag isolation ----------------------------- #


@pytest.mark.asyncio
async def test_replayed_tick_does_not_reset_watchdog() -> None:
    """A tick carrying ``replayed=True`` (from broker gap-fill) must not
    reset the watchdog gauge or update ``_last_tick_time`` — otherwise
    a long replay would mask the fact that the live stream is still
    dead.  Code review #3."""
    bus = AsyncEventBus()
    collector = _isolated_collector(bus)
    collector._WATCHDOG_INTERVAL_SECONDS = 0.05  # type: ignore[misc]
    try:
        await collector.start()
        # Pretend the live stream went silent 30s ago.
        thirty_seconds_ago = datetime.now(UTC) - timedelta(seconds=30)
        collector._last_tick_time[SYMBOL] = thirty_seconds_ago

        # Now a replayed historical tick arrives — should be ignored by
        # watchdog state.
        replayed_tick = Tick(
            symbol=SYMBOL,
            time=datetime.now(UTC) - timedelta(minutes=20),
            bid=1900.0,
            ask=1900.2,
            replayed=True,
        )
        await bus.publish(TickEvent(tick=replayed_tick))
        for _ in range(3):
            await asyncio.sleep(0)
        await asyncio.sleep(0.1)  # one watchdog cycle

        gauge_val = collector.metrics[
            "tick_stream_seconds_since_last"
        ].labels(symbol=SYMBOL)._value.get()
        # Still climbing from the stale baseline — replay must not zero it.
        assert gauge_val >= 29.0, (
            f"watchdog should still see stale stream; got {gauge_val}"
        )
        # _last_tick_time must NOT be advanced by the replay tick.
        assert collector._last_tick_time[SYMBOL] == thirty_seconds_ago
    finally:
        await collector.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_replayed_tick_does_not_pollute_lag_metric() -> None:
    """A replayed tick whose ``tick.time`` is 30 minutes old must not
    push ``tick_pump_lag_seconds`` to 1800s — that would page on-call
    for "tick lag SLO breach" every time we reconnect.  Code review #2."""
    bus = AsyncEventBus()
    collector = _isolated_collector(bus)
    try:
        await collector.start()
        old_tick = Tick(
            symbol=SYMBOL,
            time=datetime.now(UTC) - timedelta(minutes=30),
            bid=1900.0,
            ask=1900.2,
            replayed=True,
        )
        await bus.publish(TickEvent(tick=old_tick))
        for _ in range(3):
            await asyncio.sleep(0)

        # tick_pump_lag_seconds should NOT have been set by this replay.
        # prometheus_client _value defaults to 0 if never set.
        lag = collector.metrics[
            "tick_pump_lag_seconds"
        ].labels(symbol=SYMBOL)._value.get()
        assert lag == pytest.approx(0.0, abs=0.001), (
            f"replay must not push lag gauge; got {lag}"
        )

        # But the received counter SHOULD have incremented — we did
        # observe a tick from the broker, even if it was replay.
        count = collector.metrics[
            "ticks_received_total"
        ].labels(symbol=SYMBOL)._value.get()
        assert count == 1
    finally:
        await collector.stop()
    await bus.close()


# --- Code review #6 — prune state on unsubscribe ---------------------------- #


@pytest.mark.asyncio
async def test_unsubscribe_event_prunes_state() -> None:
    """``TickStreamUnsubscribedEvent`` must clear the symbol from
    ``_last_tick_time`` and remove its gauge label so the watchdog
    stops reporting forever-climbing staleness for a retired stream.
    Code review #6."""
    bus = AsyncEventBus()
    collector = _isolated_collector(bus)
    try:
        await collector.start()
        # Seed a tick so the symbol exists in collector state.
        await bus.publish(TickEvent(tick=_tick(datetime.now(UTC))))
        for _ in range(3):
            await asyncio.sleep(0)
        assert SYMBOL in collector._last_tick_time

        # Broker says: stream retired.
        await bus.publish(
            TickStreamUnsubscribedEvent(broker_name="mt5", symbol=SYMBOL)
        )
        for _ in range(3):
            await asyncio.sleep(0)

        assert SYMBOL not in collector._last_tick_time, (
            "unsubscribe must prune symbol from _last_tick_time"
        )
        # Gauge label must be gone — re-accessing .labels() would
        # re-create it, so check the internal metric registry directly.
        gauge = collector.metrics["tick_stream_seconds_since_last"]
        existing_labels = {tuple(m.name for m in c) for c in gauge._metrics.keys()} \
            if hasattr(gauge, "_metrics") else None
        # Either the label set is empty, or our symbol isn't in it.
        assert all(
            SYMBOL not in label_tuple
            for label_tuple in (gauge._metrics if hasattr(gauge, "_metrics") else {})
        ), "gauge label for unsubscribed symbol should be removed"
    finally:
        await collector.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_unsubscribe_for_unknown_symbol_is_noop() -> None:
    """Receiving unsubscribe for a symbol we never tracked must not
    raise — handler tolerates KeyError on the gauge .remove() call."""
    bus = AsyncEventBus()
    collector = _isolated_collector(bus)
    try:
        await collector.start()
        await bus.publish(
            TickStreamUnsubscribedEvent(broker_name="mt5", symbol="NEVER_SEEN")
        )
        for _ in range(3):
            await asyncio.sleep(0)
        # Watchdog still alive.
        assert not collector._watchdog_task.done()  # type: ignore[union-attr]
    finally:
        await collector.stop()
    await bus.close()


# --- Code review #8 — watchdog resilience ---------------------------------- #


@pytest.mark.asyncio
async def test_watchdog_survives_exception_in_body() -> None:
    """If a single watchdog iteration raises (e.g. a prometheus label op
    blows up), the loop must log + continue rather than die silently.
    Code review #8."""
    bus = AsyncEventBus()
    collector = _isolated_collector(bus)
    collector._WATCHDOG_INTERVAL_SECONDS = 0.02  # type: ignore[misc]
    try:
        await collector.start()
        # Inject a "symbol" whose label op will raise. Use a non-string
        # value so prometheus_client's label coercion raises.
        collector._last_tick_time[SYMBOL] = datetime.now(UTC) - timedelta(seconds=5)

        # Monkey-patch the gauge to raise on .labels() once.
        gauge = collector.metrics["tick_stream_seconds_since_last"]
        real_labels = gauge.labels
        call_count = {"n": 0}

        def flaky_labels(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated label op failure")
            return real_labels(*args, **kwargs)

        gauge.labels = flaky_labels  # type: ignore[method-assign]

        # Let several watchdog cycles run.
        await asyncio.sleep(0.15)

        # Watchdog still alive after the failure.
        assert collector._watchdog_task is not None
        assert not collector._watchdog_task.done(), (
            f"watchdog died on exception: {collector._watchdog_task.exception()}"
        )
        assert call_count["n"] >= 2, "watchdog should have retried after failure"
    finally:
        await collector.stop()
    await bus.close()
