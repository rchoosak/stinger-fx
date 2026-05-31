"""Integration test for MT5Broker's reconnect tick gap-fill (Plan A3).

After ``BrokerReconnectedEvent`` fires (Phase 6.1.A), the broker now
calls ``_gap_fill_after_reconnect``:
  * For each subscribed symbol with a known ``_last_tick_time``,
    fetch all ticks via ``copy_ticks_range(symbol, last_seen, now)``.
  * Filter out any tick whose timestamp is <= ``last_seen`` (dedupe
    against what the live loop already published).
  * Republish each missed tick as ``TickEvent`` on the bus.
  * Update ``_last_tick_time`` so the live ``_tick_loop`` doesn't
    re-publish them as "new".

This test pins:
  1. Missed ticks reach the bus after reconnect.
  2. Ticks already seen (timestamp <= last_seen) are NOT republished.
  3. Symbols with no ``_last_tick_time`` (newly subscribed during
     outage) are skipped — no baseline to gap-fill from.
  4. Failure during gap-fill does not roll the broker back into a
     disconnected state.

The test extends the ``_FakeMT5`` pattern from
``tests/unit/brokers/test_mt5_reconnect.py`` so we can stage tick
history for ``copy_ticks_range`` to return.
"""

from __future__ import annotations

import asyncio
import sys
import types
from datetime import UTC, datetime

import pytest

from stinger_fx.brokers.mt5.broker import MT5Broker
from stinger_fx.config.models import MT5Config
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import (
    BrokerDisconnectedEvent,
    BrokerReconnectedEvent,
    TickEvent,
    TickStreamUnsubscribedEvent,
)
from stinger_fx.domain import Tick


class _FakeMT5WithTickHistory:
    """Fake MetaTrader5 module that records gap-fill queries and returns
    pre-staged historical ticks via ``copy_ticks_range``."""

    TRADE_RETCODE_DONE = 10009
    COPY_TICKS_ALL = -1

    def __init__(self) -> None:
        self.connected = True
        self.init_calls = 0
        self.shutdown_calls = 0
        # Disconnect simulation
        self.terminal_info_none_count = 0
        # Staged historical tick stream (returned by copy_ticks_range)
        self.staged_history: list[dict] = []
        # Record copy_ticks_range calls so the test can verify args
        self.history_queries: list[tuple[str, datetime, datetime]] = []

    def stage_history(self, ticks: list[dict]) -> None:
        """Set the tick list copy_ticks_range will return on next call."""
        self.staged_history = ticks

    # --- lifecycle / probe ----------------------------------------------

    def initialize(self, **_kwargs) -> bool:
        self.init_calls += 1
        self.connected = True
        return True

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.connected = False

    def terminal_info(self):
        if self.terminal_info_none_count > 0:
            self.terminal_info_none_count -= 1
            return None
        if not self.connected:
            return None
        return types.SimpleNamespace(connected=True)

    def last_error(self):
        return (0, "no error")

    def symbol_select(self, _symbol: str, _enable: bool) -> bool:
        return True

    def symbol_info_tick(self, _symbol: str):
        # Live tick poller — return None so we control all ticks via
        # the gap-fill path only.  Avoids racing with the live loop.
        return None

    def copy_ticks_range(
        self, symbol: str, start: datetime, end: datetime, _flags: int,
    ):
        self.history_queries.append((symbol, start, end))
        return list(self.staged_history)


def _install(monkeypatch, fake) -> None:
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)


def _make_broker(bus) -> MT5Broker:
    cfg = MT5Config(terminal_path="", login=0, password="", server="", timeout_ms=1000)
    broker = MT5Broker(bus, cfg)
    # Fast loops so the test doesn't sit idle.
    broker._health_check_interval = 0.02
    broker._tick_poll_interval = 0.5    # high so live poll doesn't interfere
    broker._reconnect_backoff = [0.01]
    return broker


# ---------------------------------------------------------------------- #
# Tests                                                                    #
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_gap_fill_replays_missed_ticks_after_reconnect(monkeypatch) -> None:
    """After reconnect, ticks staged in ``copy_ticks_range`` whose
    timestamps fall AFTER the last-seen tick are republished on the
    bus.  Each new tick lands as a fresh ``TickEvent``."""
    fake = _FakeMT5WithTickHistory()
    _install(monkeypatch, fake)

    bus = AsyncEventBus()
    ticks_captured: list[TickEvent] = []

    async def collect_tick(evt: TickEvent) -> None:
        ticks_captured.append(evt)

    reconnected_evts: list[BrokerReconnectedEvent] = []

    async def collect_rc(evt: BrokerReconnectedEvent) -> None:
        reconnected_evts.append(evt)

    bus.subscribe(TickEvent, collect_tick, name="probe.tick")
    bus.subscribe(BrokerReconnectedEvent, collect_rc, name="probe.rc")

    broker = _make_broker(bus)
    try:
        await broker.connect()
        await broker.subscribe_ticks("XAUUSD")

        # Pretend the live loop had seen a tick at 12:00:00.
        baseline = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        broker._last_tick_time["XAUUSD"] = baseline

        # Stage 3 ticks that "arrived during the outage":
        #   * 11:59:59  (BEFORE baseline — must be filtered)
        #   * 12:00:00  (EQUAL baseline — must be filtered, dedupe)
        #   * 12:00:30  (AFTER baseline — must be replayed)
        #   * 12:00:45  (AFTER baseline — must be replayed)
        fake.stage_history([
            {
                "time": int(datetime(2024, 1, 1, 11, 59, 59, tzinfo=UTC).timestamp()),
                "bid": 1900.0, "ask": 1900.2, "last": 1900.0,
                "volume": 1, "flags": 0,
            },
            {
                "time": int(baseline.timestamp()),
                "bid": 1901.0, "ask": 1901.2, "last": 1901.0,
                "volume": 1, "flags": 0,
            },
            {
                "time": int(datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC).timestamp()),
                "bid": 1902.0, "ask": 1902.2, "last": 1902.0,
                "volume": 1, "flags": 0,
            },
            {
                "time": int(datetime(2024, 1, 1, 12, 0, 45, tzinfo=UTC).timestamp()),
                "bid": 1903.0, "ask": 1903.2, "last": 1903.0,
                "volume": 1, "flags": 0,
            },
        ])

        # Trigger a disconnect-then-reconnect cycle.
        fake.terminal_info_none_count = 1
        for _ in range(80):
            await asyncio.sleep(0.02)
            if reconnected_evts:
                break
        assert reconnected_evts, "broker never reported reconnected"

        # Drain the bus so the gap-fill TickEvent publishes land in
        # the captured list.
        for _ in range(5):
            await asyncio.sleep(0.01)

        # Only the 2 AFTER-baseline ticks should have been replayed.
        bids = sorted(evt.tick.bid for evt in ticks_captured)
        assert bids == [1902.0, 1903.0], (
            f"only ticks after baseline should be replayed; got {bids}"
        )

        # ``copy_ticks_range`` was called for the subscribed symbol.
        assert any(
            sym == "XAUUSD" and start == baseline
            for (sym, start, _end) in fake.history_queries
        ), f"expected copy_ticks_range(XAUUSD, {baseline}, ...); got {fake.history_queries}"

        # ``_last_tick_time`` updated to the latest replayed tick so
        # the live loop won't double-publish.
        assert broker._last_tick_time["XAUUSD"] == datetime(
            2024, 1, 1, 12, 0, 45, tzinfo=UTC,
        )
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_gap_fill_skips_symbols_with_no_baseline(monkeypatch) -> None:
    """A symbol that was never observed (no entry in
    ``_last_tick_time``) is skipped — we don't know what window to
    query MT5 for."""
    fake = _FakeMT5WithTickHistory()
    _install(monkeypatch, fake)

    bus = AsyncEventBus()
    broker = _make_broker(bus)
    try:
        await broker.connect()
        await broker.subscribe_ticks("XAUUSD")
        # Deliberately leave _last_tick_time["XAUUSD"] unset.
        assert "XAUUSD" not in broker._last_tick_time

        await broker._gap_fill_after_reconnect()

        # No copy_ticks_range calls — nothing to query for.
        assert fake.history_queries == []
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_gap_fill_swallows_broker_errors(monkeypatch) -> None:
    """If ``copy_ticks_range`` raises (broker bug, permission, etc.)
    the gap-fill is silently skipped — must NOT bubble up and crash
    the reconnect handler that just succeeded."""
    fake = _FakeMT5WithTickHistory()

    def _failing_copy_ticks_range(*args, **kwargs):
        raise RuntimeError("simulated SDK error")

    fake.copy_ticks_range = _failing_copy_ticks_range  # type: ignore[method-assign]
    _install(monkeypatch, fake)

    bus = AsyncEventBus()
    broker = _make_broker(bus)
    try:
        await broker.connect()
        await broker.subscribe_ticks("XAUUSD")
        broker._last_tick_time["XAUUSD"] = datetime(
            2024, 1, 1, 12, 0, 0, tzinfo=UTC,
        )

        # Should not raise.
        await broker._gap_fill_after_reconnect()

        # Broker stays connected.
        assert broker._connected is True
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_gap_fill_handles_empty_history(monkeypatch) -> None:
    """An empty tick window (no missed ticks, e.g. weekend disconnect)
    is a no-op — no TickEvents published, no errors."""
    fake = _FakeMT5WithTickHistory()
    fake.stage_history([])
    _install(monkeypatch, fake)

    bus = AsyncEventBus()
    captured: list[TickEvent] = []

    async def collect(evt: TickEvent) -> None:
        captured.append(evt)

    bus.subscribe(TickEvent, collect, name="probe.tick")

    broker = _make_broker(bus)
    try:
        await broker.connect()
        await broker.subscribe_ticks("XAUUSD")
        broker._last_tick_time["XAUUSD"] = datetime(
            2024, 1, 1, 12, 0, 0, tzinfo=UTC,
        )

        await broker._gap_fill_after_reconnect()

        for _ in range(3):
            await asyncio.sleep(0.01)

        assert captured == []
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_disconnect_event_still_fires_before_gap_fill(monkeypatch) -> None:
    """Sanity: the gap-fill change must not break the existing
    Disconnect → Reconnect event sequence (regression for Phase 6.1.A
    tests).  Gap-fill is appended *after* the Reconnected event."""
    fake = _FakeMT5WithTickHistory()
    fake.stage_history([])   # no missed ticks
    _install(monkeypatch, fake)

    bus = AsyncEventBus()
    dc: list[BrokerDisconnectedEvent] = []
    rc: list[BrokerReconnectedEvent] = []

    async def collect_dc(evt: BrokerDisconnectedEvent) -> None:
        dc.append(evt)

    async def collect_rc(evt: BrokerReconnectedEvent) -> None:
        rc.append(evt)

    bus.subscribe(BrokerDisconnectedEvent, collect_dc, name="probe.dc")
    bus.subscribe(BrokerReconnectedEvent, collect_rc, name="probe.rc")

    broker = _make_broker(bus)
    try:
        await broker.connect()
        await broker.subscribe_ticks("XAUUSD")
        broker._last_tick_time["XAUUSD"] = datetime(
            2024, 1, 1, 12, 0, 0, tzinfo=UTC,
        )

        fake.terminal_info_none_count = 1
        for _ in range(80):
            await asyncio.sleep(0.02)
            if rc:
                break

        assert len(dc) == 1
        assert len(rc) == 1
    finally:
        await broker.disconnect()
        await bus.close()


# --------------------------------------------------------------------- #
# Code review fixes                                                       #
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_replayed_ticks_are_tagged(monkeypatch) -> None:
    """Code review #2 / #3 — gap-fill must tag every replayed TickEvent
    with ``evt.replayed=True`` (the flag lives on TickEvent, not Tick)
    so MetricsCollector can skip lag updates and watchdog resets for
    them."""
    fake = _FakeMT5WithTickHistory()
    _install(monkeypatch, fake)

    bus = AsyncEventBus()
    captured: list[TickEvent] = []

    async def collect(evt: TickEvent) -> None:
        captured.append(evt)

    bus.subscribe(TickEvent, collect, name="probe.tick")

    broker = _make_broker(bus)
    try:
        await broker.connect()
        await broker.subscribe_ticks("XAUUSD")
        baseline = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        broker._last_tick_time["XAUUSD"] = baseline
        fake.stage_history([
            {
                "time": int(datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC).timestamp()),
                "bid": 1902.0, "ask": 1902.2, "last": 1902.0,
                "volume": 1, "flags": 0,
            },
        ])

        await broker._gap_fill_after_reconnect()
        for _ in range(3):
            await asyncio.sleep(0.01)

        assert captured, "expected at least one replayed tick"
        assert all(evt.replayed is True for evt in captured), (
            "every gap-fill TickEvent must carry replayed=True"
        )
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_reconnected_event_fires_after_gap_fill(monkeypatch) -> None:
    """Code review #1 / #4 — ``BrokerReconnectedEvent`` must be
    published AFTER the gap-fill TickEvents land on the bus, so
    downstream consumers see the broker as fully caught up when they
    observe the reconnect."""
    fake = _FakeMT5WithTickHistory()
    _install(monkeypatch, fake)

    bus = AsyncEventBus()
    # Recorded order of events as they arrive at subscribers.
    order: list[str] = []

    async def collect_tick(evt: TickEvent) -> None:
        if evt.replayed:
            order.append("tick")

    async def collect_rc(_evt: BrokerReconnectedEvent) -> None:
        order.append("reconnected")

    bus.subscribe(TickEvent, collect_tick, name="probe.tick")
    bus.subscribe(BrokerReconnectedEvent, collect_rc, name="probe.rc")

    broker = _make_broker(bus)
    try:
        await broker.connect()
        await broker.subscribe_ticks("XAUUSD")
        baseline = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        broker._last_tick_time["XAUUSD"] = baseline
        fake.stage_history([
            {
                "time": int(datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC).timestamp()),
                "bid": 1902.0, "ask": 1902.2, "last": 1902.0,
                "volume": 1, "flags": 0,
            },
            {
                "time": int(datetime(2024, 1, 1, 12, 0, 45, tzinfo=UTC).timestamp()),
                "bid": 1903.0, "ask": 1903.2, "last": 1903.0,
                "volume": 1, "flags": 0,
            },
        ])

        fake.terminal_info_none_count = 1
        for _ in range(80):
            await asyncio.sleep(0.02)
            if "reconnected" in order:
                break
        for _ in range(5):
            await asyncio.sleep(0.01)

        # Every replayed "tick" must precede "reconnected".
        rc_idx = order.index("reconnected")
        assert rc_idx > 0, f"reconnected fired before any gap-fill tick: {order}"
        assert all(label == "tick" for label in order[:rc_idx])
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_gap_fill_per_call_timeout_does_not_block(monkeypatch) -> None:
    """Code review #9 — if ``copy_ticks_range`` hangs for one symbol,
    we abort that symbol via per-call timeout and continue.  The
    reconnect handler must NOT block forever."""
    import stinger_fx.brokers.mt5.broker as broker_mod

    fake = _FakeMT5WithTickHistory()

    def _hanging_copy_ticks_range(*_args, **_kwargs):
        # Simulate a wedged SDK call by blocking the executor thread.
        # The per-call asyncio.wait_for must abort us anyway.
        import time as _time
        _time.sleep(5.0)
        return []

    fake.copy_ticks_range = _hanging_copy_ticks_range  # type: ignore[method-assign]
    _install(monkeypatch, fake)
    # Shrink the timeout so the test runs fast.
    monkeypatch.setattr(
        broker_mod, "GAP_FILL_PER_CALL_TIMEOUT_SECONDS", 0.1,
    )
    monkeypatch.setattr(
        broker_mod, "GAP_FILL_OVERALL_TIMEOUT_SECONDS", 2.0,
    )

    bus = AsyncEventBus()
    broker = _make_broker(bus)
    try:
        await broker.connect()
        await broker.subscribe_ticks("XAUUSD")
        broker._last_tick_time["XAUUSD"] = datetime(
            2024, 1, 1, 12, 0, 0, tzinfo=UTC,
        )

        # Should return promptly (timeout aborts the hung call).
        # Round 2 #11 — use time.monotonic() instead of the deprecated
        # asyncio.get_event_loop().time() pattern (DeprecationWarning in
        # Python 3.10+ when called without a running loop).
        import time as _time
        start = _time.monotonic()
        await broker._gap_fill_after_reconnect()
        elapsed = _time.monotonic() - start
        assert elapsed < 1.0, (
            f"gap_fill should abort hung SDK call quickly; took {elapsed:.2f}s"
        )
        # Broker must still be marked connected — gap-fill failure
        # mustn't roll the connection state back.
        assert broker._connected is True
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_inner_loop_exception_does_not_abort_other_symbols(monkeypatch) -> None:
    """Code review #5 — if one tick in the batch fails to deserialize,
    gap-fill must skip it and continue with the rest (and with other
    symbols)."""
    fake = _FakeMT5WithTickHistory()
    _install(monkeypatch, fake)

    bus = AsyncEventBus()
    captured: list[TickEvent] = []

    async def collect(evt: TickEvent) -> None:
        captured.append(evt)

    bus.subscribe(TickEvent, collect, name="probe.tick")

    broker = _make_broker(bus)
    try:
        await broker.connect()
        await broker.subscribe_ticks("XAUUSD")
        baseline = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        broker._last_tick_time["XAUUSD"] = baseline
        # Stage a batch where the FIRST tick is malformed (missing 'bid')
        # and the second is fine.
        fake.stage_history([
            {
                "time": int(datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC).timestamp()),
                # 'bid' key intentionally missing → KeyError in float(raw["bid"])
                "ask": 1902.2, "last": 1902.0,
                "volume": 1, "flags": 0,
            },
            {
                "time": int(datetime(2024, 1, 1, 12, 0, 45, tzinfo=UTC).timestamp()),
                "bid": 1903.0, "ask": 1903.2, "last": 1903.0,
                "volume": 1, "flags": 0,
            },
        ])

        await broker._gap_fill_after_reconnect()
        for _ in range(3):
            await asyncio.sleep(0.01)

        # The second (well-formed) tick must still have been published.
        bids = sorted(evt.tick.bid for evt in captured)
        assert bids == [1903.0], (
            f"well-formed tick should survive bad sibling; got {bids}"
        )
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_unsubscribe_publishes_tick_stream_event(monkeypatch) -> None:
    """Code review #6 — broker.unsubscribe(symbol) must fire a
    ``TickStreamUnsubscribedEvent`` so MetricsCollector can prune
    watchdog state."""
    fake = _FakeMT5WithTickHistory()
    _install(monkeypatch, fake)

    bus = AsyncEventBus()
    events: list[TickStreamUnsubscribedEvent] = []

    async def collect(evt: TickStreamUnsubscribedEvent) -> None:
        events.append(evt)

    bus.subscribe(TickStreamUnsubscribedEvent, collect, name="probe.unsub")

    broker = _make_broker(bus)
    try:
        await broker.connect()
        await broker.subscribe_ticks("XAUUSD")
        broker._last_tick_time["XAUUSD"] = datetime(
            2024, 1, 1, 12, 0, 0, tzinfo=UTC,
        )

        await broker.unsubscribe("XAUUSD")
        for _ in range(3):
            await asyncio.sleep(0.01)

        assert len(events) == 1
        assert events[0].symbol == "XAUUSD"
        assert events[0].broker_name == "mt5"
        # Broker also pruned its own dedupe state.
        assert "XAUUSD" not in broker._last_tick_time
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_unsubscribe_for_non_subscribed_symbol_is_silent(monkeypatch) -> None:
    """Unsubscribing a symbol we never subscribed to must not fire a
    spurious TickStreamUnsubscribedEvent — only real stream stops
    generate the lifecycle signal."""
    fake = _FakeMT5WithTickHistory()
    _install(monkeypatch, fake)

    bus = AsyncEventBus()
    events: list[TickStreamUnsubscribedEvent] = []

    async def collect(evt: TickStreamUnsubscribedEvent) -> None:
        events.append(evt)

    bus.subscribe(TickStreamUnsubscribedEvent, collect, name="probe.unsub")

    broker = _make_broker(bus)
    try:
        await broker.connect()
        # Never subscribed.
        await broker.unsubscribe("XAUUSD")
        for _ in range(3):
            await asyncio.sleep(0.01)
        assert events == []
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_time_msc_precision_preserved(monkeypatch) -> None:
    """Code review #10 — gap-fill must use ``time_msc`` when available
    to preserve sub-second precision; multiple ticks within the same
    second must not collapse to the same timestamp."""
    fake = _FakeMT5WithTickHistory()
    _install(monkeypatch, fake)

    bus = AsyncEventBus()
    captured: list[TickEvent] = []

    async def collect(evt: TickEvent) -> None:
        captured.append(evt)

    bus.subscribe(TickEvent, collect, name="probe.tick")

    broker = _make_broker(bus)
    try:
        await broker.connect()
        await broker.subscribe_ticks("XAUUSD")
        baseline = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        broker._last_tick_time["XAUUSD"] = baseline
        # Two ticks within the same second — only ``time_msc``
        # distinguishes them.
        base_ts_sec = int(datetime(2024, 1, 1, 12, 0, 1, tzinfo=UTC).timestamp())
        fake.stage_history([
            {
                "time": base_ts_sec,
                "time_msc": base_ts_sec * 1000 + 100,
                "bid": 1902.0, "ask": 1902.2, "last": 1902.0,
                "volume": 1, "flags": 0,
            },
            {
                "time": base_ts_sec,
                "time_msc": base_ts_sec * 1000 + 700,
                "bid": 1903.0, "ask": 1903.2, "last": 1903.0,
                "volume": 1, "flags": 0,
            },
        ])

        await broker._gap_fill_after_reconnect()
        for _ in range(3):
            await asyncio.sleep(0.01)

        # Both ticks should be present with distinct sub-second times.
        times = sorted(evt.tick.time for evt in captured)
        assert len(times) == 2, (
            f"sub-second ticks should not collapse; got {times}"
        )
        assert times[0] != times[1]
        # And the microseconds should reflect the 100 ms / 700 ms split.
        assert times[0].microsecond == 100_000
        assert times[1].microsecond == 700_000
    finally:
        await broker.disconnect()
        await bus.close()


# ---------------------------------------------------------------------- #
# Round 2 regressions                                                      #
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_out_of_order_ticks_are_all_published(monkeypatch) -> None:
    """Round 2 #1 regression: gap-fill must publish every tick whose
    timestamp is strictly after the pre-disconnect baseline, regardless
    of intra-batch ordering. The previous per-iteration cutoff silently
    dropped any tick whose timestamp was older than the latest already
    published one within the same batch — a real data-loss bug when MT5
    returns ticks slightly out of monotonic order."""
    fake = _FakeMT5WithTickHistory()
    _install(monkeypatch, fake)

    bus = AsyncEventBus()
    captured: list[TickEvent] = []

    async def collect(evt: TickEvent) -> None:
        captured.append(evt)

    bus.subscribe(TickEvent, collect, name="probe.tick")

    broker = _make_broker(bus)
    try:
        await broker.connect()
        await broker.subscribe_ticks("XAUUSD")
        baseline = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        broker._last_tick_time["XAUUSD"] = baseline
        # Three ticks all AFTER baseline but in non-monotonic order.
        # Under the buggy per-iteration cutoff, the T+33 tick would be
        # dropped because T+45 had already advanced the cutoff past it.
        fake.stage_history([
            {
                "time": int(datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC).timestamp()),
                "bid": 1902.0, "ask": 1902.2, "last": 1902.0,
                "volume": 1, "flags": 0,
            },
            {
                "time": int(datetime(2024, 1, 1, 12, 0, 45, tzinfo=UTC).timestamp()),
                "bid": 1904.0, "ask": 1904.2, "last": 1904.0,
                "volume": 1, "flags": 0,
            },
            {  # out-of-order — older than the previous tick in the batch
                "time": int(datetime(2024, 1, 1, 12, 0, 33, tzinfo=UTC).timestamp()),
                "bid": 1903.0, "ask": 1903.2, "last": 1903.0,
                "volume": 1, "flags": 0,
            },
        ])

        await broker._gap_fill_after_reconnect()
        for _ in range(3):
            await asyncio.sleep(0.01)

        bids = sorted(evt.tick.bid for evt in captured)
        assert bids == [1902.0, 1903.0, 1904.0], (
            f"all three after-baseline ticks must be published "
            f"regardless of order; got {bids}"
        )
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_gap_fill_aborts_remaining_symbols_on_timeout(monkeypatch) -> None:
    """Round 2 #2 / #10 regression: when one symbol's copy_ticks_range
    times out, gap-fill must abort the WHOLE batch rather than queue
    further `_sdk(...)` calls behind the (still-blocked) executor thread.
    We prove this by making EVERY copy_ticks_range call hang and
    verifying only ONE symbol was ever queried (the first one), not
    all subscribed symbols."""
    import stinger_fx.brokers.mt5.broker as broker_mod

    fake = _FakeMT5WithTickHistory()

    queried: list[str] = []

    def _always_hangs(symbol, _start, _end, _flags):
        queried.append(symbol)
        import time as _time
        _time.sleep(5.0)  # simulate wedged SDK call
        return []

    fake.copy_ticks_range = _always_hangs  # type: ignore[method-assign]
    _install(monkeypatch, fake)
    monkeypatch.setattr(broker_mod, "GAP_FILL_PER_CALL_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(broker_mod, "GAP_FILL_OVERALL_TIMEOUT_SECONDS", 5.0)

    bus = AsyncEventBus()
    broker = _make_broker(bus)
    try:
        await broker.connect()
        # Subscribe THREE symbols. After the first one times out, the
        # outer loop must `break` — leaving the other two unattempted.
        for sym in ("XAUUSD", "EURUSD", "USDJPY"):
            await broker.subscribe_ticks(sym)
            broker._last_tick_time[sym] = datetime(
                2024, 1, 1, 12, 0, 0, tzinfo=UTC,
            )

        import time as _time
        start = _time.monotonic()
        await broker._gap_fill_after_reconnect()
        elapsed = _time.monotonic() - start

        # Per-call timeout 0.1s, abort after first → should finish in
        # well under 1 second. Without abort, it would either hang the
        # whole executor (worst case) or take 3 × 0.1s and time out per
        # symbol (still > 0.3s with overhead).
        assert elapsed < 1.0, (
            f"gap-fill should abort after first timeout; took {elapsed:.2f}s"
        )
        # The key assertion: only ONE symbol was queried — abort worked.
        # Without the `break`, all three would have been attempted (and
        # the test would observe queried == 3, each fronted by a 0.1s
        # wait_for + a stuck executor thread piling up underneath).
        assert len(queried) == 1, (
            f"gap-fill must abort after first per-call timeout; "
            f"got {len(queried)} attempts: {queried}"
        )
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_tick_equality_independent_of_replay_origin(monkeypatch) -> None:
    """Round 2 #3 — `Tick.__eq__` / `__hash__` must NOT depend on whether
    the tick was delivered live or replayed. (The `replayed` flag lives
    on TickEvent, not Tick.)"""
    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    t1 = Tick(symbol="XAUUSD", time=now, bid=1900.0, ask=1900.2)
    t2 = Tick(symbol="XAUUSD", time=now, bid=1900.0, ask=1900.2)
    assert t1 == t2, "two ticks with the same data must compare equal"
    assert hash(t1) == hash(t2), "equal ticks must hash identically"
    # And TickEvent carries replayed flag independent of the wrapped Tick.
    e_live = TickEvent(tick=t1, replayed=False)
    e_replay = TickEvent(tick=t1, replayed=True)
    assert e_live.tick == e_replay.tick  # same underlying tick
    assert e_live.replayed != e_replay.replayed
