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
)


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
