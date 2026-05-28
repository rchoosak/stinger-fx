"""MT5Broker — auto-reconnect with exponential backoff (Phase 6.1.A).

These tests use a fake `MetaTrader5` module injected via ``sys.modules`` so
the broker's real code path runs end-to-end without the SDK installed.

The fake exposes the surface the broker actually touches: ``initialize``,
``shutdown``, ``terminal_info``, ``last_error``, ``symbol_select``, and the
``TRADE_RETCODE_DONE`` constant.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from stinger_fx.brokers.mt5.broker import MT5Broker
from stinger_fx.config.models import MT5Config
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import BrokerDisconnectedEvent, BrokerReconnectedEvent
from tests._helpers import collect_into


class _FakeMT5:
    """Minimal stand-in for the MetaTrader5 module.

    State flags control how ``initialize`` and ``terminal_info`` behave so
    tests can simulate disconnect / reconnect sequences deterministically.
    """

    # MT5 retcode constants the broker references.
    TRADE_RETCODE_DONE = 10009

    def __init__(self) -> None:
        # initialise as "connected"
        self.connected = True
        self.init_calls = 0
        self.shutdown_calls = 0
        self.symbol_select_calls: list[tuple[str, bool]] = []
        # If non-empty, terminal_info returns None for the first N probes
        # (after which it returns truthy again).
        self.terminal_info_none_count = 0
        # If non-empty, initialize fails N times before succeeding.
        self.initialize_fail_count = 0

    def initialize(self, **kwargs):
        self.init_calls += 1
        if self.initialize_fail_count > 0:
            self.initialize_fail_count -= 1
            return False
        self.connected = True
        return True

    def shutdown(self):
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

    def symbol_select(self, symbol: str, enable: bool):
        self.symbol_select_calls.append((symbol, enable))
        return True


def _install_fake_mt5(monkeypatch, fake: _FakeMT5) -> None:
    """Make `import MetaTrader5` return our fake."""
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)


@pytest.mark.asyncio
async def test_health_loop_detects_disconnect_and_reconnects(monkeypatch) -> None:
    """terminal_info() returns None → BrokerDisconnectedEvent fires,
    initialize() succeeds on retry → BrokerReconnectedEvent fires."""
    fake = _FakeMT5()
    _install_fake_mt5(monkeypatch, fake)

    bus = AsyncEventBus()
    disconnected: list[BrokerDisconnectedEvent] = []
    reconnected: list[BrokerReconnectedEvent] = []
    sub_d = bus.subscribe(BrokerDisconnectedEvent, collect_into(disconnected))
    sub_r = bus.subscribe(BrokerReconnectedEvent, collect_into(reconnected))

    cfg = MT5Config(terminal_path="", login=0, password="", server="", timeout_ms=1000)
    broker = MT5Broker(bus, cfg)
    # Speed up: very short health-check tick + tiny backoff
    broker._health_check_interval = 0.02
    broker._reconnect_backoff = [0.01]

    try:
        await broker.connect()
        assert broker._connected is True
        assert fake.init_calls == 1

        # Simulate one missed health probe — terminal_info returns None once
        fake.terminal_info_none_count = 1
        # Wait long enough for: health probe → disconnect event → reconnect attempt
        # → init succeeds → reconnect event published
        for _ in range(40):
            await asyncio.sleep(0.02)
            if reconnected:
                break

        assert len(disconnected) == 1, f"expected 1 disconnect, got {len(disconnected)}"
        assert disconnected[0].broker_name == "mt5"
        assert len(reconnected) == 1, f"expected 1 reconnect, got {len(reconnected)}"
        assert broker._connected is True
        # initialize() called twice total — once at connect, once on reconnect
        assert fake.init_calls >= 2
    finally:
        await broker.disconnect()
        await sub_d.unsubscribe()
        await sub_r.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_reconnect_backoff_handles_multiple_failures(monkeypatch) -> None:
    """When initialize() fails twice, the backoff loop keeps trying until success."""
    fake = _FakeMT5()
    _install_fake_mt5(monkeypatch, fake)

    bus = AsyncEventBus()
    reconnected: list[BrokerReconnectedEvent] = []
    sub_r = bus.subscribe(BrokerReconnectedEvent, collect_into(reconnected))

    cfg = MT5Config(terminal_path="", login=0, password="", server="", timeout_ms=1000)
    broker = MT5Broker(bus, cfg)
    broker._health_check_interval = 0.02
    broker._reconnect_backoff = [0.01]

    try:
        await broker.connect()
        # Cause one disconnect detection AND two initialize failures before success.
        fake.terminal_info_none_count = 1
        fake.initialize_fail_count = 2

        for _ in range(60):
            await asyncio.sleep(0.02)
            if reconnected:
                break

        assert len(reconnected) == 1
        # 1 (initial) + 2 (failures) + 1 (success) = 4 total
        assert fake.init_calls == 4
    finally:
        await broker.disconnect()
        await sub_r.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_reconnect_resubscribes_symbols(monkeypatch) -> None:
    """After a reconnect, every tick subscription must be re-armed via
    symbol_select() because MT5 forgets MarketWatch state across shutdown."""
    fake = _FakeMT5()
    _install_fake_mt5(monkeypatch, fake)

    bus = AsyncEventBus()
    cfg = MT5Config(terminal_path="", login=0, password="", server="", timeout_ms=1000)
    broker = MT5Broker(bus, cfg)
    broker._health_check_interval = 0.02
    broker._reconnect_backoff = [0.01]

    try:
        await broker.connect()
        # Register two symbols as subscribed (skip the thread by setting directly).
        broker._tick_subs.add("EURUSD")
        broker._tick_subs.add("GBPUSD")
        fake.symbol_select_calls.clear()

        # Trigger one disconnect → reconnect cycle.
        fake.terminal_info_none_count = 1

        for _ in range(40):
            await asyncio.sleep(0.02)
            if broker._connected and fake.symbol_select_calls:
                break

        # Both symbols should have been re-armed.
        symbols = {call[0] for call in fake.symbol_select_calls}
        assert "EURUSD" in symbols
        assert "GBPUSD" in symbols
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_disconnect_stops_health_task(monkeypatch) -> None:
    """Calling disconnect() cleanly cancels the health-check task."""
    fake = _FakeMT5()
    _install_fake_mt5(monkeypatch, fake)
    bus = AsyncEventBus()
    cfg = MT5Config(terminal_path="", login=0, password="", server="", timeout_ms=1000)
    broker = MT5Broker(bus, cfg)
    broker._health_check_interval = 5.0  # long — we don't want it to fire

    try:
        await broker.connect()
        assert broker._health_task is not None
        assert not broker._health_task.done()

        await broker.disconnect()
        assert broker._health_task is None
        assert broker._connected is False
    finally:
        await bus.close()
