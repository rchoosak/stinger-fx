"""Regression tests for the MT5 SDK serialization hotfix.

Pre-fix bug: ``MT5Broker._tick_pump()`` ran in a separate daemon thread
and called ``mt5.symbol_info_tick()`` directly. That bypassed the
single-worker ``_executor`` that serializes every other SDK call
(``order_send``, ``account_info``, ``copy_rates_range``, ``terminal_info``,
``shutdown``, …). The MetaTrader5 SDK is documented as **synchronous and
not thread-safe**, so any concurrent SDK access from the daemon thread
+ the executor thread is a data race under live load.

The hotfix moves the tick poller onto an asyncio task that calls
``symbol_info_tick`` through ``_sdk()``. After the fix, **every** SDK
call funnels through the single executor worker, so the SDK is only ever
entered from one thread.

These tests pin that invariant:

  1. ``symbol_info_tick`` records the calling thread; concurrent
     ``account_info`` / ``order_send`` calls record their threads too.
     **All recorded threads must be the same** — the executor's single
     worker.
  2. The tick poller is an asyncio task (``_tick_task``), not a daemon
     ``threading.Thread`` — the bug was structural and the data-type
     change is the structural fix.
  3. ``disconnect()`` cancels the asyncio task cleanly.
  4. While ``_connected`` is False (reconnect in progress) the loop does
     not poll the SDK — same gating as the pre-fix thread.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types

import pytest

from stinger_fx.brokers.mt5.broker import MT5Broker
from stinger_fx.config.models import MT5Config
from stinger_fx.core import AsyncEventBus


class _ThreadRecordingMT5:
    """Fake MetaTrader5 that records which thread each SDK call ran on.

    Used to assert the single-thread serialization invariant: every method
    invocation must come from the same thread (the executor's worker).
    """

    TRADE_RETCODE_DONE = 10009
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_FILLING_IOC = 1
    ORDER_TIME_GTC = 0

    def __init__(self) -> None:
        self.threads_seen: set[int] = set()
        self.call_log: list[tuple[str, int]] = []
        self.connected = True
        self._tick_seq = 0
        # Per-symbol "current" tick — increment time so the broker treats
        # each poll as a new tick and publishes it.
        self._tick_time = 1_700_000_000

    # --- bookkeeping ----------------------------------------------------

    def _record(self, name: str) -> None:
        tid = threading.get_ident()
        self.threads_seen.add(tid)
        self.call_log.append((name, tid))

    # --- lifecycle ------------------------------------------------------

    def initialize(self, **_kwargs) -> bool:
        self._record("initialize")
        self.connected = True
        return True

    def shutdown(self) -> None:
        self._record("shutdown")
        self.connected = False

    def terminal_info(self):
        self._record("terminal_info")
        return types.SimpleNamespace(connected=True) if self.connected else None

    def last_error(self):
        return (0, "no error")

    def symbol_select(self, _symbol: str, _enable: bool) -> bool:
        self._record("symbol_select")
        return True

    # --- ticks ----------------------------------------------------------

    def symbol_info_tick(self, _symbol: str):
        self._record("symbol_info_tick")
        # Force a brand-new timestamp every call so the broker treats it as
        # a new tick worth publishing.
        self._tick_time += 1
        return types.SimpleNamespace(
            time=self._tick_time,
            bid=1.1000,
            ask=1.1002,
            last=1.1001,
            volume=1,
            flags=0,
        )

    # --- account / orders ----------------------------------------------

    def account_info(self):
        self._record("account_info")
        return types.SimpleNamespace(
            login=12345,
            company="Test",
            server="Demo",
            currency="USD",
            leverage=100,
            name="Tester",
            balance=10_000.0,
            equity=10_000.0,
            margin=0.0,
            margin_free=10_000.0,
            profit=0.0,
        )

    def order_send(self, _request):
        self._record("order_send")
        return types.SimpleNamespace(
            retcode=self.TRADE_RETCODE_DONE,
            order=42,
            volume=0.1,
            price=1.1001,
            comment="ok",
        )


def _install(monkeypatch, fake) -> None:
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)


def _make_broker(bus, *, tick_interval: float = 0.001) -> MT5Broker:
    cfg = MT5Config(terminal_path="", login=0, password="", server="", timeout_ms=1000)
    broker = MT5Broker(bus, cfg)
    # Keep tests fast: don't let the health task fire during the run, and
    # poll ticks aggressively.
    broker._health_check_interval = 60.0
    broker._tick_poll_interval = tick_interval
    return broker


# --- 1. The core invariant: single-thread SDK access -----------------------


@pytest.mark.asyncio
async def test_tick_polls_run_on_executor_thread_not_daemon(monkeypatch) -> None:
    """Regression: ``symbol_info_tick`` must run on the executor's worker
    thread — the same thread every other SDK call runs on.

    Pre-fix, the tick pump was a separate daemon thread, so
    ``symbol_info_tick`` ran on a different ``threading.get_ident()`` than
    ``account_info`` / ``order_send`` — a data race against the
    not-thread-safe SDK.
    """
    fake = _ThreadRecordingMT5()
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)
    try:
        await broker.connect()
        await broker.subscribe_ticks("EURUSD")
        # Let the tick task spin enough iterations to call symbol_info_tick.
        for _ in range(20):
            await asyncio.sleep(0.005)
            if any(name == "symbol_info_tick" for name, _ in fake.call_log):
                break
        # Now exercise the other SDK paths concurrently while the tick
        # loop is running — these must land on the same thread.
        await broker.get_account_info()
        await broker.get_account_snapshot()
        # Let a few more ticks fly.
        for _ in range(5):
            await asyncio.sleep(0.005)

        # Filter to actual SDK methods we care about (skip pre-connect
        # bookkeeping).
        relevant = {
            name for name, _ in fake.call_log
            if name in {
                "symbol_info_tick", "account_info", "initialize",
                "terminal_info", "shutdown", "symbol_select",
            }
        }
        # Sanity: the tick loop actually ran.
        assert "symbol_info_tick" in relevant, (
            f"tick loop didn't fire — call_log={fake.call_log}"
        )
        # Sanity: the account path actually ran.
        assert "account_info" in relevant

        # THE invariant: only ONE thread ever called into the SDK.
        assert len(fake.threads_seen) == 1, (
            f"SDK was entered from {len(fake.threads_seen)} threads — "
            f"pre-fix bug regressed. threads={fake.threads_seen} "
            f"calls={fake.call_log}"
        )
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_sdk_thread_is_not_the_loop_thread(monkeypatch) -> None:
    """Sanity check: the executor worker is a different thread from the
    asyncio loop thread. (Confirms the executor is actually doing work —
    if it weren't, ``threads_seen`` would equal the loop thread and the
    invariant test would still pass trivially.)"""
    fake = _ThreadRecordingMT5()
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)
    loop_thread = threading.get_ident()
    try:
        await broker.connect()
        await broker.subscribe_ticks("EURUSD")
        for _ in range(20):
            await asyncio.sleep(0.005)
            if fake.threads_seen:
                break
        assert fake.threads_seen, "no SDK calls recorded"
        sdk_thread = next(iter(fake.threads_seen))
        assert sdk_thread != loop_thread, (
            "SDK ran on the asyncio loop thread — executor is not in use; "
            "the serialization invariant would pass vacuously."
        )
    finally:
        await broker.disconnect()
        await bus.close()


# --- 2. Structural: tick pump is an asyncio task, not a daemon thread -----


@pytest.mark.asyncio
async def test_tick_poller_is_asyncio_task_not_thread(monkeypatch) -> None:
    """The structural fix: the tick poller must be an ``asyncio.Task``,
    not a ``threading.Thread``. Pre-fix, ``_tick_thread`` was a daemon
    ``threading.Thread`` — preserving the asyncio-task structure prevents
    a regression that re-introduces the parallel-thread bug."""
    fake = _ThreadRecordingMT5()
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)
    try:
        await broker.connect()
        await broker.subscribe_ticks("EURUSD")
        # Public-ish surface: _tick_task is set, _tick_thread does not exist.
        assert hasattr(broker, "_tick_task"), "missing _tick_task attribute"
        assert broker._tick_task is not None
        assert isinstance(broker._tick_task, asyncio.Task), (
            f"_tick_task must be an asyncio.Task, got {type(broker._tick_task)}"
        )
        assert not hasattr(broker, "_tick_thread"), (
            "_tick_thread attribute still exists — the daemon-thread "
            "model is the bug. Use the asyncio task instead."
        )
    finally:
        await broker.disconnect()
        await bus.close()


# --- 3. Clean shutdown ----------------------------------------------------


@pytest.mark.asyncio
async def test_disconnect_cancels_tick_task(monkeypatch) -> None:
    """``disconnect()`` must cancel the tick task and clear the reference.
    Pre-fix this was a thread join with a 2 s timeout; with an asyncio
    task we use proper cancellation semantics."""
    fake = _ThreadRecordingMT5()
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)
    await broker.connect()
    await broker.subscribe_ticks("EURUSD")
    # Make sure the task is running.
    assert broker._tick_task is not None
    assert not broker._tick_task.done()

    await broker.disconnect()

    # Task should be cleared and either cancelled or finished.
    assert broker._tick_task is None
    await bus.close()


# --- 4. Reconnect gating --------------------------------------------------


@pytest.mark.asyncio
async def test_tick_loop_pauses_while_disconnected(monkeypatch) -> None:
    """While ``_connected`` is False (reconnect in progress), the tick
    loop must NOT poll the SDK — spinning calls into a wedged SDK is
    what the pre-fix daemon thread did too, and we preserve that gating
    in the asyncio version.
    """
    fake = _ThreadRecordingMT5()
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus, tick_interval=0.001)
    # Tighten the reconnect-pause sleep so the test can observe it
    # without wasting wall-clock time.
    broker._health_check_interval = 0.05
    try:
        await broker.connect()
        await broker.subscribe_ticks("EURUSD")
        # Spin once so the loop is past initial setup.
        await asyncio.sleep(0.01)

        # Simulate a disconnect (the health task would do this in prod).
        broker._connected = False
        # Snapshot the call count after the gating takes effect.
        await asyncio.sleep(0.02)  # let the task notice the flag
        baseline = sum(1 for n, _ in fake.call_log if n == "symbol_info_tick")
        # Now wait significantly longer than tick_poll_interval (0.001s).
        # If the loop were still polling, baseline would grow a lot.
        await asyncio.sleep(0.04)
        after = sum(1 for n, _ in fake.call_log if n == "symbol_info_tick")
        # Allow at most 1 extra call (the one in flight when we flipped).
        assert after - baseline <= 1, (
            f"tick loop kept polling while disconnected: "
            f"baseline={baseline} after={after}"
        )
    finally:
        broker._connected = True  # so disconnect can cleanly shutdown
        await broker.disconnect()
        await bus.close()
