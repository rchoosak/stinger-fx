"""MT5Broker — SDK-call latency telemetry (Phase 6.1.B).

When the broker is given a `metrics` dict, every call routed through `_sdk`
records its elapsed time in the `mt5_call_seconds` histogram labelled by
SDK method name. Verifies the hook is wired up and is best-effort (a
broken metric must not crash the trading path).
"""

from __future__ import annotations

import sys
import types

import pytest
from prometheus_client import CollectorRegistry

from stinger_fx.brokers.mt5.broker import MT5Broker
from stinger_fx.config.models import MT5Config
from stinger_fx.core import AsyncEventBus
from stinger_fx.observability.metrics import make_metrics


class _FakeMT5:
    TRADE_RETCODE_DONE = 10009

    def __init__(self) -> None:
        self.connected = True

    def initialize(self, **_):
        return True

    def shutdown(self):
        self.connected = False

    def terminal_info(self):
        return types.SimpleNamespace(connected=True)

    def last_error(self):
        return (0, "no error")

    def account_info(self):
        return types.SimpleNamespace(
            login=12345, company="Demo", server="DemoSrv", currency="USD",
            leverage=100, name="x", balance=10_000, equity=10_000,
            margin=0, margin_free=10_000, profit=0,
        )

    def symbol_select(self, _symbol, _enable):
        return True


def _histogram_count(metric, **labels) -> float:
    name = metric._name + "_count"
    for sample in metric.collect()[0].samples:
        if sample.name == name and sample.labels == labels:
            return sample.value
    return 0.0


@pytest.mark.asyncio
async def test_sdk_call_records_latency_per_method(monkeypatch) -> None:
    """Every _sdk(fn, ...) call must record into mt5_call_seconds labelled by
    fn.__name__ when a metrics dict is provided."""
    fake = _FakeMT5()
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)

    bus = AsyncEventBus()
    metrics = make_metrics(registry=CollectorRegistry())
    cfg = MT5Config(terminal_path="", login=0, password="", server="", timeout_ms=1000)
    broker = MT5Broker(bus, cfg, metrics=metrics)
    broker._health_check_interval = 60.0

    try:
        await broker.connect()
        await broker.get_account_info()

        # `account_info` is called via `mt5.account_info` directly so its
        # __name__ is captured. `initialize` is called via a local `_do_init`
        # closure (so __name__ == "_do_init") — we check for both.
        hist = metrics["mt5_call_seconds"]
        assert _histogram_count(hist, method="account_info") >= 1
        assert _histogram_count(hist, method="_do_init") >= 1
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_sdk_calls_work_without_metrics(monkeypatch) -> None:
    """When metrics=None (default), broker must work without any histogram side-effect."""
    fake = _FakeMT5()
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)

    bus = AsyncEventBus()
    cfg = MT5Config(terminal_path="", login=0, password="", server="", timeout_ms=1000)
    broker = MT5Broker(bus, cfg)  # no metrics
    broker._health_check_interval = 60.0

    try:
        await broker.connect()
        info = await broker.get_account_info()
        assert info.account_id == "12345"
    finally:
        await broker.disconnect()
        await bus.close()
