"""MetricsCollector — Phase 6.1.B latency histograms + broker connectivity counters.

Uses a fresh `CollectorRegistry` per test so the global default registry
isn't polluted across tests.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from prometheus_client import CollectorRegistry

from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import (
    BrokerDisconnectedEvent,
    BrokerReconnectedEvent,
    OrderFilledEvent,
    TickEvent,
)
from stinger_fx.domain import Order, OrderStatus, OrderType, Side, Tick
from stinger_fx.observability.metrics import MetricsCollector, make_metrics


def _metric_value(metric, **labels) -> float:
    """Return the current sample value for a labelled metric."""
    for sample in metric.collect()[0].samples:
        if sample.labels == labels and (
            sample.name.endswith("_count") or sample.name.endswith("_total")
            or sample.name == metric._name
        ):
            return sample.value
    return 0.0


def _histogram_count(metric, **labels) -> float:
    """Read the _count suffix for a Histogram."""
    name = metric._name + "_count"
    for sample in metric.collect()[0].samples:
        if sample.name == name and sample.labels == labels:
            return sample.value
    return 0.0


def _histogram_sum(metric, **labels) -> float:
    name = metric._name + "_sum"
    for sample in metric.collect()[0].samples:
        if sample.name == name and sample.labels == labels:
            return sample.value
    return 0.0


async def _drain(bus: AsyncEventBus, *, ticks: int = 3) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_order_filled_records_submission_latency() -> None:
    """OrderFilledEvent with requested_at + filled_at must increment the
    submission-latency histogram with the correct delta."""
    bus = AsyncEventBus()
    metrics = make_metrics(registry=CollectorRegistry())
    collector = MetricsCollector(bus, metrics=metrics)
    await collector.start()

    try:
        requested = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        filled = requested + timedelta(milliseconds=125)
        order = Order(
            ticket=1,
            strategy_id="s",
            symbol="EURUSD",
            side=Side.BUY,
            type=OrderType.MARKET,
            volume=0.1,
            filled_volume=0.1,
            fill_price=1.10,
            status=OrderStatus.FILLED,
            requested_at=requested,
            filled_at=filled,
        )
        await bus.publish(OrderFilledEvent(order=order))
        await _drain(bus)

        hist = metrics["order_submission_seconds"]
        count = _histogram_count(hist, strategy_id="s", symbol="EURUSD")
        total = _histogram_sum(hist, strategy_id="s", symbol="EURUSD")
        assert count == 1
        assert total == pytest.approx(0.125, abs=0.001)
    finally:
        await collector.stop()
        await bus.close()


@pytest.mark.asyncio
async def test_tick_event_records_e2e_lag_and_gauge() -> None:
    """TickEvent must populate both tick_e2e_seconds histogram + tick_pump_lag_seconds gauge."""
    bus = AsyncEventBus()
    metrics = make_metrics(registry=CollectorRegistry())
    collector = MetricsCollector(bus, metrics=metrics)
    await collector.start()

    try:
        # Tick stamped 200ms in the past — collector observes it ~now.
        tick_time = datetime.now(UTC) - timedelta(milliseconds=200)
        tick = Tick(symbol="EURUSD", time=tick_time, bid=1.10, ask=1.1002)
        await bus.publish(TickEvent(tick=tick))
        await _drain(bus)

        hist = metrics["tick_e2e_seconds"]
        gauge = metrics["tick_pump_lag_seconds"]
        assert _histogram_count(hist, symbol="EURUSD") == 1
        # Lag should be ≥ 200ms (with a generous upper bound for test scheduler noise)
        lag_sum = _histogram_sum(hist, symbol="EURUSD")
        assert 0.18 <= lag_sum <= 5.0, f"unexpected lag: {lag_sum}"
        # Gauge holds the last observed value
        gauge_value = _metric_value(gauge, symbol="EURUSD")
        assert 0.18 <= gauge_value <= 5.0
    finally:
        await collector.stop()
        await bus.close()


@pytest.mark.asyncio
async def test_broker_disconnect_increments_counter() -> None:
    """BrokerDisconnectedEvent must bump the disconnects counter."""
    bus = AsyncEventBus()
    metrics = make_metrics(registry=CollectorRegistry())
    collector = MetricsCollector(bus, metrics=metrics)
    await collector.start()

    try:
        await bus.publish(BrokerDisconnectedEvent(broker_name="mt5", reason="network blip"))
        await _drain(bus)
        await bus.publish(BrokerDisconnectedEvent(broker_name="mt5", reason="another"))
        await _drain(bus)

        counter = metrics["broker_disconnects_total"]
        assert _metric_value(counter, broker="mt5") == 2
    finally:
        await collector.stop()
        await bus.close()


@pytest.mark.asyncio
async def test_broker_reconnect_records_downtime_histogram() -> None:
    """BrokerReconnectedEvent must increment the reconnects counter AND
    record downtime_seconds in the downtime histogram."""
    bus = AsyncEventBus()
    metrics = make_metrics(registry=CollectorRegistry())
    collector = MetricsCollector(bus, metrics=metrics)
    await collector.start()

    try:
        await bus.publish(BrokerReconnectedEvent(broker_name="mt5", downtime_seconds=42.5))
        await _drain(bus)

        counter = metrics["broker_reconnects_total"]
        hist = metrics["broker_downtime_seconds"]
        assert _metric_value(counter, broker="mt5") == 1
        assert _histogram_count(hist, broker="mt5") == 1
        assert _histogram_sum(hist, broker="mt5") == pytest.approx(42.5)
    finally:
        await collector.stop()
        await bus.close()


@pytest.mark.asyncio
async def test_skips_latency_when_timestamps_missing() -> None:
    """Order with no requested_at must not error and must not record into histogram."""
    bus = AsyncEventBus()
    metrics = make_metrics(registry=CollectorRegistry())
    collector = MetricsCollector(bus, metrics=metrics)
    await collector.start()

    try:
        order = Order(
            ticket=1,
            strategy_id="s",
            symbol="EURUSD",
            side=Side.BUY,
            type=OrderType.MARKET,
            volume=0.1,
            status=OrderStatus.FILLED,
            requested_at=None,
            filled_at=datetime.now(UTC),
        )
        await bus.publish(OrderFilledEvent(order=order))
        await _drain(bus)

        hist = metrics["order_submission_seconds"]
        assert _histogram_count(hist, strategy_id="s", symbol="EURUSD") == 0
    finally:
        await collector.stop()
        await bus.close()
