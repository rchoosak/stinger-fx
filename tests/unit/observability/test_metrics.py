"""MetricsCollector — bus event → Prometheus counter / gauge."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from prometheus_client import CollectorRegistry

from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import (
    AccountSnapshotEvent,
    BarEvent,
    DecisionEvent,
    EngineStartedEvent,
    EngineStoppedEvent,
    OrderFilledEvent,
    SignalEvent,
    StrategyStateChangedEvent,
    TickEvent,
)
from stinger_fx.domain import (
    AccountSnapshot,
    Bar,
    Decision,
    Order,
    OrderStatus,
    OrderType,
    Side,
    Signal,
    SignalStrength,
    Tick,
    Timeframe,
)
from stinger_fx.observability.metrics import MetricsCollector, _rule_label, make_metrics


def _signal(strategy_id: str = "s1", side: Side = Side.BUY) -> Signal:
    return Signal(
        strategy_id=strategy_id,
        time=datetime.now(UTC),
        symbol="EURUSD",
        side=side,
        strength=SignalStrength.NORMAL,
    )


@pytest.fixture
def collector() -> tuple[AsyncEventBus, MetricsCollector, CollectorRegistry]:
    """Build a collector against a fresh registry so tests don't clash on the
    global default registry."""
    bus = AsyncEventBus()
    registry = CollectorRegistry()
    metrics = make_metrics(registry)
    return bus, MetricsCollector(bus, metrics=metrics), registry


def _value(metric, **labels) -> float:
    """Look up the current value of a labelled metric. Skips the `_created`
    sidecar samples that Prometheus emits next to every Counter (they're
    timestamps, not values)."""
    sample_value = 0.0
    family = next(iter(metric.collect()))
    for sample in family.samples:
        if sample.name.endswith("_created"):
            continue
        if all(sample.labels.get(k) == v for k, v in labels.items()):
            sample_value = sample.value
    return sample_value


@pytest.mark.asyncio
async def test_engine_up_gauge_toggles(collector) -> None:
    bus, mc, _ = collector
    await mc.start()
    assert _value(mc.metrics["engine_up"]) == 0
    await bus.publish(EngineStartedEvent())
    await asyncio.sleep(0.05)
    assert _value(mc.metrics["engine_up"]) == 1
    await bus.publish(EngineStoppedEvent())
    await asyncio.sleep(0.05)
    assert _value(mc.metrics["engine_up"]) == 0
    await mc.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_tick_counter_increments_per_symbol(collector) -> None:
    bus, mc, _ = collector
    await mc.start()
    for _ in range(3):
        await bus.publish(
            TickEvent(
                tick=Tick(symbol="EURUSD", time=datetime.now(UTC), bid=1.1, ask=1.1001)
            )
        )
    await bus.publish(
        TickEvent(tick=Tick(symbol="GBPUSD", time=datetime.now(UTC), bid=1.27, ask=1.2702))
    )
    await asyncio.sleep(0.05)
    assert _value(mc.metrics["ticks_received_total"], symbol="EURUSD") == 3.0
    assert _value(mc.metrics["ticks_received_total"], symbol="GBPUSD") == 1.0
    await mc.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_closed_bars_counter(collector) -> None:
    bus, mc, _ = collector
    await mc.start()
    await bus.publish(
        BarEvent(
            bar=Bar(
                symbol="EURUSD",
                timeframe=Timeframe.M15,
                time=datetime.now(UTC),
                open=1.10, high=1.11, low=1.09, close=1.105,
                is_closed=True,
            )
        )
    )
    # An in-progress bar must NOT bump the counter
    await bus.publish(
        BarEvent(
            bar=Bar(
                symbol="EURUSD",
                timeframe=Timeframe.M15,
                time=datetime.now(UTC),
                open=1.10, high=1.11, low=1.09, close=1.105,
                is_closed=False,
            )
        )
    )
    await asyncio.sleep(0.05)
    assert _value(mc.metrics["bars_closed_total"], symbol="EURUSD", timeframe="M15") == 1.0
    await mc.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_signal_and_fill_counters(collector) -> None:
    bus, mc, _ = collector
    await mc.start()
    await bus.publish(SignalEvent(signal=_signal("ma_test", Side.BUY)))
    await bus.publish(SignalEvent(signal=_signal("ma_test", Side.BUY)))
    await bus.publish(SignalEvent(signal=_signal("ma_test", Side.SELL)))
    await bus.publish(
        OrderFilledEvent(
            order=Order(
                ticket=1, strategy_id="ma_test", symbol="EURUSD",
                side=Side.BUY, type=OrderType.MARKET, volume=0.1,
                status=OrderStatus.FILLED,
            )
        )
    )
    await asyncio.sleep(0.05)
    assert _value(mc.metrics["signals_total"], strategy_id="ma_test", side="buy") == 2.0
    assert _value(mc.metrics["signals_total"], strategy_id="ma_test", side="sell") == 1.0
    assert (
        _value(
            mc.metrics["orders_filled_total"],
            strategy_id="ma_test", symbol="EURUSD", side="buy",
        )
        == 1.0
    )
    await mc.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_signal_rejected_by_risk_uses_rule_label(collector) -> None:
    bus, mc, _ = collector
    await mc.start()
    sig = _signal("ma_test")
    decision = Decision(
        signal=sig,
        time=sig.time,
        action="rejected",
        reason="max_open_positions_per_strategy=5 reached (strategy ma_test has 5 open)",
        risk_check_passed=False,
    )
    await bus.publish(DecisionEvent(decision=decision))
    await asyncio.sleep(0.05)
    assert (
        _value(
            mc.metrics["signals_rejected_by_risk_total"],
            strategy_id="ma_test",
            rule="max_open_positions_per_strategy",
        )
        == 1.0
    )
    # A non-rejected decision should NOT increment the counter
    await bus.publish(
        DecisionEvent(
            decision=Decision(
                signal=sig, time=sig.time, action="placed", reason="",
            )
        )
    )
    await asyncio.sleep(0.05)
    assert (
        _value(
            mc.metrics["signals_rejected_by_risk_total"],
            strategy_id="ma_test",
            rule="max_open_positions_per_strategy",
        )
        == 1.0
    )
    await mc.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_account_gauges_track_latest_snapshot(collector) -> None:
    bus, mc, _ = collector
    await mc.start()
    await bus.publish(
        AccountSnapshotEvent(
            snapshot=AccountSnapshot(
                account_id="acc1", time=datetime.now(UTC),
                balance=10_000, equity=10_120, margin=0, free_margin=10_120,
                profit=120,
            )
        )
    )
    await asyncio.sleep(0.05)
    assert _value(mc.metrics["account_balance"], account_id="acc1") == 10_000
    assert _value(mc.metrics["account_equity"], account_id="acc1") == 10_120
    assert _value(mc.metrics["account_profit"], account_id="acc1") == 120
    # A second snapshot must overwrite (gauge, not counter)
    await bus.publish(
        AccountSnapshotEvent(
            snapshot=AccountSnapshot(
                account_id="acc1", time=datetime.now(UTC),
                balance=10_000, equity=9_900, margin=0, free_margin=9_900,
                profit=-100,
            )
        )
    )
    await asyncio.sleep(0.05)
    assert _value(mc.metrics["account_equity"], account_id="acc1") == 9_900
    assert _value(mc.metrics["account_profit"], account_id="acc1") == -100
    await mc.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_strategy_state_counter(collector) -> None:
    bus, mc, _ = collector
    await mc.start()
    await bus.publish(StrategyStateChangedEvent(strategy_id="s1", state="started"))
    await bus.publish(StrategyStateChangedEvent(strategy_id="s2", state="started"))
    await bus.publish(StrategyStateChangedEvent(strategy_id="s1", state="paused"))
    await asyncio.sleep(0.05)
    assert _value(mc.metrics["strategies_total"], state="started") == 2.0
    assert _value(mc.metrics["strategies_total"], state="paused") == 1.0
    await mc.stop()
    await bus.close()


# --- Rule-label extractor ----------------------------------------------------


def test_rule_label_extracts_short_name() -> None:
    assert (
        _rule_label("max_open_positions_per_strategy=5 reached ...")
        == "max_open_positions_per_strategy"
    )
    assert _rule_label("kill_switch_tripped") == "kill_switch_tripped"
    assert _rule_label("max_daily_loss_pct=5.0 hit (loss 5.1%)") == "max_daily_loss_pct"
    assert _rule_label("") == "unknown"
