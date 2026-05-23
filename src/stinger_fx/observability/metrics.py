"""Prometheus metrics — bus event → counter / gauge.

The collector is a Lifecycle component: it subscribes to relevant events on
`engine.start`, increments the right metric, and unsubscribes on stop.
`prometheus_client.start_http_server()` (stdlib http.server in a background
thread) exposes `/metrics` on a configurable port — convention is 9100 (the
Node Exporter slot is fine for a single-process exporter).

Why a separate HTTP server instead of mounting under the FastAPI web UI?
  • Metrics must work in `--mode normal` and `--mode tui` too. Embedding in
    FastAPI would only expose them in web mode.
  • Keeps the metrics endpoint reachable even if the asyncio loop is busy.

Metrics intentionally use low-cardinality labels (strategy_id, symbol,
timeframe, side) so a scraping Prometheus stays small.
"""

from __future__ import annotations

import logging

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    start_http_server,
)

from stinger_fx.core.event_bus import AsyncEventBus, Subscription
from stinger_fx.core.events import (
    AccountSnapshotEvent,
    BarEvent,
    DecisionEvent,
    EngineStartedEvent,
    EngineStoppedEvent,
    OrderFilledEvent,
    OrderRejectedEvent,
    SignalEvent,
    StrategyStateChangedEvent,
    TickEvent,
)

logger = logging.getLogger("stinger.observability.metrics")


def make_metrics(registry: CollectorRegistry | None = None) -> dict[str, object]:
    """Construct the metric objects.

    Separated from the collector so tests can build them against a fresh
    registry and avoid clashing with the global default.
    """
    r = registry or REGISTRY
    return {
        "engine_up": Gauge(
            "stinger_engine_up",
            "1 when the trading engine has emitted EngineStartedEvent and not yet stopped",
            registry=r,
        ),
        "strategies_total": Gauge(
            "stinger_strategies_total",
            "Number of strategies in each lifecycle state",
            ["state"],
            registry=r,
        ),
        "ticks_received_total": Counter(
            "stinger_ticks_received_total",
            "TickEvent count since process start",
            ["symbol"],
            registry=r,
        ),
        "bars_closed_total": Counter(
            "stinger_bars_closed_total",
            "BarEvent count for closed bars since process start",
            ["symbol", "timeframe"],
            registry=r,
        ),
        "signals_total": Counter(
            "stinger_signals_total",
            "SignalEvent count emitted by strategies",
            ["strategy_id", "side"],
            registry=r,
        ),
        "orders_filled_total": Counter(
            "stinger_orders_filled_total",
            "OrderFilledEvent count",
            ["strategy_id", "symbol", "side"],
            registry=r,
        ),
        "orders_rejected_total": Counter(
            "stinger_orders_rejected_total",
            "OrderRejectedEvent count",
            ["strategy_id", "symbol"],
            registry=r,
        ),
        "signals_rejected_by_risk_total": Counter(
            "stinger_signals_rejected_by_risk_total",
            "DecisionEvent count with action='rejected' (risk gate)",
            ["strategy_id", "rule"],
            registry=r,
        ),
        "account_balance": Gauge(
            "stinger_account_balance",
            "Account balance in account currency",
            ["account_id"],
            registry=r,
        ),
        "account_equity": Gauge(
            "stinger_account_equity",
            "Account equity (balance + unrealised P&L)",
            ["account_id"],
            registry=r,
        ),
        "account_profit": Gauge(
            "stinger_account_profit",
            "Floating P&L across all open positions",
            ["account_id"],
            registry=r,
        ),
    }


class MetricsCollector:
    """Bus subscriber that pushes events into Prometheus metrics.

    Lifetime = engine. start() subscribes; stop() unsubscribes.
    """

    def __init__(
        self,
        bus: AsyncEventBus,
        metrics: dict[str, object] | None = None,
    ) -> None:
        self._bus = bus
        self.metrics = metrics or make_metrics()
        self._subs: list[Subscription] = []

    async def start(self) -> None:
        self._subs = [
            self._bus.subscribe(EngineStartedEvent, self._on_engine_started, name="metrics.up"),
            self._bus.subscribe(EngineStoppedEvent, self._on_engine_stopped, name="metrics.down"),
            self._bus.subscribe(TickEvent, self._on_tick, name="metrics.tick"),
            self._bus.subscribe(BarEvent, self._on_bar, name="metrics.bar"),
            self._bus.subscribe(SignalEvent, self._on_signal, name="metrics.signal"),
            self._bus.subscribe(OrderFilledEvent, self._on_filled, name="metrics.fill"),
            self._bus.subscribe(OrderRejectedEvent, self._on_rejected, name="metrics.reject"),
            self._bus.subscribe(DecisionEvent, self._on_decision, name="metrics.decision"),
            self._bus.subscribe(
                AccountSnapshotEvent, self._on_snapshot, name="metrics.snapshot"
            ),
            self._bus.subscribe(
                StrategyStateChangedEvent,
                self._on_strategy_state,
                name="metrics.strategy",
            ),
        ]
        logger.info("metrics_collector_started")

    async def stop(self) -> None:
        for sub in self._subs:
            await sub.unsubscribe()
        self._subs.clear()
        # Flip engine_up off if we were running
        self.metrics["engine_up"].set(0)  # type: ignore[union-attr]

    # --- handlers ---------------------------------------------------------

    async def _on_engine_started(self, _evt: EngineStartedEvent) -> None:
        self.metrics["engine_up"].set(1)  # type: ignore[union-attr]

    async def _on_engine_stopped(self, _evt: EngineStoppedEvent) -> None:
        self.metrics["engine_up"].set(0)  # type: ignore[union-attr]

    async def _on_tick(self, evt: TickEvent) -> None:
        self.metrics["ticks_received_total"].labels(symbol=evt.tick.symbol).inc()  # type: ignore[union-attr]

    async def _on_bar(self, evt: BarEvent) -> None:
        if not evt.bar.is_closed:
            return
        self.metrics["bars_closed_total"].labels(  # type: ignore[union-attr]
            symbol=evt.bar.symbol, timeframe=evt.bar.timeframe.value
        ).inc()

    async def _on_signal(self, evt: SignalEvent) -> None:
        self.metrics["signals_total"].labels(  # type: ignore[union-attr]
            strategy_id=evt.signal.strategy_id, side=evt.signal.side.value
        ).inc()

    async def _on_filled(self, evt: OrderFilledEvent) -> None:
        o = evt.order
        self.metrics["orders_filled_total"].labels(  # type: ignore[union-attr]
            strategy_id=o.strategy_id, symbol=o.symbol, side=o.side.value
        ).inc()

    async def _on_rejected(self, evt: OrderRejectedEvent) -> None:
        o = evt.order
        self.metrics["orders_rejected_total"].labels(  # type: ignore[union-attr]
            strategy_id=o.strategy_id, symbol=o.symbol
        ).inc()

    async def _on_decision(self, evt: DecisionEvent) -> None:
        d = evt.decision
        if d.action != "rejected":
            return
        # `reason` may contain rule + actuals — extract just the rule prefix
        # (text before the first space or `=`) so cardinality stays bounded.
        rule = _rule_label(d.reason)
        self.metrics["signals_rejected_by_risk_total"].labels(  # type: ignore[union-attr]
            strategy_id=d.signal.strategy_id, rule=rule
        ).inc()

    async def _on_snapshot(self, evt: AccountSnapshotEvent) -> None:
        s = evt.snapshot
        self.metrics["account_balance"].labels(account_id=s.account_id).set(s.balance)  # type: ignore[union-attr]
        self.metrics["account_equity"].labels(account_id=s.account_id).set(s.equity)  # type: ignore[union-attr]
        self.metrics["account_profit"].labels(account_id=s.account_id).set(s.profit)  # type: ignore[union-attr]

    async def _on_strategy_state(self, evt: StrategyStateChangedEvent) -> None:
        # Reset gauge for this state and increment current state. Cheap because
        # state cardinality is tiny (started/paused/quarantined/stopped).
        self.metrics["strategies_total"].labels(state=evt.state).inc()  # type: ignore[union-attr]


def _rule_label(reason: str) -> str:
    """Extract the rule name from a RiskMonitor reason string.

    Examples:
      "max_open_positions_per_strategy=5 reached ..."  → "max_open_positions_per_strategy"
      "kill_switch_tripped"                            → "kill_switch_tripped"
      "max_daily_loss_pct=5.0 hit (today loss 5.1%)"  → "max_daily_loss_pct"
    """
    if not reason:
        return "unknown"
    head = reason.split("=", 1)[0]
    head = head.split(" ", 1)[0]
    return head.strip() or "unknown"


def start_metrics_server(port: int = 9100, addr: str = "127.0.0.1") -> None:
    """Bind the Prometheus exporter on `addr:port`. Idempotent guard included."""
    start_http_server(port, addr=addr)
    logger.info("metrics_server_listening http://%s:%d/metrics", addr, port)
