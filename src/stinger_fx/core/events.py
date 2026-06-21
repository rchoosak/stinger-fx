"""Typed events that flow through the AsyncEventBus.

Subscribers register against a concrete subclass of `Event`; the bus dispatches
by `isinstance` so subscribing to a parent type catches all children.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from stinger_fx.domain import (
    AccountSnapshot,
    Bar,
    Decision,
    Order,
    OrderRequest,
    Position,
    Signal,
    Tick,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Event(BaseModel):
    """Base class for all events. Subclass and add payload fields."""

    model_config = ConfigDict(frozen=True)

    ts: datetime = Field(default_factory=_utcnow)   # event emission time (UTC)


# --- Market data --------------------------------------------------------------


class TickEvent(Event):
    tick: Tick
    # True if this delivery is a gap-fill replay (broker re-publishing
    # historical ticks after a reconnect), not a live observation.
    # Consumers like MetricsCollector skip lag/watchdog updates for these
    # so historical timestamps don't pollute "is the live stream healthy?"
    # signals. Kept on the event (delivery metadata), not on Tick (data),
    # so two ticks with identical fields stay equal regardless of origin.
    replayed: bool = False


class BarEvent(Event):
    bar: Bar


# --- Strategy / order flow ----------------------------------------------------


class SignalEvent(Event):
    signal: Signal


class DecisionEvent(Event):
    """Emitted by the OrderRouter for every signal it sees (accepted or not)."""

    decision: Decision


class OrderRequestEvent(Event):
    request: OrderRequest


class OrderSubmittedEvent(Event):
    order: Order


class OrderFilledEvent(Event):
    order: Order


class OrderRejectedEvent(Event):
    order: Order
    reason: str


class OrderCancelledEvent(Event):
    order: Order


class PositionOpenedEvent(Event):
    position: Position


class PositionUpdatedEvent(Event):
    position: Position


class PositionClosedEvent(Event):
    position: Position
    realized_pnl: float


# --- Order management (Phase 4 — modify / partial close) ----------------------


class ModifyOrderRequestEvent(Event):
    """Strategy → router request to modify an existing order.

    For open positions the relevant fields are ``sl`` / ``tp`` (price /
    volume / stop_price are ignored — you can't change a position's
    entry price post-fill).

    For pending orders (Phase 6.2.D) ``price`` / ``volume`` /
    ``stop_price`` adjust the unfilled order. ``sl`` / ``tp`` work too
    (they'll attach to the eventual position when the order triggers).

    Every field is independent: ``None`` means "leave unchanged".
    """

    strategy_id: str
    ticket: int
    sl: float | None = None
    tp: float | None = None
    price: float | None = None
    stop_price: float | None = None
    volume: float | None = None
    reason: str = ""


class PartialCloseRequestEvent(Event):
    """Strategy → router request to reduce an existing position by `volume`."""

    strategy_id: str
    ticket: int
    volume: float = Field(gt=0)
    reason: str = ""


class OrderModifiedEvent(Event):
    """Broker confirmed an SL / TP modification.

    `position` is the *updated* snapshot — callers should treat the previous
    position object as stale.
    """

    position: Position
    reason: str = ""


class PartialClosedEvent(Event):
    """Broker confirmed a partial close — `position` is the *remaining* leg
    after `closed_volume` was reduced; `realized_pnl` is the P&L of the chunk
    that closed."""

    position: Position
    closed_volume: float = Field(gt=0)
    realized_pnl: float
    reason: str = ""


class CancelOrderRequestEvent(Event):
    """Strategy → router request to cancel a pending order.

    Ownership is enforced by the router (magic match). Use this when the
    OCO manager wants to drop a sibling pending after another leg fills,
    or when the strategy decides to abort a pending entry.
    """

    strategy_id: str
    ticket: int
    reason: str = ""


class ClosePositionRequestEvent(Event):
    """Strategy → router request to fully close an existing position.

    Unlike `PartialCloseRequestEvent` (which reduces volume), this always
    closes 100 % of the position. The router enforces ownership via magic.
    """

    strategy_id: str
    ticket: int
    reason: str = ""


# --- Account ------------------------------------------------------------------


class AccountSnapshotEvent(Event):
    snapshot: AccountSnapshot


# --- Engine / lifecycle -------------------------------------------------------


class EngineStartedEvent(Event):
    pass


class EngineStoppedEvent(Event):
    pass


class EngineHeartbeatEvent(Event):
    """Periodic heartbeat — UIs use this to keep their views fresh."""

    interval_seconds: float


class StrategyStateChangedEvent(Event):
    strategy_id: str
    state: str                       # "started" | "stopped" | "paused" | "quarantined"
    reason: str = ""


class BrokerDisconnectedEvent(Event):
    """Broker connection lost — engine should treat the broker as offline until
    a corresponding ``BrokerReconnectedEvent`` arrives."""

    broker_name: str
    reason: str = ""


class BrokerReconnectedEvent(Event):
    """Broker connection re-established after a disconnect.  Tick pumps and
    order queues should resume.  Fired *after* tick gap-fill (A3) completes so
    callers see the broker as fully caught up by the time they observe this."""

    broker_name: str
    downtime_seconds: float = 0.0


class BacktestEquitySampleEvent(Event):
    """Periodic equity snapshot published by ``FileBacktester`` during replay.

    Bar mode emits one per bar; tick mode emits one per UTC-minute boundary
    (matches the existing ``equity_curve`` sampling cadence). Live-backtest
    UIs subscribe to this so the equity chart can advance progressively
    rather than waiting for the run to finish.

    Time is the simulated bar/tick time, not wall-clock. ``balance`` is the
    realised cash balance; ``equity`` = balance + open-position MTM at
    sample time.
    """

    time: datetime
    balance: float
    equity: float


class BacktestTradeClosedEvent(Event):
    """Fired by ``SimBroker`` right after a position fully closes and the
    matching ``TradeRecord`` is appended to ``broker._trades``.

    Live-backtest UIs subscribe to this to update the Orders table on the
    fly — ``PositionClosedEvent`` alone doesn't carry ``close_price`` /
    ``close_time``, and pulling them from ``broker._trades`` from outside
    the broker is ugly. The payload mirrors ``TradeRecord`` plus
    ``strategy_id`` + ``symbol`` so the table can show every column
    without joining other events.

    Time is the simulated broker time (not wall-clock). ``pnl`` is the
    realised P&L of this close (broker currency) **net of slippage,
    commission, and swap**.
    """

    ticket: int
    strategy_id: str
    symbol: str
    side: str                  # "buy" | "sell"
    volume: float
    open_price: float
    close_price: float
    open_ts: datetime
    close_ts: datetime
    pnl: float


class TickStreamUnsubscribedEvent(Event):
    """Broker stopped streaming ticks for a symbol.  Consumers that hold
    per-symbol state (e.g. ``MetricsCollector`` watchdog dict) should prune
    on receipt — otherwise long-running processes accumulate dead symbols and
    fire false 'stale stream' alerts forever."""

    broker_name: str
    symbol: str


class ReconciliationMismatchEvent(Event):
    """A `Reconciler` detected that broker state diverges from internal DB.

    ``mismatch_type`` mirrors :class:`stinger_fx.data.schemas.ReconciliationRow`.
    Notification sinks subscribe to this so the operator gets paged when
    the engine and broker disagree.
    """

    ticket: int
    strategy_id: str
    mismatch_type: str
    expected_value: float | None = None
    actual_value: float | None = None
    details: str = ""


class ConfigReloadedEvent(Event):
    changes: dict[str, Any] = Field(default_factory=dict)


class ConfigReloadFailedEvent(Event):
    file: str
    error: str


class LogEvent(Event):
    """For UIs that want to fan logs out via the bus instead of tailing files."""

    logger: str
    level: str
    message: str
    fields: dict[str, Any] = Field(default_factory=dict)
