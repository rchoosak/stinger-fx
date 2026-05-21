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
