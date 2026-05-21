"""Strategy → engine signaling primitives."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from stinger_fx.domain.positions import Side


class SignalStrength(str, Enum):
    WEAK = "weak"
    NORMAL = "normal"
    STRONG = "strong"


class Signal(BaseModel):
    """A strategy's recommendation — not yet an order.

    The OrderRouter validates risk + dedupes, then turns this into an OrderRequest.
    """

    model_config = ConfigDict(frozen=True)

    strategy_id: str
    time: datetime                            # UTC
    symbol: str
    side: Side                                # buy / sell (close-position signals use opposite side)
    strength: SignalStrength = SignalStrength.NORMAL
    suggested_volume: float | None = None
    suggested_sl: float | None = None
    suggested_tp: float | None = None
    comment: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    """The router's verdict on a signal — kept for audit, even when rejected."""

    model_config = ConfigDict(frozen=True)

    signal: Signal
    time: datetime                            # UTC
    action: str                               # "placed" | "rejected" | "deduped" | "throttled"
    reason: str = ""
    risk_check_passed: bool = True
    client_order_id: str | None = None
