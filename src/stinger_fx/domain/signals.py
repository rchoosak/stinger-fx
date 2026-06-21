"""Strategy → engine signaling primitives."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from stinger_fx.domain.orders import OrderType
from stinger_fx.domain.positions import Side


class SignalStrength(StrEnum):
    WEAK = "weak"
    NORMAL = "normal"
    STRONG = "strong"


class Signal(BaseModel):
    """A strategy's recommendation — not yet an order.

    The OrderRouter validates risk + dedupes, then turns this into an OrderRequest.

    Pending-order fields (Phase 6.2.B):
      * ``order_type``           — defaults to MARKET; set to STOP / LIMIT /
                                   STOP_LIMIT to request a pending order
      * ``suggested_price``      — trigger price for STOP / LIMIT orders
                                   (required when ``order_type != MARKET``)
      * ``suggested_stop_price`` — extra price for STOP_LIMIT (the "limit"
                                   leg that activates after the stop fires)
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
    order_type: OrderType = OrderType.MARKET
    suggested_price: float | None = None
    suggested_stop_price: float | None = None
    # Reference entry price used ONLY for risk-based position sizing
    # (stop distance = |entry_ref_price − suggested_sl|). Stamped by
    # `ctx.buy/sell`; never becomes the OrderRequest price, so market orders
    # stay price-less.
    entry_ref_price: float | None = None
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
