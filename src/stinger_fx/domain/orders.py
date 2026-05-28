"""Order request, status, and result types."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from stinger_fx.domain.positions import Side


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(StrEnum):
    PENDING = "pending"               # request created locally, not sent
    SUBMITTED = "submitted"           # accepted by broker, awaiting fill
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderRequest(BaseModel):
    """A strategy's intent to place an order — pre-broker."""

    model_config = ConfigDict(frozen=True)

    strategy_id: str
    symbol: str
    side: Side
    type: OrderType = OrderType.MARKET
    volume: float = Field(gt=0)
    price: float | None = None       # required for non-MARKET
    sl: float | None = None
    tp: float | None = None
    comment: str = ""
    magic: int = 0
    client_order_id: str               # idempotency key; engine generates


class Order(BaseModel):
    """An order as tracked in our database (after submission)."""

    model_config = ConfigDict(frozen=True)

    ticket: int                        # broker order id
    strategy_id: str
    symbol: str
    side: Side
    type: OrderType
    volume: float = Field(gt=0)
    filled_volume: float = 0.0
    price: float | None = None         # request price for pending; fill price for market
    fill_price: float | None = None
    sl: float | None = None
    tp: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    comment: str = ""
    magic: int = 0
    client_order_id: str = ""
    requested_at: datetime | None = None
    filled_at: datetime | None = None


class OrderResult(BaseModel):
    """Outcome of a place/modify/close/cancel call."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    ticket: int | None = None
    status: OrderStatus
    message: str = ""
    raw_code: int | None = None        # broker-specific return code
    order: Order | None = None
