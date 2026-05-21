"""Open position state."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        """+1 for BUY, -1 for SELL — handy for P&L math."""
        return 1 if self is Side.BUY else -1


class Position(BaseModel):
    """An open position at the broker."""

    model_config = ConfigDict(frozen=True)

    ticket: int                          # broker position id
    symbol: str
    side: Side
    volume: float = Field(gt=0)
    open_price: float = Field(gt=0)
    open_time: datetime                   # UTC
    sl: float | None = None
    tp: float | None = None
    swap: float = 0.0
    profit: float = 0.0                   # unrealized P&L in account currency
    comment: str = ""
    magic: int = 0                        # strategy tag (see strategies/runner.py)
