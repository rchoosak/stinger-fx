"""Account info + periodic snapshots."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AccountInfo(BaseModel):
    """Mostly-static account metadata."""

    model_config = ConfigDict(frozen=True)

    account_id: str                   # broker login as string
    broker: str
    server: str
    currency: str
    leverage: int = Field(gt=0)
    name: str = ""


class AccountSnapshot(BaseModel):
    """Point-in-time account balance/equity, recorded periodically."""

    model_config = ConfigDict(frozen=True)

    account_id: str
    time: datetime                     # UTC
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float = 0.0          # equity / margin * 100, or 0 if no margin
    profit: float = 0.0                # floating P&L
