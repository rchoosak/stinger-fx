"""Symbol metadata and subscription tuples."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from stinger_fx.domain.timeframes import Timeframe


class SymbolInfo(BaseModel):
    """Broker-provided metadata for a tradable symbol."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    digits: int = Field(ge=0, le=10)
    point: float = Field(gt=0)              # smallest price increment (e.g. 0.00001)
    contract_size: float = Field(gt=0)
    volume_min: float = Field(gt=0)
    volume_max: float = Field(gt=0)
    volume_step: float = Field(gt=0)
    spread: int = 0                          # in points
    currency_base: str
    currency_profit: str
    currency_margin: str
    trade_allowed: bool = True


class Subscription(BaseModel):
    """A (symbol, timeframe) pair a strategy or broker subscribes to."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    timeframe: Timeframe

    def __str__(self) -> str:
        return f"{self.symbol}@{self.timeframe.value}"
