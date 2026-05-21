"""OHLCV bar."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stinger_fx.domain.timeframes import Timeframe


class Bar(BaseModel):
    """A single OHLCV bar. `time` is the bar's open time, UTC."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    timeframe: Timeframe
    time: datetime                       # bar open, UTC
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    tick_volume: int = 0
    real_volume: int = 0
    spread: int = 0
    is_closed: bool = True

    @model_validator(mode="after")
    def _check_ohlc(self) -> Bar:
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"open {self.open} outside [low {self.low}, high {self.high}]")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"close {self.close} outside [low {self.low}, high {self.high}]")
        return self
