"""Tick — a single market price observation."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Tick(BaseModel):
    """A single market tick. `time` is always UTC."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    time: datetime                # UTC
    bid: float = Field(gt=0)
    ask: float = Field(gt=0)
    last: float = 0.0             # 0 if not provided by broker
    volume: int = 0
    flags: int = 0                # broker-defined flag bits (MT5: TICK_FLAG_*)
    # NOTE: `replayed` lives on TickEvent (delivery metadata), not Tick.
    # Tick is pure market data — two ticks with the same fields must compare
    # equal regardless of whether one came from a gap-fill replay.

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0
