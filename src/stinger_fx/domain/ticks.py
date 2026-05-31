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
    replayed: bool = False        # True if replayed from history by gap-fill;
                                  # consumers (MetricsCollector) skip live-stream
                                  # signals (lag gauge, watchdog reset) for these.

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0
