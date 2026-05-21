"""Clock abstraction — live vs. simulated time.

Backtest replaces `LiveClock` with `SimClock` so strategy code asking for
"now" sees the simulated bar time, not wall-clock.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime


class Clock(ABC):
    @abstractmethod
    def now(self) -> datetime:
        """Return current UTC time."""


class LiveClock(Clock):
    def now(self) -> datetime:
        return datetime.now(UTC)


class SimClock(Clock):
    """Driven by the backtest engine — advance() is called as events replay."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("SimClock requires tz-aware datetime")
        self._t = start.astimezone(UTC)

    def now(self) -> datetime:
        return self._t

    def advance(self, to: datetime) -> None:
        if to.tzinfo is None:
            raise ValueError("advance() requires tz-aware datetime")
        new_t = to.astimezone(UTC)
        if new_t < self._t:
            raise ValueError(f"clock cannot run backwards (now={self._t}, requested={new_t})")
        self._t = new_t
