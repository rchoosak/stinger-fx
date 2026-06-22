"""Durable per-strategy position state for swing strategies that hold a single
position across engine restarts.

A swing strategy (e.g. ``D1H4TrendStrategy``) trails a ratcheting stop over days
or weeks. That stop can't be reconstructed from price data after a restart — the
ratchet only ever tightened, so a fresh recompute would be looser than the value
the strategy actually committed to. So the strategy persists it, keyed to the
exact position it belongs to.

On restart the persisted state is **reconciled** against the live broker
positions: it is restored only when strategy id, symbol, side, entry price and
broker ticket all match an actually-open position. Anything else (the position
was closed while we were down, a different position now occupies the slot, a
hand-edited file) is treated as stale and cleared — we never resurrect a stop
for a position that isn't there.

The store is pluggable: backtests / tests use :class:`InMemoryStateStore` (no
files); live uses :class:`JsonFileStateStore`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from stinger_fx.domain import Position

logger = logging.getLogger("stinger.strategies.state")


@dataclass(frozen=True)
class PositionState:
    """The minimum durable identity + trailing stop for one held position."""

    strategy_id: str
    symbol: str
    side: str          # Side.value — "BUY" | "SELL"
    entry_price: float
    ticket: int
    chandelier_stop: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PositionState:
        return cls(
            strategy_id=str(d["strategy_id"]),
            symbol=str(d["symbol"]),
            side=str(d["side"]),
            entry_price=float(d["entry_price"]),
            ticket=int(d["ticket"]),
            chandelier_stop=float(d["chandelier_stop"]),
        )


class StrategyStateStore(Protocol):
    """Loads / saves / clears one :class:`PositionState` per strategy id."""

    def load(self, strategy_id: str) -> PositionState | None: ...

    def save(self, state: PositionState) -> None: ...

    def clear(self, strategy_id: str) -> None: ...


class InMemoryStateStore:
    """Process-local store — the default for backtests and unit tests where
    there is no restart and nothing should touch the filesystem."""

    def __init__(self) -> None:
        self._states: dict[str, PositionState] = {}

    def load(self, strategy_id: str) -> PositionState | None:
        return self._states.get(strategy_id)

    def save(self, state: PositionState) -> None:
        self._states[state.strategy_id] = state

    def clear(self, strategy_id: str) -> None:
        self._states.pop(strategy_id, None)


class JsonFileStateStore:
    """Persists all strategies' states in one JSON file as
    ``{strategy_id: state_dict}``. Writes are atomic (temp file + ``rename``) so
    a crash mid-write never corrupts the file. A read error is logged and
    treated as "no state" — a bad file must never block startup."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def _read_all(self) -> dict[str, dict[str, object]]:
        if not self._path.exists():
            return {}
        try:
            with self._path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            logger.exception("strategy_state_read_failed path=%s", self._path)
            return {}

    def _write_all(self, data: dict[str, dict[str, object]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
            os.replace(tmp, self._path)
        except OSError:
            logger.exception("strategy_state_write_failed path=%s", self._path)
            with contextlib.suppress(OSError):
                os.unlink(tmp)

    def load(self, strategy_id: str) -> PositionState | None:
        raw = self._read_all().get(strategy_id)
        if raw is None:
            return None
        try:
            return PositionState.from_dict(raw)
        except (KeyError, TypeError, ValueError):
            logger.exception("strategy_state_parse_failed id=%s", strategy_id)
            return None

    def save(self, state: PositionState) -> None:
        data = self._read_all()
        data[state.strategy_id] = state.to_dict()  # type: ignore[assignment]
        self._write_all(data)

    def clear(self, strategy_id: str) -> None:
        data = self._read_all()
        if data.pop(strategy_id, None) is not None:
            self._write_all(data)


def reconcile(
    persisted: PositionState | None,
    positions: Iterable[Position],
    *,
    strategy_id: str,
    symbol: str,
    entry_tolerance: float = 1e-6,
) -> PositionState | None:
    """Return the persisted state only if it still matches a live position.

    Match requires identical strategy id, symbol, side, broker ticket, and
    entry price within ``entry_tolerance``. (Broker order id ≠ position ticket,
    so the match is on the *position* ticket.) Any mismatch — closed, replaced,
    or edited — yields ``None`` so the caller clears the stale state.
    """
    if persisted is None:
        return None
    if persisted.strategy_id != strategy_id or persisted.symbol != symbol:
        return None
    for pos in positions:
        if (
            pos.ticket == persisted.ticket
            and pos.symbol == symbol
            and pos.side.value == persisted.side
            and abs(pos.open_price - persisted.entry_price) <= entry_tolerance
        ):
            return persisted
    return None


def state_for_position(
    *, strategy_id: str, position: Position, chandelier_stop: float
) -> PositionState:
    """Build a :class:`PositionState` from a live position + current stop."""
    return PositionState(
        strategy_id=strategy_id,
        symbol=position.symbol,
        side=position.side.value,
        entry_price=position.open_price,
        ticket=position.ticket,
        chandelier_stop=chandelier_stop,
    )


__all__ = [
    "InMemoryStateStore",
    "JsonFileStateStore",
    "PositionState",
    "StrategyStateStore",
    "reconcile",
    "state_for_position",
]
