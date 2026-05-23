"""BrokerPool — keyed access to brokers in a multi-account setup.

Each broker has an `account_id` (matching its `BrokerConfig.id`). Strategies
pick which broker they trade against via `StrategyEntry.account`. The pool
is owned by the engine and exposed through `EngineHandle`.

For single-broker configs the pool still works — it just holds one entry
with id="default" and the legacy code paths still resolve.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from stinger_fx.brokers.base import BaseBroker
from stinger_fx.core.errors import ConfigError

logger = logging.getLogger("stinger.broker.pool")


class BrokerPool:
    """Mapping account_id → BaseBroker. Iteration order = insertion order."""

    def __init__(self, brokers: Iterable[tuple[str, BaseBroker]] = ()) -> None:
        self._brokers: dict[str, BaseBroker] = {}
        for account_id, broker in brokers:
            self.add(account_id, broker)

    def add(self, account_id: str, broker: BaseBroker) -> None:
        if account_id in self._brokers:
            raise ConfigError(f"broker pool already has account_id={account_id!r}")
        self._brokers[account_id] = broker

    def get(self, account_id: str) -> BaseBroker:
        if account_id not in self._brokers:
            raise KeyError(
                f"no broker registered for account_id={account_id!r} "
                f"(known: {sorted(self._brokers)})"
            )
        return self._brokers[account_id]

    def has(self, account_id: str) -> bool:
        return account_id in self._brokers

    def primary(self) -> BaseBroker:
        """First-added broker — used by UIs that show a single account."""
        if not self._brokers:
            raise KeyError("broker pool is empty")
        return next(iter(self._brokers.values()))

    def primary_id(self) -> str:
        if not self._brokers:
            raise KeyError("broker pool is empty")
        return next(iter(self._brokers))

    def all(self) -> list[BaseBroker]:
        return list(self._brokers.values())

    def items(self) -> list[tuple[str, BaseBroker]]:
        return list(self._brokers.items())

    def __len__(self) -> int:
        return len(self._brokers)

    def __contains__(self, account_id: str) -> bool:
        return account_id in self._brokers
