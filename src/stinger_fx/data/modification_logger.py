"""ModificationLogger — writes OrderModifiedEvent and PartialClosedEvent to SQLite.

This component subscribes to the event bus and persists every SL/TP
modification and partial-close into the ``order_modifications`` table.
It is optional — the engine passes `None` for `store` when no SQLite
database is configured, and the logger becomes a no-op.

Usage::

    logger = ModificationLogger(bus, store=sqlite_store, strategy_magic=magic_map)
    await logger.start()
    ...
    await logger.stop()
"""

from __future__ import annotations

import logging

from stinger_fx.core.event_bus import AsyncEventBus, Subscription
from stinger_fx.core.events import OrderModifiedEvent, PartialClosedEvent
from stinger_fx.data.repositories import OrderModificationRepo
from stinger_fx.data.sqlite_store import SqliteStore

logger = logging.getLogger("stinger.data.modification_logger")


class ModificationLogger:
    """Subscribes to order-management events and persists them to SQLite."""

    def __init__(
        self,
        bus: AsyncEventBus,
        store: SqliteStore | None,
        *,
        strategy_magic: dict[str, int] | None = None,
    ) -> None:
        self._bus = bus
        self._store = store
        # Reverse-lookup: magic → strategy_id for event attribution.
        self._magic_to_strategy: dict[int, str] = {
            v: k for k, v in (strategy_magic or {}).items()
        }
        self._subs: list[Subscription] = []

    async def start(self) -> None:
        if self._store is None:
            return
        self._subs.append(
            self._bus.subscribe(OrderModifiedEvent, self._on_modified, name="mod_logger.modify")
        )
        self._subs.append(
            self._bus.subscribe(PartialClosedEvent, self._on_partial, name="mod_logger.partial")
        )
        logger.info("modification_logger_started")

    async def stop(self) -> None:
        for sub in self._subs:
            await sub.unsubscribe()
        self._subs.clear()

    # --- Handlers -----------------------------------------------------------

    async def _on_modified(self, evt: OrderModifiedEvent) -> None:
        if self._store is None:
            return
        pos = evt.position
        strategy_id = self._magic_to_strategy.get(pos.magic, f"magic:{pos.magic}")
        repo = OrderModificationRepo(self._store)
        try:
            repo.record_modify(
                ts=evt.ts,
                ticket=pos.ticket,
                strategy_id=strategy_id,
                old_sl=None,      # pre-modify snapshot not available at this point;
                new_sl=pos.sl,    # pos is the *updated* snapshot from the router
                old_tp=None,
                new_tp=pos.tp,
                reason=evt.reason,
            )
        except Exception:
            logger.exception("modification_logger_write_failed ticket=%s", pos.ticket)

    async def _on_partial(self, evt: PartialClosedEvent) -> None:
        if self._store is None:
            return
        pos = evt.position
        strategy_id = self._magic_to_strategy.get(pos.magic, f"magic:{pos.magic}")
        repo = OrderModificationRepo(self._store)
        try:
            repo.record_partial_close(
                ts=evt.ts,
                ticket=pos.ticket,
                strategy_id=strategy_id,
                closed_volume=evt.closed_volume,
                realized_pnl=evt.realized_pnl,
                reason=evt.reason,
            )
        except Exception:
            logger.exception("modification_logger_write_failed ticket=%s", pos.ticket)
