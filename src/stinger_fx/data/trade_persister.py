"""TradePersister — records each fully-closed live position to the `trades`
table.

This is the live engine's writer for `TradeRow`. Without it the table stays
empty, so `TradeRepo.realized_since` (used to rehydrate the RiskMonitor's
daily-loss counter after a restart) always reads zero. Only **full** closes are
recorded — `RiskMonitor` counts daily realized P&L off `PositionClosedEvent`
(not `PartialClosedEvent`), so persisting full closes only keeps the
reconstructed value consistent with the live counter.

Persistence must never break trading: a DB error is logged and swallowed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from stinger_fx.core.event_bus import AsyncEventBus, Subscription
from stinger_fx.core.events import PositionClosedEvent
from stinger_fx.data.repositories import TradeRepo
from stinger_fx.data.sqlite_store import SqliteStore

logger = logging.getLogger("stinger.data.trade_persister")


class TradePersister:
    """Subscribes to PositionClosedEvent and writes one TradeRow per close."""

    def __init__(
        self,
        bus: AsyncEventBus,
        store: SqliteStore,
        *,
        strategy_for_magic: Callable[[int], str | None],
    ) -> None:
        self._bus = bus
        self._repo = TradeRepo(store)
        # Resolves a position's magic tag back to the owning strategy_id.
        # Positions with an unknown magic (manual trades, other EAs) get "".
        self._strategy_for_magic = strategy_for_magic
        self._sub: Subscription | None = None

    async def start(self) -> None:
        self._sub = self._bus.subscribe(
            PositionClosedEvent, self._on_closed, name="trade_persister.close"
        )
        logger.info("trade_persister_started")

    async def stop(self) -> None:
        if self._sub is not None:
            await self._sub.unsubscribe()
            self._sub = None

    async def _on_closed(self, evt: PositionClosedEvent) -> None:
        pos = evt.position
        try:
            self._repo.add(
                position_id=pos.ticket,
                strategy_id=self._strategy_for_magic(pos.magic) or "",
                symbol=pos.symbol,
                side=pos.side.value,
                open_ts=pos.open_time,
                close_ts=datetime.now(UTC),
                open_price=pos.open_price,
                # Fall back to open_price so the (non-null) column is always set
                # even if a publisher omits the close price.
                close_price=evt.close_price
                if evt.close_price is not None
                else pos.open_price,
                volume=pos.volume,
                pnl=evt.realized_pnl,
            )
        except Exception:
            # Never let a persistence hiccup propagate into the trading path.
            logger.exception("trade_persist_failed ticket=%s", pos.ticket)
