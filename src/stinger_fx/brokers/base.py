"""BaseBroker — the abstraction every concrete broker must implement.

The interface is intentionally async even where the SDK call is synchronous;
that lets implementations bridge sync work into a dedicated executor without
forcing the contract on callers.

A broker holds a reference to the engine's `AsyncEventBus` so it can push
market data and order-flow events from its own pollers/callbacks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pyarrow as pa

from stinger_fx.core.event_bus import AsyncEventBus
from stinger_fx.domain import (
    AccountInfo,
    AccountSnapshot,
    Order,
    OrderRequest,
    OrderResult,
    Position,
    SymbolInfo,
    Timeframe,
)


class BaseBroker(ABC):
    """Concrete brokers subclass this and set `name`."""

    name: str = "base"

    def __init__(self, bus: AsyncEventBus) -> None:
        self.bus = bus
        # BrokerPool replaces this with the configured account id. Keeping a
        # default preserves standalone brokers used by backtests and tests.
        self.account_id = "default"

    # --- Lifecycle ----------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def is_connected(self) -> bool: ...

    # --- Account & symbols --------------------------------------------------

    @abstractmethod
    async def get_account_info(self) -> AccountInfo: ...

    @abstractmethod
    async def get_account_snapshot(self) -> AccountSnapshot:
        """Point-in-time balance/equity/margin. Polled periodically by the
        engine and published as an AccountSnapshotEvent so UIs see live P&L."""

    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> SymbolInfo: ...

    @abstractmethod
    async def list_symbols(self) -> list[str]: ...

    # --- Subscriptions (push model — broker publishes to bus) --------------

    @abstractmethod
    async def subscribe_ticks(self, symbol: str) -> None: ...

    @abstractmethod
    async def subscribe_bars(self, symbol: str, tf: Timeframe) -> None: ...

    @abstractmethod
    async def unsubscribe(self, symbol: str, tf: Timeframe | None = None) -> None: ...

    # --- Historical (pull) --------------------------------------------------

    @abstractmethod
    async def get_history_bars(
        self,
        symbol: str,
        tf: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pa.Table: ...

    @abstractmethod
    async def get_history_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pa.Table: ...

    # --- Orders -------------------------------------------------------------

    @abstractmethod
    async def place_order(self, req: OrderRequest) -> OrderResult: ...

    @abstractmethod
    async def modify_order(
        self,
        ticket: int,
        *,
        sl: float | None = None,
        tp: float | None = None,
        price: float | None = None,
        volume: float | None = None,
    ) -> OrderResult:
        """Modify an open position (sl/tp only) or a pending order.

        For an **open position** only `sl` and `tp` are meaningful — entry
        price and volume are immutable post-fill. Implementations must
        either ignore `price`/`volume` for positions or reject the request.

        For a **pending order** any combination of `sl`, `tp`, `price`,
        and `volume` may be supplied (Phase 6.2.D — pending order volume
        adjustment). `None` means "leave that field unchanged".
        """
        ...

    @abstractmethod
    async def close_position(self, ticket: int, volume: float | None = None) -> OrderResult: ...

    @abstractmethod
    async def cancel_order(self, ticket: int) -> OrderResult: ...

    # --- State queries ------------------------------------------------------

    @abstractmethod
    async def get_positions(self) -> list[Position]: ...

    @abstractmethod
    async def get_open_orders(self) -> list[Order]: ...

    # --- Lifecycle protocol (start/stop) -----------------------------------

    async def start(self) -> None:
        await self.connect()

    async def stop(self) -> None:
        await self.disconnect()
