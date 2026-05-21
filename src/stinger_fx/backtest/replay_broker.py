"""SimBroker — an in-memory `BaseBroker` for backtest replay.

It honours the same interface as MT5Broker but never opens a real connection.
Order fills are simulated at the next bar's open + a configurable slippage.

This keeps the strategy code path identical between live and backtest.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pyarrow as pa

from stinger_fx.backtest.reports import TradeRecord
from stinger_fx.backtest.slippage import fixed_pips_slippage
from stinger_fx.brokers.base import BaseBroker
from stinger_fx.core.event_bus import AsyncEventBus
from stinger_fx.core.events import (
    OrderFilledEvent,
    PositionClosedEvent,
)
from stinger_fx.domain import (
    AccountInfo,
    Order,
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    Position,
    Side,
    SymbolInfo,
    Timeframe,
)

logger = logging.getLogger("stinger.backtest.sim_broker")


class SimBroker(BaseBroker):
    name = "sim"

    def __init__(
        self,
        bus: AsyncEventBus,
        *,
        initial_balance: float,
        slippage_pips: float = 0.0,
        contract_size: float = 100_000.0,
        point: float = 0.0001,
    ) -> None:
        super().__init__(bus)
        self.balance = initial_balance
        self._slippage_pips = slippage_pips
        self._contract = contract_size
        self._point = point
        self._next_ticket = 1
        self._positions: dict[int, Position] = {}
        self._sim_time: datetime = datetime.now(UTC)
        self._last_price: dict[str, float] = {}
        self._trades: list[TradeRecord] = []

    # --- For the file backtester to drive the sim ---------------------------

    def advance_clock(self, t: datetime) -> None:
        self._sim_time = t

    def set_market(self, symbol: str, price: float) -> None:
        self._last_price[symbol] = price

    @property
    def trades(self) -> list[TradeRecord]:
        return list(self._trades)

    def realized_balance(self) -> float:
        return self.balance

    # --- BaseBroker -------------------------------------------------------

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def is_connected(self) -> bool:
        return True

    async def get_account_info(self) -> AccountInfo:
        return AccountInfo(
            account_id="sim",
            broker="sim",
            server="sim",
            currency="USD",
            leverage=100,
            name="simulated",
        )

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        return SymbolInfo(
            symbol=symbol,
            digits=5,
            point=self._point,
            contract_size=self._contract,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            currency_base="EUR",
            currency_profit="USD",
            currency_margin="USD",
        )

    async def list_symbols(self) -> list[str]:
        return list(self._last_price.keys())

    async def subscribe_ticks(self, symbol: str) -> None:
        return None

    async def subscribe_bars(self, symbol: str, tf: Timeframe) -> None:
        return None

    async def unsubscribe(self, symbol: str, tf: Timeframe | None = None) -> None:
        return None

    async def get_history_bars(
        self,
        symbol: str,
        tf: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pa.Table:
        from stinger_fx.data.parquet_store import BAR_SCHEMA

        return BAR_SCHEMA.empty_table()

    async def get_history_ticks(
        self, symbol: str, start: datetime, end: datetime
    ) -> pa.Table:
        from stinger_fx.data.parquet_store import TICK_SCHEMA

        return TICK_SCHEMA.empty_table()

    async def place_order(self, req: OrderRequest) -> OrderResult:
        if req.type != OrderType.MARKET:
            return OrderResult(
                ok=False,
                status=OrderStatus.REJECTED,
                message=f"sim broker supports MARKET only (got {req.type})",
            )
        mid = self._last_price.get(req.symbol)
        if mid is None:
            return OrderResult(
                ok=False,
                status=OrderStatus.REJECTED,
                message=f"no market price for {req.symbol}",
            )
        fill_price = fixed_pips_slippage(mid, req.side, self._slippage_pips, self._point)
        ticket = self._next_ticket
        self._next_ticket += 1
        pos = Position(
            ticket=ticket,
            symbol=req.symbol,
            side=req.side,
            volume=req.volume,
            open_price=fill_price,
            open_time=self._sim_time,
            sl=req.sl,
            tp=req.tp,
            comment=req.comment,
            magic=req.magic,
        )
        self._positions[ticket] = pos

        order = Order(
            ticket=ticket,
            strategy_id=req.strategy_id,
            symbol=req.symbol,
            side=req.side,
            type=OrderType.MARKET,
            volume=req.volume,
            filled_volume=req.volume,
            price=mid,
            fill_price=fill_price,
            sl=req.sl,
            tp=req.tp,
            status=OrderStatus.FILLED,
            comment=req.comment,
            magic=req.magic,
            client_order_id=req.client_order_id,
            requested_at=self._sim_time,
            filled_at=self._sim_time,
        )
        await self.bus.publish(OrderFilledEvent(order=order))
        return OrderResult(ok=True, ticket=ticket, status=OrderStatus.FILLED, order=order)

    async def modify_order(
        self,
        ticket: int,
        *,
        sl: float | None = None,
        tp: float | None = None,
        price: float | None = None,
    ) -> OrderResult:
        pos = self._positions.get(ticket)
        if pos is None:
            return OrderResult(ok=False, status=OrderStatus.REJECTED, message="not found")
        # Pydantic frozen — rebuild
        updated = pos.model_copy(update={"sl": sl, "tp": tp})
        self._positions[ticket] = updated
        return OrderResult(ok=True, ticket=ticket, status=OrderStatus.SUBMITTED)

    async def close_position(
        self, ticket: int, volume: float | None = None
    ) -> OrderResult:
        pos = self._positions.pop(ticket, None)
        if pos is None:
            return OrderResult(ok=False, status=OrderStatus.REJECTED, message="not found")
        mid = self._last_price.get(pos.symbol)
        if mid is None:
            return OrderResult(ok=False, status=OrderStatus.REJECTED, message="no market price")
        close_side = Side.SELL if pos.side is Side.BUY else Side.BUY
        close_price = fixed_pips_slippage(mid, close_side, self._slippage_pips, self._point)
        # P&L = (close - open) * sign * volume * contract_size, in profit currency.
        pnl = (close_price - pos.open_price) * pos.side.sign * pos.volume * self._contract
        self.balance += pnl
        self._trades.append(
            TradeRecord(
                open_ts=pos.open_time,
                close_ts=self._sim_time,
                side=pos.side.value,
                open_price=pos.open_price,
                close_price=close_price,
                volume=pos.volume,
                pnl=pnl,
            )
        )
        await self.bus.publish(
            PositionClosedEvent(position=pos, realized_pnl=pnl)
        )
        return OrderResult(ok=True, ticket=ticket, status=OrderStatus.FILLED)

    async def cancel_order(self, ticket: int) -> OrderResult:
        # No pending orders in the sim — only market fills.
        return OrderResult(ok=False, status=OrderStatus.REJECTED, message="no pending orders")

    async def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    async def get_open_orders(self) -> list[Order]:
        return []

    # --- Backtest helpers ---------------------------------------------------

    def check_sl_tp(self, symbol: str, bar_high: float, bar_low: float) -> list[Position]:
        """Returns positions that should be closed (one of SL/TP is breached this bar)."""
        to_close: list[Position] = []
        for pos in list(self._positions.values()):
            if pos.symbol != symbol:
                continue
            if pos.side is Side.BUY:
                if pos.sl is not None and bar_low <= pos.sl:
                    to_close.append(pos)
                    continue
                if pos.tp is not None and bar_high >= pos.tp:
                    to_close.append(pos)
            else:
                if pos.sl is not None and bar_high >= pos.sl:
                    to_close.append(pos)
                    continue
                if pos.tp is not None and bar_low <= pos.tp:
                    to_close.append(pos)
        return to_close
