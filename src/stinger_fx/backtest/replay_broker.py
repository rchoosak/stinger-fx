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
from stinger_fx.backtest.slippage import SlippageModel, fixed_pips_model
from stinger_fx.brokers.base import BaseBroker
from stinger_fx.core.event_bus import AsyncEventBus
from stinger_fx.core.events import (
    OrderCancelledEvent,
    OrderFilledEvent,
    OrderSubmittedEvent,
    PartialClosedEvent,
    PositionClosedEvent,
)
from stinger_fx.domain import (
    AccountInfo,
    AccountSnapshot,
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


def _pending_triggered(order: Order, bid: float, ask: float) -> bool:
    """Return True when current bid/ask triggers this pending order."""
    if order.price is None:
        return False
    if order.type is OrderType.STOP:
        if order.side is Side.BUY:
            return ask >= order.price
        return bid <= order.price
    if order.type is OrderType.LIMIT:
        if order.side is Side.BUY:
            return ask <= order.price
        return bid >= order.price
    if order.type is OrderType.STOP_LIMIT:
        # Phase 6.2.A treats STOP_LIMIT like STOP — full two-stage behaviour
        # (stop triggers a limit order) is deferred since MT5 brokers vary.
        if order.side is Side.BUY:
            return ask >= order.price
        return bid <= order.price
    return False


class SimBroker(BaseBroker):
    name = "sim"

    def __init__(
        self,
        bus: AsyncEventBus,
        *,
        initial_balance: float,
        slippage_pips: float = 0.0,
        slippage_fn: SlippageModel | None = None,
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
        # Phase 6.2.A — pending orders (BUY/SELL STOP & LIMIT) waiting on a
        # trigger price. Key is ticket; check_pending() iterates this dict
        # every tick and promotes triggered orders into open positions.
        self._pending: dict[int, Order] = {}
        self._sim_time: datetime = datetime.now(UTC)
        # Per-symbol bid/ask — set by advance_market(). `_last_price` kept for
        # backward compat (check_sl_tp uses it as a mark-to-market reference).
        self._last_price: dict[str, float] = {}
        self._last_bid: dict[str, float] = {}
        self._last_ask: dict[str, float] = {}
        self._trades: list[TradeRecord] = []
        # Slippage callable — if not supplied, fall back to the legacy fixed-pips model.
        self._slippage_fn: SlippageModel = slippage_fn or fixed_pips_model(
            pips=slippage_pips, point=point
        )

    # --- For the file backtester to drive the sim ---------------------------

    def advance_clock(self, t: datetime) -> None:
        self._sim_time = t

    def set_market(self, symbol: str, price: float) -> None:
        """Set both bid and ask to *price* (no spread).  Bar-mode entry point."""
        self._last_price[symbol] = price
        self._last_bid[symbol] = price
        self._last_ask[symbol] = price

    def set_market_tick(self, symbol: str, bid: float, ask: float) -> None:
        """Set bid and ask independently.  Tick-mode entry point."""
        self._last_bid[symbol] = bid
        self._last_ask[symbol] = ask
        self._last_price[symbol] = bid  # mark-to-market reference = bid

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

    async def get_account_snapshot(self) -> AccountSnapshot:
        # Unrealized P&L on still-open positions, marked-to-market at the last
        # price the backtester pushed in.
        unrealized = 0.0
        for p in self._positions.values():
            mid = self._last_price.get(p.symbol)
            if mid is None:
                continue
            unrealized += (mid - p.open_price) * p.side.sign * p.volume * self._contract
        equity = self.balance + unrealized
        return AccountSnapshot(
            account_id="sim",
            time=self._sim_time,
            balance=self.balance,
            equity=equity,
            margin=0.0,
            free_margin=equity,
            margin_level=0.0,
            profit=unrealized,
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
        bid = self._last_bid.get(req.symbol)
        ask = self._last_ask.get(req.symbol)
        if bid is None or ask is None:
            return OrderResult(
                ok=False,
                status=OrderStatus.REJECTED,
                message=f"no market price for {req.symbol}",
            )

        # --- Pending orders (Phase 6.2.A) -----------------------------------
        # Anything that's not MARKET gets parked in self._pending and lives
        # until either check_pending() triggers it (price crossed) or
        # cancel_order() removes it.
        if req.type != OrderType.MARKET:
            if req.price is None:
                return OrderResult(
                    ok=False,
                    status=OrderStatus.REJECTED,
                    message=f"{req.type.value} order requires a price",
                )
            ticket = self._next_ticket
            self._next_ticket += 1
            pending = Order(
                ticket=ticket,
                strategy_id=req.strategy_id,
                symbol=req.symbol,
                side=req.side,
                type=req.type,
                volume=req.volume,
                price=req.price,
                sl=req.sl,
                tp=req.tp,
                status=OrderStatus.SUBMITTED,
                comment=req.comment,
                magic=req.magic,
                client_order_id=req.client_order_id,
                requested_at=self._sim_time,
            )
            self._pending[ticket] = pending
            await self.bus.publish(OrderSubmittedEvent(order=pending))
            return OrderResult(
                ok=True, ticket=ticket, status=OrderStatus.SUBMITTED, order=pending
            )

        # --- Market orders (unchanged) --------------------------------------
        fill_price = self._slippage_fn(req.side, bid=bid, ask=ask)
        mid = (bid + ask) / 2
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
        # The OrderRouter is responsible for publishing OrderFilledEvent
        # for market fills — matches MT5Broker's pattern and avoids a
        # double-emit when the router runs upstream. (Pending orders
        # are different: check_pending() emits OrderFilledEvent itself
        # because it runs outside the router's call frame.)
        return OrderResult(ok=True, ticket=ticket, status=OrderStatus.FILLED, order=order)

    async def check_pending(self, symbol: str, bid: float, ask: float) -> list[Order]:
        """Trigger any pending orders whose conditions are met by the current
        bid/ask. Returns the list of orders that fired (for telemetry / tests).

        Trigger semantics (consistent with MT5 conventions):
          * BUY_STOP   — fires when ask reaches/exceeds the stop price
          * SELL_STOP  — fires when bid falls to/below the stop price
          * BUY_LIMIT  — fires when ask falls to/below the limit price
          * SELL_LIMIT — fires when bid rises to/exceeds the limit price

        Fill price:
          * STOP orders convert to MARKET on trigger → slippage applied
          * LIMIT orders fill exactly at the limit price (the strategy
            specified that as their worst acceptable price)
        """
        triggered: list[Order] = []
        for ticket, pending in list(self._pending.items()):
            if pending.symbol != symbol:
                continue
            if not _pending_triggered(pending, bid, ask):
                continue
            # Compute fill price
            if pending.type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
                fill_price = pending.price or bid
            else:
                # STOP → market-style fill with slippage
                fill_price = self._slippage_fn(pending.side, bid=bid, ask=ask)
            filled = await self._promote_pending(ticket, pending, fill_price)
            triggered.append(filled)
        return triggered

    async def check_pending_bar(
        self, symbol: str, bar_high: float, bar_low: float
    ) -> list[Order]:
        """Bar-mode variant of ``check_pending``.

        A bar reports the price extent it covered during the period
        (``[low, high]``) but not the intra-bar path. A pending order
        whose trigger price lies inside that range was touched at some
        point during the bar — that's the condition we fire on.

        Pre-fix the bar-mode replay loop never called this, so STOP /
        LIMIT orders placed in a bar backtest sat in ``_pending``
        forever and the backtest equity curve didn't reflect them.

        Trigger semantics in bar mode:
          * All four types fire when ``bar_low <= pending.price <= bar_high``.
            The directional predicates collapse here — a pending order
            wouldn't have been placed if the trigger had already been
            crossed by the time of placement, and once placed, the only
            way it fires is by the bar's range covering the price.

        Fill price (bar mode lacks bid/ask, so we use conservative proxies):
          * STOP / STOP_LIMIT — fill at the trigger price with slippage
            applied (slippage_fn called with bid=ask=trigger). The market
            crossed the trigger at some intra-bar moment; the exact path
            is unknown so the trigger is the conservative midpoint.
          * LIMIT — fill at the limit price exactly (the strategy
            specified that as their worst acceptable price).
        """
        triggered: list[Order] = []
        for ticket, pending in list(self._pending.items()):
            if pending.symbol != symbol or pending.price is None:
                continue
            if not (bar_low <= pending.price <= bar_high):
                continue
            if pending.type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
                fill_price = pending.price
            else:
                # STOP → trigger + slippage; bar data has no spread,
                # so feed the slippage_fn the trigger as both bid and ask.
                fill_price = self._slippage_fn(
                    pending.side, bid=pending.price, ask=pending.price
                )
            filled = await self._promote_pending(ticket, pending, fill_price)
            triggered.append(filled)
        return triggered

    async def _promote_pending(
        self, ticket: int, pending: Order, fill_price: float
    ) -> Order:
        """Convert a triggered pending order into an open position +
        emit OrderFilledEvent. Shared by ``check_pending`` (tick mode)
        and ``check_pending_bar`` (bar mode)."""
        pos = Position(
            ticket=ticket,
            symbol=pending.symbol,
            side=pending.side,
            volume=pending.volume,
            open_price=fill_price,
            open_time=self._sim_time,
            sl=pending.sl,
            tp=pending.tp,
            comment=pending.comment,
            magic=pending.magic,
        )
        self._positions[ticket] = pos
        del self._pending[ticket]

        filled = pending.model_copy(update={
            "status": OrderStatus.FILLED,
            "filled_volume": pending.volume,
            "fill_price": fill_price,
            "filled_at": self._sim_time,
        })
        await self.bus.publish(OrderFilledEvent(order=filled))
        return filled

    async def modify_order(
        self,
        ticket: int,
        *,
        sl: float | None = None,
        tp: float | None = None,
        price: float | None = None,
        volume: float | None = None,
    ) -> OrderResult:
        # Phase 6.2.D — pending orders can have their trigger price and
        # volume adjusted as well as SL/TP. Open positions can only have
        # SL/TP changed (entry price is immutable post-fill).
        pending = self._pending.get(ticket)
        if pending is not None:
            if volume is not None and volume <= 0:
                return OrderResult(
                    ok=False, status=OrderStatus.REJECTED,
                    message=f"modify pending volume must be > 0 (got {volume})",
                )
            updated = pending.model_copy(
                update={
                    "sl": sl if sl is not None else pending.sl,
                    "tp": tp if tp is not None else pending.tp,
                    "price": price if price is not None else pending.price,
                    "volume": volume if volume is not None else pending.volume,
                }
            )
            self._pending[ticket] = updated
            return OrderResult(
                ok=True, ticket=ticket, status=OrderStatus.SUBMITTED, order=updated
            )

        pos = self._positions.get(ticket)
        if pos is None:
            return OrderResult(ok=False, status=OrderStatus.REJECTED, message="not found")
        # Pydantic frozen — rebuild. Pass through current values for fields
        # the caller didn't supply so we don't accidentally clear an existing
        # SL when only TP was being moved (and vice-versa).
        updated = pos.model_copy(
            update={
                "sl": sl if sl is not None else pos.sl,
                "tp": tp if tp is not None else pos.tp,
            }
        )
        self._positions[ticket] = updated
        return OrderResult(ok=True, ticket=ticket, status=OrderStatus.SUBMITTED)

    async def close_position(
        self, ticket: int, volume: float | None = None
    ) -> OrderResult:
        pos = self._positions.get(ticket)
        if pos is None:
            return OrderResult(ok=False, status=OrderStatus.REJECTED, message="not found")
        mid = self._last_price.get(pos.symbol)
        if mid is None:
            return OrderResult(ok=False, status=OrderStatus.REJECTED, message="no market price")

        # Phase 4: a partial close passes a `volume` strictly less than the
        # position's current volume. We close that chunk and leave the rest
        # open with a shrunken `volume`. A None / equal / over-sized volume
        # request closes the whole position (the over-sized case is filtered
        # at the router; brokers in the wild typically clamp).
        full_close = volume is None or volume >= pos.volume
        close_qty = pos.volume if full_close else float(volume)

        # When closing a BUY we exit at bid (sell side), and vice versa.
        close_side = Side.SELL if pos.side is Side.BUY else Side.BUY
        bid = self._last_bid.get(pos.symbol, mid)
        ask = self._last_ask.get(pos.symbol, mid)
        close_price = self._slippage_fn(close_side, bid=bid, ask=ask)
        # P&L on the closed chunk only.
        pnl = (close_price - pos.open_price) * pos.side.sign * close_qty * self._contract
        self.balance += pnl
        self._trades.append(
            TradeRecord(
                open_ts=pos.open_time,
                close_ts=self._sim_time,
                side=pos.side.value,
                open_price=pos.open_price,
                close_price=close_price,
                volume=close_qty,
                pnl=pnl,
            )
        )

        if full_close:
            self._positions.pop(ticket, None)
            await self.bus.publish(
                PositionClosedEvent(position=pos, realized_pnl=pnl)
            )
        else:
            remaining = pos.model_copy(update={"volume": pos.volume - close_qty})
            self._positions[ticket] = remaining
            await self.bus.publish(
                PartialClosedEvent(
                    position=remaining,
                    closed_volume=close_qty,
                    realized_pnl=pnl,
                )
            )
        return OrderResult(ok=True, ticket=ticket, status=OrderStatus.FILLED)

    async def cancel_order(self, ticket: int) -> OrderResult:
        pending = self._pending.pop(ticket, None)
        if pending is None:
            return OrderResult(
                ok=False, status=OrderStatus.REJECTED, message="no pending order with that ticket"
            )
        cancelled = pending.model_copy(update={"status": OrderStatus.CANCELLED})
        await self.bus.publish(OrderCancelledEvent(order=cancelled))
        return OrderResult(
            ok=True, ticket=ticket, status=OrderStatus.CANCELLED, order=cancelled
        )

    async def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    async def get_open_orders(self) -> list[Order]:
        return list(self._pending.values())

    # --- Backtest helpers ---------------------------------------------------

    def check_sl_tp(self, symbol: str, bar_high: float, bar_low: float) -> list[Position]:
        """Returns positions that should be closed (one of SL/TP is breached this bar).

        Bar-mode check: scans bar high/low. Ambiguous when a wick touches both
        SL and TP — `tick`-mode uses `check_sl_tp_tick` for unambiguous timing.
        """
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

    def check_sl_tp_tick(self, symbol: str, bid: float, ask: float) -> list[Position]:
        """Tick-mode SL/TP check (Phase 4).

        Conservative quoting model — a long position closes on:
          • SL when **bid** ≤ sl   (we sell at bid, so SL fires when bid hits)
          • TP when **bid** ≥ tp   (same logic — we sell at bid for profit)

        A short position closes on:
          • SL when **ask** ≥ sl   (we buy back at ask)
          • TP when **ask** ≤ tp

        Unlike bar-mode `check_sl_tp`, this is unambiguous chronologically:
        each tick reveals one bid/ask snapshot, so SL and TP can't both fire
        on the same event.
        """
        to_close: list[Position] = []
        for pos in list(self._positions.values()):
            if pos.symbol != symbol:
                continue
            if pos.side is Side.BUY:
                if pos.sl is not None and bid <= pos.sl:
                    to_close.append(pos)
                    continue
                if pos.tp is not None and bid >= pos.tp:
                    to_close.append(pos)
            else:
                if pos.sl is not None and ask >= pos.sl:
                    to_close.append(pos)
                    continue
                if pos.tp is not None and ask <= pos.tp:
                    to_close.append(pos)
        return to_close
