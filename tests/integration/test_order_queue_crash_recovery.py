"""End-to-end: OrderQueue survives a crash mid-pending and replays on restart.

We simulate "engine instance 1" writing pending rows then dying without
ever completing them. "Instance 2" reuses the same SQLite store, calls
`replay_pending()`, and verifies the broker now sees the requests + the
rows transition to "sent".

Also exercises the OrderRouter ↔ queue integration: an actual signal goes
through risk → router → queue → broker, with an OrderFilledEvent emitted
on success.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from stinger_fx.backtest.order_router import OrderRouter
from stinger_fx.brokers.base import BaseBroker
from stinger_fx.brokers.order_queue import OrderQueue
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import OrderFilledEvent, SignalEvent
from stinger_fx.data import SqliteStore
from stinger_fx.data.schemas import PendingOrderRequestRow
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
    Signal,
    SignalStrength,
    SymbolInfo,
)
from sqlmodel import select


class _AlwaysFillBroker(BaseBroker):
    name = "always_fill"

    def __init__(self, bus: AsyncEventBus) -> None:
        super().__init__(bus)
        self.calls: list[OrderRequest] = []

    async def connect(self): ...
    async def disconnect(self): ...
    async def is_connected(self): return True
    async def get_account_info(self):
        return AccountInfo(account_id="x", broker="r", server="r",
                           currency="USD", leverage=100)
    async def get_account_snapshot(self):
        return AccountSnapshot(account_id="x", time=datetime.now(UTC),
                               balance=10_000, equity=10_000, margin=0, free_margin=10_000)
    async def get_symbol_info(self, symbol):  # noqa: ARG002
        return SymbolInfo(symbol="EURUSD", digits=5, point=0.00001,
                          contract_size=100_000, volume_min=0.01, volume_max=100,
                          volume_step=0.01, currency_base="EUR",
                          currency_profit="USD", currency_margin="USD")
    async def list_symbols(self): return ["EURUSD"]
    async def subscribe_ticks(self, symbol): ...
    async def subscribe_bars(self, symbol, tf): ...
    async def unsubscribe(self, symbol, tf=None): ...
    async def get_history_bars(self, *a, **kw):
        from stinger_fx.data.parquet_store import BAR_SCHEMA
        return BAR_SCHEMA.empty_table()
    async def get_history_ticks(self, *a, **kw):
        from stinger_fx.data.parquet_store import TICK_SCHEMA
        return TICK_SCHEMA.empty_table()
    async def modify_order(self, ticket, **kw): raise NotImplementedError
    async def close_position(self, ticket, volume=None): raise NotImplementedError
    async def cancel_order(self, ticket): raise NotImplementedError
    async def get_positions(self) -> list[Position]: return []
    async def get_open_orders(self) -> list[Order]: return []

    async def place_order(self, req: OrderRequest) -> OrderResult:
        self.calls.append(req)
        order = Order(
            ticket=42 + len(self.calls),
            strategy_id=req.strategy_id,
            symbol=req.symbol,
            side=req.side,
            type=req.type,
            volume=req.volume,
            filled_volume=req.volume,
            fill_price=1.10,
            status=OrderStatus.FILLED,
            client_order_id=req.client_order_id,
            requested_at=datetime.now(UTC),
            filled_at=datetime.now(UTC),
        )
        return OrderResult(ok=True, ticket=order.ticket, status=OrderStatus.FILLED, order=order)


def _file_store(path: Path) -> SqliteStore:
    """Build a SqliteStore against an on-disk DB so two 'engine instances'
    can share state across a simulated restart."""
    store = SqliteStore(path)
    store.create_all()
    return store


@pytest.mark.asyncio
async def test_crash_then_replay_resubmits_pending(tmp_path: Path) -> None:
    """Instance 1 writes a pending row, dies. Instance 2 replays — broker
    sees the request, row transitions to 'sent'."""
    db = tmp_path / "stinger.db"
    store_1 = _file_store(db)

    # --- Instance 1: write a 'pending' row directly (simulating a crash
    # right after persistence but before broker submission completed).
    req = OrderRequest(
        strategy_id="s1",
        symbol="EURUSD",
        side=Side.BUY,
        type=OrderType.MARKET,
        volume=0.1,
        client_order_id="recover-1",
    )
    with store_1.session() as s:
        s.add(PendingOrderRequestRow(
            client_order_id=req.client_order_id,
            strategy_id=req.strategy_id,
            request_json=req.model_dump_json(),
            enqueued_at=datetime.now(UTC),
            status="pending",
        ))
        s.commit()

    # --- Instance 2: fresh process, same DB file
    bus = AsyncEventBus()
    store_2 = _file_store(db)
    broker = _AlwaysFillBroker(bus)
    queue = OrderQueue(store_2, broker_lookup=lambda _: broker)

    try:
        replayed = await queue.replay_pending()
        assert replayed == 1
        assert len(broker.calls) == 1
        assert broker.calls[0].client_order_id == "recover-1"
        # Row now 'sent'
        with store_2.session() as s:
            row = s.exec(
                select(PendingOrderRequestRow).where(
                    PendingOrderRequestRow.client_order_id == "recover-1"
                )
            ).first()
            assert row is not None
            assert row.status == "sent"
            assert row.broker_ticket is not None
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_router_with_queue_emits_filled_event(tmp_path: Path) -> None:
    """When OrderRouter has a queue attached, a signal still produces
    OrderFilledEvent (queue is transparent to consumers)."""
    db = tmp_path / "stinger.db"
    store = _file_store(db)
    bus = AsyncEventBus()
    broker = _AlwaysFillBroker(bus)
    queue = OrderQueue(store, broker_lookup=lambda _: broker)

    fills: list[OrderFilledEvent] = []

    async def collect(evt: OrderFilledEvent) -> None:
        fills.append(evt)

    sub = bus.subscribe(OrderFilledEvent, collect, name="t.fills")
    router = OrderRouter(bus, broker, strategy_magic={"s1": 1}, queue=queue)
    await router.attach()

    try:
        # Fire a signal through the bus
        await bus.publish(SignalEvent(signal=Signal(
            strategy_id="s1",
            time=datetime(2024, 1, 1, tzinfo=UTC),
            symbol="EURUSD",
            side=Side.BUY,
            strength=SignalStrength.NORMAL,
            suggested_volume=0.1,
        )))
        # Drain the bus
        for _ in range(5):
            await asyncio.sleep(0)

        assert len(fills) == 1
        assert len(broker.calls) == 1
        # Row recorded
        with store.session() as s:
            rows = list(s.exec(select(PendingOrderRequestRow)))
            assert len(rows) == 1
            assert rows[0].status == "sent"
    finally:
        await router.detach()
        await sub.unsubscribe()
        await bus.close()
