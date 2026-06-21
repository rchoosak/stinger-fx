"""OrderQueue — persisted outbox + idempotency + replay (Phase 6.1.C)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlmodel import select

from stinger_fx.brokers.base import BaseBroker
from stinger_fx.brokers.order_queue import OrderQueue
from stinger_fx.core import AsyncEventBus
from stinger_fx.data import in_memory_store
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
    SymbolInfo,
)


class _RecordingBroker(BaseBroker):
    """Stub broker that records every `place_order` call.

    Configurable behaviour:
      * ``results``  — list of OrderResult to return, one per call (popped
                       in order; if exhausted returns a default success)
      * ``calls``    — accumulated list of OrderRequest seen
      * ``raise_on`` — number of times to raise an exception before
                       returning normal results
    """

    name = "rec"

    def __init__(self, bus: AsyncEventBus) -> None:
        super().__init__(bus)
        self.results: list[OrderResult] = []
        self.calls: list[OrderRequest] = []
        self.raise_on = 0

    # BaseBroker boilerplate -------------------------------------------------
    async def connect(self): ...
    async def disconnect(self): ...
    async def is_connected(self): return True
    async def get_account_info(self):
        return AccountInfo(account_id="x", broker="r", server="r",
                           currency="USD", leverage=100)
    async def get_account_snapshot(self):
        return AccountSnapshot(account_id="x", time=datetime.now(UTC),
                               balance=10_000, equity=10_000, margin=0, free_margin=10_000)
    async def get_symbol_info(self, symbol):
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

    # The one we actually drive -------------------------------------------
    async def place_order(self, req: OrderRequest) -> OrderResult:
        self.calls.append(req)
        if self.raise_on > 0:
            self.raise_on -= 1
            raise RuntimeError("simulated broker outage")
        if self.results:
            return self.results.pop(0)
        # default success
        order = Order(
            ticket=999,
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
        return OrderResult(ok=True, ticket=999, status=OrderStatus.FILLED, order=order)


def _req(client_order_id: str = "coid-1") -> OrderRequest:
    return OrderRequest(
        strategy_id="s1",
        symbol="EURUSD",
        side=Side.BUY,
        type=OrderType.MARKET,
        volume=0.1,
        client_order_id=client_order_id,
    )


@pytest.mark.asyncio
async def test_submit_persists_and_forwards() -> None:
    """submit() must write a `sent` row and forward the call to the broker."""
    bus = AsyncEventBus()
    store = in_memory_store()
    broker = _RecordingBroker(bus)
    queue = OrderQueue(store, broker_lookup=lambda _: broker)

    try:
        result = await queue.submit(_req("coid-1"), broker)
        assert result.ok is True
        assert len(broker.calls) == 1
        # Row exists with status="sent"
        row = queue.row_for("coid-1")
        assert row is not None
        assert row.status == "sent"
        assert row.broker_ticket == 999
        assert row.attempts == 1
        assert row.completed_at is not None
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_duplicate_client_order_id_is_refused() -> None:
    """Second submit with the same client_order_id must be rejected without
    a second broker call."""
    bus = AsyncEventBus()
    store = in_memory_store()
    broker = _RecordingBroker(bus)
    queue = OrderQueue(store, broker_lookup=lambda _: broker)

    try:
        r1 = await queue.submit(_req("dup-1"), broker)
        assert r1.ok is True
        assert len(broker.calls) == 1

        r2 = await queue.submit(_req("dup-1"), broker)
        assert r2.ok is False
        assert r2.status == OrderStatus.REJECTED
        assert "duplicate" in r2.message.lower()
        # No second broker call
        assert len(broker.calls) == 1
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_broker_rejection_marks_failed() -> None:
    """A broker reject (ok=False) must update the row to status="failed"
    with the reason in last_error."""
    bus = AsyncEventBus()
    store = in_memory_store()
    broker = _RecordingBroker(bus)
    broker.results = [
        OrderResult(ok=False, status=OrderStatus.REJECTED, message="insufficient margin")
    ]
    queue = OrderQueue(store, broker_lookup=lambda _: broker)

    try:
        result = await queue.submit(_req("fail-1"), broker)
        assert result.ok is False
        row = queue.row_for("fail-1")
        assert row is not None
        assert row.status == "failed"
        assert "insufficient margin" in row.last_error
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_broker_exception_marks_failed_and_reraises() -> None:
    """If broker.place_order raises, row is marked failed AND exception propagates."""
    bus = AsyncEventBus()
    store = in_memory_store()
    broker = _RecordingBroker(bus)
    broker.raise_on = 1
    queue = OrderQueue(store, broker_lookup=lambda _: broker)

    try:
        with pytest.raises(RuntimeError, match="simulated broker outage"):
            await queue.submit(_req("err-1"), broker)
        row = queue.row_for("err-1")
        assert row is not None
        assert row.status == "failed"
        assert "RuntimeError" in row.last_error
        assert "simulated broker outage" in row.last_error
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_replay_pending_resubmits_only_pending() -> None:
    """replay_pending() must re-fire only rows still in 'pending' state."""
    bus = AsyncEventBus()
    store = in_memory_store()
    broker = _RecordingBroker(bus)
    queue = OrderQueue(store, broker_lookup=lambda _: broker)

    # Seed three rows directly (simulating a previous run that wrote them
    # but crashed before submission completed)
    with store.session() as s:
        for i, status in enumerate(["pending", "sent", "pending"]):
            req = _req(f"replay-{i}")
            s.add(PendingOrderRequestRow(
                client_order_id=req.client_order_id,
                strategy_id=req.strategy_id,
                request_json=req.model_dump_json(),
                enqueued_at=datetime.now(UTC),
                status=status,
            ))
        s.commit()

    try:
        replayed = await queue.replay_pending()
        assert replayed == 2  # only the two 'pending' rows
        assert len(broker.calls) == 2
        # Both pending rows should now be "sent"
        with store.session() as s:
            rows = list(s.exec(select(PendingOrderRequestRow)))
        statuses = {r.client_order_id: r.status for r in rows}
        assert statuses == {"replay-0": "sent", "replay-1": "sent", "replay-2": "sent"}
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_replay_pending_routes_by_strategy_lookup() -> None:
    bus = AsyncEventBus()
    store = in_memory_store()
    primary = _RecordingBroker(bus)
    secondary = _RecordingBroker(bus)
    brokers = {"s1": primary, "s2": secondary}
    queue = OrderQueue(store, broker_lookup=lambda sid: brokers[sid])

    with store.session() as s:
        req = _req("route-1").model_copy(update={"strategy_id": "s2"})
        s.add(PendingOrderRequestRow(
            client_order_id=req.client_order_id,
            strategy_id=req.strategy_id,
            request_json=req.model_dump_json(),
            enqueued_at=datetime.now(UTC),
            status="pending",
        ))
        s.commit()

    try:
        replayed = await queue.replay_pending()
        assert replayed == 1
        assert primary.calls == []
        assert [c.client_order_id for c in secondary.calls] == ["route-1"]
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_pending_count_reflects_state() -> None:
    """pending_count() returns the number of un-completed rows."""
    bus = AsyncEventBus()
    store = in_memory_store()
    broker = _RecordingBroker(bus)
    queue = OrderQueue(store, broker_lookup=lambda _: broker)

    assert queue.pending_count() == 0

    # Seed two pending + one sent
    with store.session() as s:
        for i, status in enumerate(["pending", "sent", "pending"]):
            req = _req(f"cnt-{i}")
            s.add(PendingOrderRequestRow(
                client_order_id=req.client_order_id,
                strategy_id=req.strategy_id,
                request_json=req.model_dump_json(),
                enqueued_at=datetime.now(UTC),
                status=status,
            ))
        s.commit()

    try:
        assert queue.pending_count() == 2
    finally:
        await bus.close()
