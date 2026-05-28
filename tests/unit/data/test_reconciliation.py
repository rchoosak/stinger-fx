"""Reconciler — detects broker / DB mismatches after fills (Phase 6.1.D)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from stinger_fx.brokers.base import BaseBroker
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import (
    OrderFilledEvent,
    PositionClosedEvent,
    ReconciliationMismatchEvent,
)
from stinger_fx.data import (
    Reconciler,
    ReconciliationRepo,
    in_memory_store,
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
)


class _StubBroker(BaseBroker):
    """Stub broker whose `get_positions` returns whatever the test wants."""

    name = "stub"

    def __init__(self, bus: AsyncEventBus) -> None:
        super().__init__(bus)
        self._positions: list[Position] = []

    def set_positions(self, positions: list[Position]) -> None:
        self._positions = list(positions)

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
    async def place_order(self, req: OrderRequest) -> OrderResult:
        return OrderResult(ok=False, status=OrderStatus.REJECTED)
    async def modify_order(self, ticket, **kw): raise NotImplementedError
    async def close_position(self, ticket, volume=None): raise NotImplementedError
    async def cancel_order(self, ticket): raise NotImplementedError
    async def get_positions(self) -> list[Position]: return list(self._positions)
    async def get_open_orders(self) -> list[Order]: return []


def _make_order(
    ticket: int = 1, volume: float = 0.1, fill_price: float = 1.10
) -> Order:
    return Order(
        ticket=ticket,
        strategy_id="s1",
        symbol="EURUSD",
        side=Side.BUY,
        type=OrderType.MARKET,
        volume=volume,
        filled_volume=volume,
        fill_price=fill_price,
        status=OrderStatus.FILLED,
        requested_at=datetime.now(UTC),
        filled_at=datetime.now(UTC),
    )


def _make_position(
    ticket: int, volume: float, open_price: float = 1.10
) -> Position:
    return Position(
        ticket=ticket,
        symbol="EURUSD",
        side=Side.BUY,
        volume=volume,
        open_price=open_price,
        open_time=datetime.now(UTC),
        magic=0,
    )


async def _drain(bus: AsyncEventBus, *, ticks: int = 3) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_match_records_nothing() -> None:
    """Broker shows exactly what we filled — no mismatch."""
    bus = AsyncEventBus()
    broker = _StubBroker(bus)
    store = in_memory_store()
    rec = Reconciler(bus, broker, store, verify_delay_seconds=0.01)
    await rec.start()

    try:
        order = _make_order(ticket=1, volume=0.1, fill_price=1.10)
        broker.set_positions([_make_position(1, 0.1, 1.10)])
        await bus.publish(OrderFilledEvent(order=order))
        # Wait past verify delay
        await asyncio.sleep(0.05)
        await _drain(bus)

        rows = ReconciliationRepo(store).recent(10)
        assert rows == []
    finally:
        await rec.stop()
        await bus.close()


@pytest.mark.asyncio
async def test_missing_position_records_mismatch() -> None:
    """Broker shows no matching position — record position_missing."""
    bus = AsyncEventBus()
    broker = _StubBroker(bus)
    store = in_memory_store()
    rec = Reconciler(bus, broker, store, verify_delay_seconds=0.01)
    await rec.start()

    mismatches: list[ReconciliationMismatchEvent] = []

    async def collect(evt: ReconciliationMismatchEvent) -> None:
        mismatches.append(evt)

    sub = bus.subscribe(ReconciliationMismatchEvent, collect, name="t.mm")

    try:
        order = _make_order(ticket=5, volume=0.2)
        broker.set_positions([])  # empty!
        await bus.publish(OrderFilledEvent(order=order))
        await asyncio.sleep(0.05)
        await _drain(bus)

        rows = ReconciliationRepo(store).recent(10)
        assert len(rows) == 1
        assert rows[0].mismatch_type == "position_missing"
        assert rows[0].ticket == 5
        assert rows[0].expected_value == pytest.approx(0.2)
        assert rows[0].actual_value == pytest.approx(0.0)

        assert len(mismatches) == 1
        assert mismatches[0].mismatch_type == "position_missing"
    finally:
        await sub.unsubscribe()
        await rec.stop()
        await bus.close()


@pytest.mark.asyncio
async def test_volume_drift_records_mismatch() -> None:
    """Broker reports a different volume → record volume_drift."""
    bus = AsyncEventBus()
    broker = _StubBroker(bus)
    store = in_memory_store()
    rec = Reconciler(bus, broker, store, verify_delay_seconds=0.01)
    await rec.start()

    try:
        order = _make_order(ticket=7, volume=0.1)
        broker.set_positions([_make_position(7, 0.05)])  # half what we expected!
        await bus.publish(OrderFilledEvent(order=order))
        await asyncio.sleep(0.05)
        await _drain(bus)

        rows = ReconciliationRepo(store).recent(10)
        assert len(rows) == 1
        assert rows[0].mismatch_type == "volume_drift"
        assert rows[0].expected_value == pytest.approx(0.1)
        assert rows[0].actual_value == pytest.approx(0.05)
    finally:
        await rec.stop()
        await bus.close()


@pytest.mark.asyncio
async def test_price_drift_records_mismatch() -> None:
    """Broker open_price differs from our fill_price beyond tolerance."""
    bus = AsyncEventBus()
    broker = _StubBroker(bus)
    store = in_memory_store()
    # 1 pip tolerance — broker reports 5 pips off, should fire
    rec = Reconciler(bus, broker, store, verify_delay_seconds=0.01,
                     price_tolerance_pips=1.0, point=0.0001)
    await rec.start()

    try:
        order = _make_order(ticket=11, volume=0.1, fill_price=1.10)
        broker.set_positions([_make_position(11, 0.1, open_price=1.1005)])
        await bus.publish(OrderFilledEvent(order=order))
        await asyncio.sleep(0.05)
        await _drain(bus)

        rows = ReconciliationRepo(store).recent(10)
        assert len(rows) == 1
        assert rows[0].mismatch_type == "price_drift"
        assert rows[0].expected_value == pytest.approx(1.10)
        assert rows[0].actual_value == pytest.approx(1.1005)
    finally:
        await rec.stop()
        await bus.close()


@pytest.mark.asyncio
async def test_close_before_verify_cancels_check() -> None:
    """If a position closes before the verify delay elapses, the verify is
    cancelled — no mismatch recorded for a perfectly normal fast-exit trade."""
    bus = AsyncEventBus()
    broker = _StubBroker(bus)
    store = in_memory_store()
    rec = Reconciler(bus, broker, store, verify_delay_seconds=0.5)  # longer delay
    await rec.start()

    try:
        order = _make_order(ticket=20, volume=0.1)
        broker.set_positions([])  # broker would report missing if we waited
        await bus.publish(OrderFilledEvent(order=order))

        # Quickly close the position before verify_delay elapses
        await asyncio.sleep(0.05)
        pos = _make_position(20, 0.1)
        await bus.publish(PositionClosedEvent(position=pos, realized_pnl=5.0))
        await _drain(bus)

        # Wait past the full delay to confirm nothing fires
        await asyncio.sleep(0.6)
        await _drain(bus)

        rows = ReconciliationRepo(store).recent(10)
        assert rows == [], (
            f"expected no mismatch — quick close should cancel verify; got: {rows}"
        )
    finally:
        await rec.stop()
        await bus.close()
