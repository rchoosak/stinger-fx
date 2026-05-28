"""Web /audit — renders decisions, modifications, reconciliation mismatches."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from stinger_fx.brokers import BrokerPool
from stinger_fx.brokers.base import BaseBroker
from stinger_fx.core import AsyncEventBus
from stinger_fx.data import in_memory_store
from stinger_fx.data.schemas import (
    DecisionRow,
    OrderModificationRow,
    ReconciliationRow,
)
from stinger_fx.domain import (
    AccountInfo,
    AccountSnapshot,
    Order,
    OrderRequest,
    OrderResult,
    OrderStatus,
    Position,
    SymbolInfo,
)
from stinger_fx.ui.handle import EngineHandle
from stinger_fx.ui.web import create_app


class _StubBroker(BaseBroker):
    name = "stub"
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
    async def get_positions(self) -> list[Position]: return []
    async def get_open_orders(self) -> list[Order]: return []


@pytest.fixture
def client_with_store() -> tuple[TestClient, object]:
    bus = AsyncEventBus()
    broker = _StubBroker(bus)
    store = in_memory_store()
    handle = EngineHandle(bus=bus, brokers=BrokerPool([("default", broker)]), runners={})
    app = create_app(handle, sqlite_store=store)
    return TestClient(app), store


@pytest.fixture
def client_no_store() -> TestClient:
    bus = AsyncEventBus()
    broker = _StubBroker(bus)
    handle = EngineHandle(bus=bus, brokers=BrokerPool([("default", broker)]), runners={})
    app = create_app(handle)  # no sqlite_store
    return TestClient(app)


def test_audit_renders_empty_when_no_store(client_no_store: TestClient) -> None:
    """No store → page renders, shows "not configured" warning."""
    r = client_no_store.get("/audit")
    assert r.status_code == 200
    assert "not configured" in r.text.lower()


def test_audit_renders_seeded_mismatches(client_with_store) -> None:
    """Seed a ReconciliationRow → page shows it in the mismatches table."""
    client, store = client_with_store
    with store.session() as s:
        s.add(ReconciliationRow(
            ts=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            ticket=42,
            strategy_id="s1",
            mismatch_type="volume_drift",
            expected_value=0.1,
            actual_value=0.05,
            details="broker reports half the expected volume",
        ))
        s.commit()

    r = client.get("/audit")
    assert r.status_code == 200
    assert "volume_drift" in r.text
    assert "42" in r.text  # ticket
    assert "s1" in r.text
    assert "broker reports half" in r.text


def test_audit_renders_modifications_and_decisions(client_with_store) -> None:
    """Seed rows in all three tables — all three sections render."""
    client, store = client_with_store
    with store.session() as s:
        s.add(OrderModificationRow(
            ts=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            ticket=10,
            strategy_id="s1",
            modification_type="modify_sl_tp",
            reason="trailing",
        ))
        s.add(DecisionRow(
            ts=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            action="rejected",
            reason="max_open_positions_per_strategy=5 reached",
        ))
        s.commit()

    r = client.get("/audit")
    assert r.status_code == 200
    assert "modify_sl_tp" in r.text
    assert "trailing" in r.text
    assert "rejected" in r.text
    assert "max_open_positions" in r.text


def test_audit_empty_sections_show_placeholder(client_with_store) -> None:
    """Store is configured but empty — each section shows its 'No …' message."""
    client, _ = client_with_store
    r = client.get("/audit")
    assert r.status_code == 200
    assert "No reconciliation mismatches" in r.text
    assert "No order modifications" in r.text
    assert "No decisions" in r.text
