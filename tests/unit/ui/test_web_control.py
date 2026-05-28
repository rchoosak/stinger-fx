"""Web control plane — /health + /control/shutdown."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from stinger_fx.brokers import BrokerPool
from stinger_fx.brokers.base import BaseBroker
from stinger_fx.core import AsyncEventBus
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
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def is_connected(self) -> bool: return True
    async def get_account_info(self) -> AccountInfo:
        return AccountInfo(
            account_id="x", broker="Demo", server="DemoSrv",
            currency="USD", leverage=100,
        )
    async def get_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            account_id="x", time=datetime.now(UTC),
            balance=10_000, equity=10_000, margin=0, free_margin=10_000,
        )
    async def get_symbol_info(self, symbol):
        return SymbolInfo(
            symbol="EURUSD", digits=5, point=0.00001, contract_size=100_000,
            volume_min=0.01, volume_max=100, volume_step=0.01,
            currency_base="EUR", currency_profit="USD", currency_margin="USD",
        )
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
def client():
    bus = AsyncEventBus()
    broker = _StubBroker(bus)
    handle = EngineHandle(
        bus=bus, brokers=BrokerPool([("default", broker)]), runners={}
    )
    app = create_app(handle)
    return TestClient(app)


def test_health_returns_engine_state(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "started_at" in body
    assert body["strategies"] == 0
    assert isinstance(body["accounts"], list)


def test_shutdown_endpoint_returns_shutting_down(client: TestClient, monkeypatch) -> None:
    """Don't actually send SIGINT during the test — patch os.kill."""
    import os

    import stinger_fx.ui.web.server as server_mod  # noqa: F401

    fired: list[tuple[int, int]] = []

    def fake_kill(pid: int, sig: int) -> None:
        fired.append((pid, sig))

    monkeypatch.setattr(os, "kill", fake_kill)
    r = client.post("/control/shutdown")
    assert r.status_code == 200
    assert r.json() == {"status": "shutting_down"}
    # The actual signal fires from a 50ms-deferred task — give it time.
    import time

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not fired:
        time.sleep(0.05)
    assert fired, "shutdown task did not deliver the signal within 1s"
    assert fired[0][0] == os.getpid()
