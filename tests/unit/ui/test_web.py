"""FastAPI web UI smoke tests using the TestClient.

We don't spin up uvicorn — instead we use Starlette's TestClient which drives
the ASGI app in-process. This is enough to verify routing, template rendering,
and partial HTMX responses without binding a port.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from stinger_fx.brokers.base import BaseBroker
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import AccountSnapshotEvent
from stinger_fx.domain import (
    AccountInfo,
    AccountSnapshot,
    Order,
    OrderRequest,
    OrderResult,
    OrderStatus,
    Position,
    Side,
    SymbolInfo,
    Timeframe,
)
from stinger_fx.ui.handle import EngineHandle
from stinger_fx.ui.web import create_app


class _StubBroker(BaseBroker):
    name = "stub"
    fake_positions: list[Position] = []

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def is_connected(self) -> bool: return True

    async def get_account_info(self) -> AccountInfo:
        return AccountInfo(
            account_id="x", broker="DemoBroker", server="DemoServer",
            currency="USD", leverage=100,
        )

    async def get_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            account_id="x", time=datetime.now(UTC),
            balance=10_000, equity=10_002, margin=0, free_margin=10_002, profit=2.0,
        )

    async def get_symbol_info(self, symbol):  # noqa: ARG002
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

    async def get_positions(self) -> list[Position]:
        return list(self.fake_positions)

    async def get_open_orders(self) -> list[Order]:
        return []


@pytest.fixture
def client() -> TestClient:
    bus = AsyncEventBus()
    broker = _StubBroker(bus)
    handle = EngineHandle(bus=bus, broker=broker, runners={})
    app = create_app(handle)
    return TestClient(app)


def test_dashboard_renders(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "Stinger-Fx" in r.text
    assert "Account" in r.text
    assert "Strategies" in r.text
    assert "Positions" in r.text
    assert "Events" in r.text
    # HTMX is loaded
    assert "htmx.org" in r.text
    # SSE extension is loaded
    assert "ext/sse" in r.text


def test_partial_account_renders_broker_meta(client: TestClient) -> None:
    r = client.get("/partial/account")
    assert r.status_code == 200
    assert "DemoBroker" in r.text
    assert "USD" in r.text


def test_partial_account_uses_cached_snapshot_when_present(client: TestClient) -> None:
    # Initially no snapshot → "—" placeholders
    r = client.get("/partial/account")
    assert "—" in r.text

    # Set the cached snapshot directly. (Live engine populates this via the
    # bus subscription installed by the FastAPI lifespan handler — we test
    # the rendering path here; the bus-cache wiring has its own integration
    # test in tests/unit/test_runtime_wiring.py.)
    client.app.state.latest_snapshot = AccountSnapshot(
        account_id="x", time=datetime.now(UTC),
        balance=12_345.0, equity=12_400.0, margin=0, free_margin=12_400.0, profit=55.0,
    )

    r2 = client.get("/partial/account")
    assert "12,345.00" in r2.text
    assert "12,400.00" in r2.text


def test_partial_strategies_renders_empty(client: TestClient) -> None:
    r = client.get("/partial/strategies")
    assert r.status_code == 200
    assert "no strategies configured" in r.text


def test_partial_positions_renders_with_rows(client: TestClient) -> None:
    broker: _StubBroker = client.app.state.handle.broker  # type: ignore[assignment]
    broker.fake_positions = [
        Position(
            ticket=99, symbol="EURUSD", side=Side.BUY, volume=0.5,
            open_price=1.10, open_time=datetime.now(UTC), profit=12.5,
        )
    ]
    r = client.get("/partial/positions")
    assert r.status_code == 200
    assert "EURUSD" in r.text
    assert "BUY" in r.text
    # Profit color class applied
    assert "profit" in r.text


def test_pause_resume_returns_strategies_partial(client: TestClient) -> None:
    # Unknown strategy → 404
    assert client.post("/strategy/nope/pause").status_code == 404
    assert client.post("/strategy/nope/resume").status_code == 404


def test_static_css_is_served(client: TestClient) -> None:
    r = client.get("/static/style.css")
    assert r.status_code == 200
    assert "body" in r.text
