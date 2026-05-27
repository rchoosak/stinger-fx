"""Web UI portfolio viewer — /portfolio + /portfolio/view + /portfolio/data.json."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

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
    async def connect(self): ...
    async def disconnect(self): ...
    async def is_connected(self): return True
    async def get_account_info(self):
        return AccountInfo(account_id="x", broker="Demo", server="D",
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
    async def place_order(self, req: OrderRequest) -> OrderResult:
        return OrderResult(ok=False, status=OrderStatus.REJECTED)
    async def modify_order(self, ticket, **kw): raise NotImplementedError
    async def close_position(self, ticket, volume=None): raise NotImplementedError
    async def cancel_order(self, ticket): raise NotImplementedError
    async def get_positions(self) -> list[Position]: return []
    async def get_open_orders(self) -> list[Order]: return []


def _seed_run(data_dir: Path, run_id: str, *, trades: list[dict], initial: float = 10_000.0) -> None:
    """Write a backtest sidecar that the portfolio aggregator can consume."""
    bt_dir = data_dir / "backtests"
    bt_dir.mkdir(parents=True, exist_ok=True)
    final = initial + sum(t["pnl"] for t in trades)
    meta = {
        "run_id": run_id,
        "strategy_id": f"strat_{run_id}",
        "symbol": "EURUSD",
        "timeframe": "M15",
        "start": "2024-01-01T00:00:00+00:00",
        "end": "2024-01-02T00:00:00+00:00",
        "initial_balance": initial,
        "final_balance": final,
        "trades": trades,
    }
    (bt_dir / f"{run_id}_trades.json").write_text(json.dumps(meta))


@pytest.fixture
def client_with_data(tmp_path: Path):
    bus = AsyncEventBus()
    broker = _StubBroker(bus)
    handle = EngineHandle(bus=bus, brokers=BrokerPool([("default", broker)]), runners={})
    app = create_app(handle, data_dir=tmp_path)
    return TestClient(app), tmp_path


def _seed_two_runs(data_dir: Path) -> None:
    base_t = datetime(2024, 1, 1, tzinfo=UTC)
    trades_a = [
        {"open_ts": base_t.isoformat(), "close_ts": base_t.isoformat(),
         "side": "buy", "open_price": 1.10, "close_price": 1.105, "volume": 0.1, "pnl": 5.0},
        {"open_ts": base_t.isoformat(), "close_ts": base_t.isoformat(),
         "side": "sell", "open_price": 1.10, "close_price": 1.095, "volume": 0.1, "pnl": 8.0},
    ]
    trades_b = [
        {"open_ts": base_t.isoformat(), "close_ts": base_t.isoformat(),
         "side": "buy", "open_price": 1.30, "close_price": 1.301, "volume": 0.1, "pnl": 3.0},
    ]
    _seed_run(data_dir, "a", trades=trades_a)
    _seed_run(data_dir, "b", trades=trades_b)


def test_portfolio_form_lists_available_runs(client_with_data) -> None:
    client, data_dir = client_with_data
    _seed_two_runs(data_dir)
    r = client.get("/portfolio")
    assert r.status_code == 200
    assert ">a<" in r.text or "code>a<" in r.text  # run id rendered
    assert ">b<" in r.text or "code>b<" in r.text


def test_portfolio_form_empty(client_with_data) -> None:
    client, _ = client_with_data
    r = client.get("/portfolio")
    assert r.status_code == 200
    assert "No backtest runs" in r.text or "Aggregate" in r.text


def test_portfolio_view_aggregates_runs(client_with_data) -> None:
    client, data_dir = client_with_data
    _seed_two_runs(data_dir)
    r = client.get("/portfolio/view?runs=a,b")
    assert r.status_code == 200
    assert "Combined equity curve" in r.text
    assert "Per-strategy contribution" in r.text
    # Both strategies appear
    assert "strat_a" in r.text
    assert "strat_b" in r.text


def test_portfolio_view_400_when_no_runs(client_with_data) -> None:
    client, _ = client_with_data
    r = client.get("/portfolio/view")
    assert r.status_code == 400


def test_portfolio_view_404_when_missing_run(client_with_data) -> None:
    client, data_dir = client_with_data
    _seed_two_runs(data_dir)
    r = client.get("/portfolio/view?runs=a,nonexistent")
    assert r.status_code == 404


def test_portfolio_data_json_returns_full_payload(client_with_data) -> None:
    client, data_dir = client_with_data
    _seed_two_runs(data_dir)
    r = client.get("/portfolio/data.json?runs=a,b")
    assert r.status_code == 200
    body = r.json()
    assert "contributions" in body
    assert "correlation_matrix" in body
    assert "equity_curve" in body
    assert len(body["contributions"]) == 2
