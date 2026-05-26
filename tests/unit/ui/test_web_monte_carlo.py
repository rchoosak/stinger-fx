"""Web UI Monte Carlo endpoint — /backtest/{id}/monte_carlo.json."""

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


def _seed_trades(data_dir: Path, run_id: str, trades: list[dict]) -> None:
    bt_dir = data_dir / "backtests"
    bt_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "strategy_id": "ma_test",
        "symbol": "EURUSD",
        "timeframe": "M15",
        "start": "2024-01-01T00:00:00",
        "end": "2024-02-01T00:00:00",
        "initial_balance": 10_000.0,
        "final_balance": 10_000.0 + sum(t["pnl"] for t in trades),
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


def test_mc_endpoint_returns_bands(client_with_data) -> None:
    client, data_dir = client_with_data
    trades = [{"pnl": 10.0}, {"pnl": -5.0}, {"pnl": 8.0}, {"pnl": -3.0}, {"pnl": 12.0}]
    _seed_trades(data_dir, "smoke", trades)
    r = client.get("/backtest/smoke/monte_carlo.json?n=50&seed=42")
    assert r.status_code == 200
    body = r.json()
    assert body["n_simulations"] == 50
    assert body["n_trades"] == 5
    for metric in ("net_pnl", "max_drawdown", "sharpe"):
        assert {"p5", "p50", "p95", "mean"} <= set(body[metric].keys())
    assert len(body["equity_envelope"]["low"]) == 5
    assert len(body["equity_envelope"]["mid"]) == 5
    assert len(body["equity_envelope"]["high"]) == 5


def test_mc_endpoint_empty_when_no_trades(client_with_data) -> None:
    client, data_dir = client_with_data
    _seed_trades(data_dir, "empty", [])
    r = client.get("/backtest/empty/monte_carlo.json")
    assert r.status_code == 200
    body = r.json()
    assert body["n_trades"] == 0
    assert "error" in body


def test_mc_endpoint_404_when_no_sidecar(client_with_data) -> None:
    client, _ = client_with_data
    r = client.get("/backtest/missing/monte_carlo.json")
    assert r.status_code == 404


def test_mc_endpoint_caps_n_simulations(client_with_data) -> None:
    """Requested n > 5000 must be clamped to keep latency bounded."""
    client, data_dir = client_with_data
    _seed_trades(data_dir, "cap", [{"pnl": 1.0}, {"pnl": -1.0}])
    r = client.get("/backtest/cap/monte_carlo.json?n=100000")
    assert r.status_code == 200
    assert r.json()["n_simulations"] == 5000
