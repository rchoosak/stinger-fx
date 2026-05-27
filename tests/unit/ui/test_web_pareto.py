"""Web UI Pareto frontier viewer — /sweep/{id}/pareto."""

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
        return AccountInfo(account_id="x", broker="D", server="D",
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
    async def get_positions(self): return []
    async def get_open_orders(self): return []


def _seed_pareto_sweep(data_dir: Path, sweep_id: str = "multi") -> None:
    """Write a sweep summary that includes a Pareto block."""
    sweep_dir = data_dir / "sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "sweep_id": sweep_id,
        "strategy_id": "ma_multi",
        "rank_by": "net_pnl",
        "total_combos": 4,
        "started_at": "2024-01-01T00:00:00",
        "finished_at": "2024-01-01T00:01:00",
        "best_params": {"fast": 10, "slow": 30},
        "best_metric_value": 250.0,
        "top_n": [],
        "all": [],
        "pareto": {
            "objectives": [
                {"metric": "net_pnl", "direction": "max"},
                {"metric": "max_drawdown", "direction": "min"},
            ],
            "points": [
                {"params": {"fast": 5},  "metrics": {"net_pnl": 100.0, "max_drawdown": 5.0},
                 "is_pareto": True},
                {"params": {"fast": 10}, "metrics": {"net_pnl": 250.0, "max_drawdown": 12.0},
                 "is_pareto": True},
                {"params": {"fast": 8},  "metrics": {"net_pnl": 150.0, "max_drawdown": 10.0},
                 "is_pareto": False},
            ],
            "frontier_size": 2,
            "total_points": 3,
        },
    }
    (sweep_dir / f"{sweep_id}_summary.json").write_text(json.dumps(summary))


@pytest.fixture
def client_with_data(tmp_path: Path):
    bus = AsyncEventBus()
    broker = _StubBroker(bus)
    handle = EngineHandle(bus=bus, brokers=BrokerPool([("default", broker)]), runners={})
    app = create_app(handle, data_dir=tmp_path)
    return TestClient(app), tmp_path


def test_pareto_view_renders_scatter(client_with_data) -> None:
    client, data_dir = client_with_data
    _seed_pareto_sweep(data_dir, "multi_objective")
    r = client.get("/sweep/multi_objective/pareto")
    assert r.status_code == 200
    assert "Pareto frontier" in r.text
    assert "net_pnl" in r.text
    assert "max_drawdown" in r.text
    # Scatter canvas + chart lib
    assert 'id="pareto-chart"' in r.text
    assert "chart.js" in r.text
    # Frontier badge / table mentions both Pareto-optimal cells
    assert "fast=5" in r.text
    assert "fast=10" in r.text


def test_pareto_view_404_when_sweep_missing(client_with_data) -> None:
    client, _ = client_with_data
    r = client.get("/sweep/nonexistent/pareto")
    assert r.status_code == 404


def test_pareto_view_404_when_sweep_has_no_pareto(client_with_data) -> None:
    """A sweep without objectives → no Pareto block → 404 with explanation."""
    client, data_dir = client_with_data
    sweep_dir = data_dir / "sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "sweep_id": "single",
        "strategy_id": "ma",
        "rank_by": "net_pnl",
        "total_combos": 1,
        "finished_at": "2024-01-01T00:00:00",
        "all": [],
        "top_n": [],
    }
    (sweep_dir / "single_summary.json").write_text(json.dumps(summary))
    r = client.get("/sweep/single/pareto")
    assert r.status_code == 404
    assert "no Pareto frontier" in r.text or "objectives" in r.text
