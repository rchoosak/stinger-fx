"""Web UI sweep viewer — /sweep list + /sweep/{id} detail (Phase 6.3.D)."""

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


def _seed_sweep(data_dir: Path, sweep_id: str = "smoke") -> None:
    """Write a fake sweep summary JSON the viewer can consume."""
    sweep_dir = data_dir / "sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "sweep_id": sweep_id,
        "strategy_id": "ma_test",
        "rank_by": "net_pnl",
        "total_combos": 4,
        "started_at": "2024-01-01T00:00:00",
        "finished_at": "2024-01-01T00:01:00",
        "best_params": {"fast": 10, "slow": 30},
        "best_metric_value": 250.5,
        "top_n": [
            {"params": {"fast": 10, "slow": 30}, "metrics": {"net_pnl": 250.5}},
            {"params": {"fast": 5, "slow": 30}, "metrics": {"net_pnl": 100.0}},
            {"params": {"fast": 10, "slow": 20}, "metrics": {"net_pnl": 50.0}},
            {"params": {"fast": 5, "slow": 20}, "metrics": {"net_pnl": -25.0}},
        ],
        "all": [
            {"params": {"fast": 10, "slow": 30}, "metrics": {"net_pnl": 250.5}},
            {"params": {"fast": 5, "slow": 30}, "metrics": {"net_pnl": 100.0}},
            {"params": {"fast": 10, "slow": 20}, "metrics": {"net_pnl": 50.0}},
            {"params": {"fast": 5, "slow": 20}, "metrics": {"net_pnl": -25.0}},
        ],
    }
    (sweep_dir / f"{sweep_id}_summary.json").write_text(json.dumps(summary))


@pytest.fixture
def client_with_data(tmp_path: Path):
    bus = AsyncEventBus()
    broker = _StubBroker(bus)
    handle = EngineHandle(bus=bus, brokers=BrokerPool([("default", broker)]), runners={})
    app = create_app(handle, data_dir=tmp_path)
    return TestClient(app), tmp_path


def test_sweep_list_empty(client_with_data) -> None:
    client, _ = client_with_data
    r = client.get("/sweep")
    assert r.status_code == 200
    assert "No sweep runs" in r.text


def test_sweep_list_shows_seeded_runs(client_with_data) -> None:
    client, data_dir = client_with_data
    _seed_sweep(data_dir, "ma_grid_2024")
    _seed_sweep(data_dir, "rsi_optuna_2024")
    r = client.get("/sweep")
    assert r.status_code == 200
    assert "ma_grid_2024" in r.text
    assert "rsi_optuna_2024" in r.text


def test_sweep_view_404_when_missing(client_with_data) -> None:
    client, _ = client_with_data
    r = client.get("/sweep/nonexistent")
    assert r.status_code == 404


def test_sweep_view_renders_with_heatmap_for_2_params(client_with_data) -> None:
    """2 params → page must include the heatmap canvas + matrix chart lib."""
    client, data_dir = client_with_data
    _seed_sweep(data_dir, "ma_smoke")
    r = client.get("/sweep/ma_smoke")
    assert r.status_code == 200
    assert "ma_smoke" in r.text
    assert "ma_test" in r.text  # strategy_id
    # Best params rendered
    assert "fast=10" in r.text
    assert "slow=30" in r.text
    # 2D case → heatmap canvas + matrix chart
    assert 'id="heatmap"' in r.text
    assert "chartjs-chart-matrix" in r.text
    # Top-N table contains all 4 cells
    assert "250" in r.text  # best metric value
