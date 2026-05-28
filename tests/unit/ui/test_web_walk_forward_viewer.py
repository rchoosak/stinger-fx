"""Web UI walk-forward viewer — /walkforward list + /walkforward/{id}."""

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


def _seed_wf(data_dir: Path, wf_id: str = "wf_smoke") -> None:
    wf_dir = data_dir / "walk_forward"
    wf_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "id": wf_id,
        "strategy_id": "ma_wf",
        "scheme": "expanding",
        "n_folds": 3,
        "rank_by": "net_pnl",
        "consistency_score": 0.65,
        "avg_oos_metric": 150.0,
        "started_at": "2024-01-01T00:00:00",
        "finished_at": "2024-01-01T00:05:00",
        "folds": [
            {
                "index": 0,
                "in_sample": ["2024-01-01T00:00:00", "2024-01-04T00:00:00"],
                "out_of_sample": ["2024-01-04T00:00:00", "2024-01-05T00:00:00"],
                "best_params": {"fast": 5, "slow": 20},
                "in_sample_metrics": {"net_pnl": 200.0},
                "oos_metrics": {"net_pnl": 100.0},
            },
            {
                "index": 1,
                "in_sample": ["2024-01-01T00:00:00", "2024-01-08T00:00:00"],
                "out_of_sample": ["2024-01-08T00:00:00", "2024-01-09T00:00:00"],
                "best_params": {"fast": 10, "slow": 30},
                "in_sample_metrics": {"net_pnl": 300.0},
                "oos_metrics": {"net_pnl": 180.0},
            },
            {
                "index": 2,
                "in_sample": ["2024-01-01T00:00:00", "2024-01-12T00:00:00"],
                "out_of_sample": ["2024-01-12T00:00:00", "2024-01-13T00:00:00"],
                "best_params": {"fast": 10, "slow": 30},
                "in_sample_metrics": {"net_pnl": 400.0},
                "oos_metrics": {"net_pnl": 170.0},
            },
        ],
    }
    (wf_dir / f"{wf_id}_summary.json").write_text(json.dumps(summary))


@pytest.fixture
def client_with_data(tmp_path: Path):
    bus = AsyncEventBus()
    broker = _StubBroker(bus)
    handle = EngineHandle(bus=bus, brokers=BrokerPool([("default", broker)]), runners={})
    app = create_app(handle, data_dir=tmp_path)
    return TestClient(app), tmp_path


def test_wf_list_empty(client_with_data) -> None:
    client, _ = client_with_data
    r = client.get("/walkforward")
    assert r.status_code == 200
    assert "No walk-forward runs" in r.text


def test_wf_list_shows_seeded_runs(client_with_data) -> None:
    client, data_dir = client_with_data
    _seed_wf(data_dir, "wf_a")
    _seed_wf(data_dir, "wf_b")
    r = client.get("/walkforward")
    assert r.status_code == 200
    assert "wf_a" in r.text
    assert "wf_b" in r.text


def test_wf_view_404_when_missing(client_with_data) -> None:
    client, _ = client_with_data
    r = client.get("/walkforward/missing")
    assert r.status_code == 404


def test_wf_view_renders_fold_breakdown(client_with_data) -> None:
    client, data_dir = client_with_data
    _seed_wf(data_dir, "wf_smoke")
    r = client.get("/walkforward/wf_smoke")
    assert r.status_code == 200
    assert "wf_smoke" in r.text
    assert "ma_wf" in r.text
    # Consistency score appears
    assert "+0.650" in r.text or "0.65" in r.text
    # All three folds listed
    for params in ("fast=5", "fast=10"):
        assert params in r.text
    # Chart canvas + chart.js included
    assert 'id="wf-chart"' in r.text
    assert "chart.js" in r.text
