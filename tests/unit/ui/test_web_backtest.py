"""Web UI trade-replay views — list + replay page + data.json endpoint."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stinger_fx.backtest.reports import BacktestReport, TradeRecord
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
            account_id="x", broker="Demo", server="Demo",
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
    async def get_positions(self): return []
    async def get_open_orders(self) -> list[Order]: return []


def _write_trades_sidecar(data_dir: Path, run_id: str, trade_count: int) -> dict:
    """Materialise a `<data_dir>/backtests/<run_id>_trades.json` + metrics +
    equity Parquet so the views have something to render."""
    bt_dir = data_dir / "backtests"
    bt_dir.mkdir(parents=True, exist_ok=True)
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    trades = [
        TradeRecord(
            open_ts=t0 + timedelta(hours=i),
            close_ts=t0 + timedelta(hours=i + 1),
            side="buy" if i % 2 == 0 else "sell",
            open_price=1.10 + 0.001 * i,
            close_price=1.10 + 0.001 * i + (0.0005 if i % 2 == 0 else -0.0003),
            volume=0.1,
            pnl=50 if i % 2 == 0 else -30,
        )
        for i in range(trade_count)
    ]
    report = BacktestReport(
        run_id=run_id,
        strategy_id="ma_test",
        started_at=t0,
        finished_at=t0 + timedelta(hours=trade_count),
        trades=trades,
        equity_curve=[
            (t0 + timedelta(hours=i), 10_000 + sum(t.pnl for t in trades[:i]))
            for i in range(trade_count + 1)
        ],
        initial_balance=10_000.0,
        final_balance=10_000.0 + sum(t.pnl for t in trades),
    )
    report.write_equity_curve(bt_dir / f"{run_id}_equity.parquet")
    (bt_dir / f"{run_id}_metrics.json").write_text(
        json.dumps(report.to_metrics_dict(), indent=2)
    )
    meta = {
        "run_id": run_id,
        "strategy_id": "ma_test",
        "symbol": "EURUSD",
        "timeframe": "M15",
        "start": t0.isoformat(),
        "end": (t0 + timedelta(hours=trade_count)).isoformat(),
        "initial_balance": 10_000.0,
        "final_balance": report.final_balance,
        "trades": report.trades_to_jsonable(),
    }
    (bt_dir / f"{run_id}_trades.json").write_text(json.dumps(meta, indent=2))
    return meta


@pytest.fixture
def client_with_data(tmp_path: Path) -> tuple[TestClient, Path]:
    bus = AsyncEventBus()
    broker = _StubBroker(bus)
    handle = EngineHandle(
        bus=bus, brokers=BrokerPool([("default", broker)]), runners={}
    )
    app = create_app(handle, data_dir=tmp_path)
    return TestClient(app), tmp_path


def test_backtest_list_empty_renders(client_with_data) -> None:
    client, _ = client_with_data
    r = client.get("/backtest")
    assert r.status_code == 200
    assert "No completed backtest runs" in r.text


def test_backtest_list_shows_seeded_runs(client_with_data) -> None:
    client, data_dir = client_with_data
    _write_trades_sidecar(data_dir, "smoke_a", trade_count=4)
    _write_trades_sidecar(data_dir, "smoke_b", trade_count=2)
    r = client.get("/backtest")
    assert r.status_code == 200
    assert "smoke_a" in r.text
    assert "smoke_b" in r.text
    # Both runs link out to /backtest/<id>
    assert 'href="/backtest/smoke_a"' in r.text
    assert 'href="/backtest/smoke_b"' in r.text


def test_replay_page_renders_chart_canvas(client_with_data) -> None:
    client, data_dir = client_with_data
    _write_trades_sidecar(data_dir, "smoke", trade_count=3)
    r = client.get("/backtest/smoke")
    assert r.status_code == 200
    assert "EURUSD" in r.text
    # The canvas exists for Chart.js
    assert 'id="replay-chart"' in r.text
    # Both chart.js and the date adapter are loaded from CDN
    assert "chart.js" in r.text
    assert "chartjs-adapter-date-fns" in r.text


def test_replay_page_404_when_no_sidecar(client_with_data) -> None:
    client, _ = client_with_data
    r = client.get("/backtest/nonexistent_run_id")
    assert r.status_code == 404


def test_data_json_returns_full_payload(client_with_data) -> None:
    client, data_dir = client_with_data
    _write_trades_sidecar(data_dir, "smoke", trade_count=5)
    r = client.get("/backtest/smoke/data.json")
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["run_id"] == "smoke"
    assert body["meta"]["symbol"] == "EURUSD"
    assert len(body["meta"]["trades"]) == 5
    # Equity curve has the expected number of points (trade_count + 1)
    assert len(body["equity"]) == 6
    # Metrics carry the standard keys
    assert "net_pnl" in body["metrics"]
    assert "sharpe" in body["metrics"]


def test_data_json_404_when_no_sidecar(client_with_data) -> None:
    client, _ = client_with_data
    r = client.get("/backtest/missing/data.json")
    assert r.status_code == 404
