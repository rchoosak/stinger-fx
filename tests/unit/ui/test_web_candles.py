"""Web UI candlestick endpoint — /backtest/{run_id}/candles.json."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stinger_fx.brokers import BrokerPool
from stinger_fx.brokers.base import BaseBroker
from stinger_fx.core import AsyncEventBus
from stinger_fx.data.parquet_store import ParquetStore
from stinger_fx.domain import (
    AccountInfo,
    AccountSnapshot,
    Bar,
    OrderResult,
    OrderStatus,
    SymbolInfo,
    Timeframe,
)
from stinger_fx.ui.handle import EngineHandle
from stinger_fx.ui.web import create_app


# Minimal stub broker — exact same shape as the other web test fixtures.
class _StubBroker(BaseBroker):
    name = "stub"
    async def connect(self): ...
    async def disconnect(self): ...
    async def is_connected(self): return True
    async def get_account_info(self):
        return AccountInfo(account_id="x", broker="Demo", server="Demo",
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
    async def place_order(self, req): return OrderResult(ok=False, status=OrderStatus.REJECTED)
    async def modify_order(self, ticket, **kw): raise NotImplementedError
    async def close_position(self, ticket, volume=None): raise NotImplementedError
    async def cancel_order(self, ticket): raise NotImplementedError
    async def get_positions(self): return []
    async def get_open_orders(self): return []


def _seed_bars(parquet_root: Path, symbol: str, base: datetime, n: int) -> None:
    """Write `n` M1 bars starting at `base` into the parquet store."""
    store = ParquetStore(parquet_root)
    bars = [
        Bar(
            symbol=symbol,
            timeframe=Timeframe.M1,
            time=base + timedelta(minutes=i),
            open=1.10 + i * 0.0001,
            high=1.10 + i * 0.0001 + 0.0002,
            low=1.10 + i * 0.0001 - 0.0001,
            close=1.10 + i * 0.0001 + 0.0001,
            tick_volume=100,
            is_closed=True,
        )
        for i in range(n)
    ]
    store.append_bars(symbol, Timeframe.M1, bars)


def _write_trades_sidecar(
    data_dir: Path, run_id: str, *, start: datetime, end: datetime
) -> None:
    bt_dir = data_dir / "backtests"
    bt_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "strategy_id": "test_strat",
        "symbol": "EURUSD",
        "timeframe": "M1",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "initial_balance": 10_000.0,
        "final_balance": 10_050.0,
        "trades": [],
    }
    (bt_dir / f"{run_id}_trades.json").write_text(json.dumps(meta))


@pytest.fixture
def client_with_data(tmp_path: Path) -> tuple[TestClient, Path]:
    bus = AsyncEventBus()
    broker = _StubBroker(bus)
    handle = EngineHandle(bus=bus, brokers=BrokerPool([("default", broker)]), runners={})
    app = create_app(handle, data_dir=tmp_path)
    return TestClient(app), tmp_path


def test_candles_endpoint_returns_bars(client_with_data) -> None:
    """When bars exist in parquet for the run's range, candles.json must
    return them as OHLC dicts."""
    client, data_dir = client_with_data
    base = datetime(2024, 1, 1, tzinfo=UTC)
    _seed_bars(data_dir / "parquet", "EURUSD", base, n=10)
    _write_trades_sidecar(
        data_dir, "smoke", start=base, end=base + timedelta(minutes=20)
    )

    r = client.get("/backtest/smoke/candles.json")
    assert r.status_code == 200
    body = r.json()
    assert "candles" in body
    candles = body["candles"]
    assert len(candles) == 10
    # First candle structure
    c0 = candles[0]
    assert {"time", "open", "high", "low", "close", "volume"} <= set(c0.keys())
    assert c0["open"] == pytest.approx(1.10)
    assert c0["high"] > c0["open"]
    assert c0["low"] < c0["open"]


def test_candles_endpoint_empty_when_no_parquet(client_with_data) -> None:
    """No parquet store → empty candles list (not 404). The frontend
    gracefully falls back to equity-only view."""
    client, data_dir = client_with_data
    base = datetime(2024, 1, 1, tzinfo=UTC)
    _write_trades_sidecar(
        data_dir, "smoke", start=base, end=base + timedelta(minutes=20)
    )

    r = client.get("/backtest/smoke/candles.json")
    assert r.status_code == 200
    assert r.json() == {"candles": []}


def test_candles_endpoint_empty_when_no_sidecar(client_with_data) -> None:
    """No trades sidecar → empty candles list (the run doesn't exist)."""
    client, _ = client_with_data
    r = client.get("/backtest/no_such_run/candles.json")
    assert r.status_code == 200
    assert r.json() == {"candles": []}


def test_replay_page_includes_candle_canvas(client_with_data) -> None:
    """The replay template must include the candle canvas + financial chart lib."""
    client, data_dir = client_with_data
    base = datetime(2024, 1, 1, tzinfo=UTC)
    _write_trades_sidecar(
        data_dir, "smoke", start=base, end=base + timedelta(minutes=20)
    )

    r = client.get("/backtest/smoke")
    assert r.status_code == 200
    assert 'id="candle-chart"' in r.text
    assert 'id="candle-status"' in r.text
    assert "chartjs-chart-financial" in r.text
    assert "/static/backtest_replay.js" in r.text
