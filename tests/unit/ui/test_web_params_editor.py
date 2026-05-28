"""Web UI live params editor — GET form, POST update, error rendering."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from pydantic import Field

from stinger_fx.brokers import BrokerPool
from stinger_fx.brokers.base import BaseBroker
from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.domain import (
    AccountInfo,
    AccountSnapshot,
    Order,
    OrderRequest,
    OrderResult,
    OrderStatus,
    Position,
    Subscription,
    SymbolInfo,
    Timeframe,
)
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.parameters import StrategyParams
from stinger_fx.strategies.runner import StrategyRunner
from stinger_fx.ui.handle import EngineHandle
from stinger_fx.ui.web import create_app

# --- Test strategy with a few typed param fields ----------------------------


class _DemoParams(StrategyParams):
    fast: int = Field(10, ge=2, le=200)
    slow: int = Field(30, ge=5, le=500)
    risk: float = Field(0.01, gt=0, le=0.1)
    enabled: bool = True


class _DemoStrategy(BaseStrategy):
    name = "demo_strategy"
    Params = _DemoParams

    @classmethod
    def subscriptions(cls, params):
        return [Subscription(symbol="EURUSD", timeframe=Timeframe.M1)]


# --- Stub broker (minimum to satisfy the engine handle) ---------------------


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
    async def place_order(self, req: OrderRequest) -> OrderResult:
        return OrderResult(ok=False, status=OrderStatus.REJECTED)
    async def modify_order(self, ticket, **kw): raise NotImplementedError
    async def close_position(self, ticket, volume=None): raise NotImplementedError
    async def cancel_order(self, ticket): raise NotImplementedError
    async def get_positions(self) -> list[Position]: return []
    async def get_open_orders(self) -> list[Order]: return []


# --- Fixture: live runner with the demo strategy ----------------------------


@pytest.fixture
def client_with_runner():
    bus = AsyncEventBus()
    broker = _StubBroker(bus)
    strategy = _DemoStrategy()
    sid = "demo_strat_1"
    runner = StrategyRunner(
        strategy_id=sid,
        strategy=strategy,
        params=_DemoParams(),
        bus=bus,
        clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)),
        reload_lock=asyncio.Lock(),
        signal_sink=lambda s: asyncio.sleep(0),
    )
    # The handle expects a runner in self.runners; the runner doesn't need to
    # be started for the params editor endpoints to work.
    handle = EngineHandle(
        bus=bus,
        brokers=BrokerPool([("default", broker)]),
        runners={sid: runner},
    )
    app = create_app(handle)
    return TestClient(app), runner, sid


# --- GET /strategy/{sid}/params ---------------------------------------------


def test_params_form_renders_current_values(client_with_runner) -> None:
    client, _, sid = client_with_runner
    r = client.get(f"/strategy/{sid}/params")
    assert r.status_code == 200
    # All four fields should appear in the form
    assert "fast" in r.text
    assert "slow" in r.text
    assert "risk" in r.text
    assert "enabled" in r.text
    # Default values are pre-filled
    assert 'value="10"' in r.text   # fast default
    assert 'value="30"' in r.text   # slow default


def test_params_form_404_for_unknown_strategy(client_with_runner) -> None:
    client, _, _ = client_with_runner
    r = client.get("/strategy/no_such_strategy/params")
    assert r.status_code == 404


# --- POST /strategy/{sid}/params --------------------------------------------


def test_params_post_updates_runner_atomically(client_with_runner) -> None:
    """Submitting a valid form must update the runner's params in place."""
    client, runner, sid = client_with_runner

    r = client.post(
        f"/strategy/{sid}/params",
        data={"fast": "20", "slow": "60", "risk": "0.02", "enabled": "1"},
    )
    assert r.status_code == 200
    # After a successful update the response swaps the strategies panel back in
    assert "Strategies" in r.text
    # Runner's params should reflect the new values
    assert runner._params.fast == 20
    assert runner._params.slow == 60
    assert runner._params.risk == pytest.approx(0.02)


def test_params_post_rejects_invalid_value(client_with_runner) -> None:
    """Invalid input must re-render the form with an error banner; the
    runner's params remain unchanged."""
    client, runner, sid = client_with_runner

    r = client.post(
        f"/strategy/{sid}/params",
        data={"fast": "999", "slow": "30", "risk": "0.01", "enabled": "1"},  # fast > 200
    )
    assert r.status_code == 200
    # Error banner is present
    assert "invalid params" in r.text.lower() or "less than or equal" in r.text.lower()
    # Runner's params unchanged
    assert runner._params.fast == 10


def test_params_post_partial_update(client_with_runner) -> None:
    """Only changing one field should leave the others alone."""
    client, runner, sid = client_with_runner

    # Only change `slow`; `fast`, `risk`, `enabled` are still submitted by the
    # form (the template renders all fields), so we send them with their
    # current values too.
    r = client.post(
        f"/strategy/{sid}/params",
        data={"fast": "10", "slow": "60", "risk": "0.01", "enabled": "1"},
    )
    assert r.status_code == 200
    assert runner._params.fast == 10
    assert runner._params.slow == 60
    assert runner._params.risk == pytest.approx(0.01)
    assert runner._params.enabled is True


def test_params_post_404_for_unknown_strategy(client_with_runner) -> None:
    client, _, _ = client_with_runner
    r = client.post("/strategy/no_such/params", data={"fast": "10"})
    assert r.status_code == 404
