"""OrderRouter — multi-account routing via the broker pool."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from stinger_fx.backtest.order_router import OrderRouter
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
    Side,
    Signal,
    SignalStrength,
    SymbolInfo,
)


class _CountingBroker(BaseBroker):
    """Records every place_order call so tests can assert routing."""

    name = "counting"

    def __init__(self, bus: AsyncEventBus, tag: str) -> None:
        super().__init__(bus)
        self.tag = tag
        self.calls: list[OrderRequest] = []

    async def connect(self): ...
    async def disconnect(self): ...
    async def is_connected(self): return True
    async def get_account_info(self) -> AccountInfo:
        return AccountInfo(
            account_id=self.tag, broker="t", server="t", currency="USD", leverage=100
        )
    async def get_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            account_id=self.tag, time=datetime.now(UTC),
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
        self.calls.append(req)
        return OrderResult(
            ok=True,
            ticket=len(self.calls),
            status=OrderStatus.FILLED,
            order=Order(
                ticket=len(self.calls),
                strategy_id=req.strategy_id,
                symbol=req.symbol,
                side=req.side,
                type=req.type,
                volume=req.volume,
                status=OrderStatus.FILLED,
            ),
        )
    async def modify_order(self, ticket, **kw): raise NotImplementedError
    async def close_position(self, ticket, volume=None): raise NotImplementedError
    async def cancel_order(self, ticket): raise NotImplementedError
    async def get_positions(self) -> list[Position]: return []
    async def get_open_orders(self) -> list[Order]: return []


def _signal(strategy_id: str) -> Signal:
    return Signal(
        strategy_id=strategy_id, time=datetime.now(UTC),
        symbol="EURUSD", side=Side.BUY,
        strength=SignalStrength.NORMAL, suggested_volume=0.1,
    )


@pytest.mark.asyncio
async def test_router_routes_signal_to_strategy_account() -> None:
    bus = AsyncEventBus()
    primary = _CountingBroker(bus, "primary")
    secondary = _CountingBroker(bus, "secondary")
    pool = BrokerPool([("primary", primary), ("secondary", secondary)])
    router = OrderRouter(
        bus, pool=pool,
        strategy_magic={"s1": 1, "s2": 2},
        strategy_accounts={"s1": "primary", "s2": "secondary"},
    )
    await router.handle_signal(_signal("s1"))
    await router.handle_signal(_signal("s2"))
    await router.handle_signal(_signal("s1"))
    await asyncio.sleep(0.02)
    assert len(primary.calls) == 2
    assert len(secondary.calls) == 1
    assert primary.calls[0].strategy_id == "s1"
    assert secondary.calls[0].strategy_id == "s2"
    await bus.close()


@pytest.mark.asyncio
async def test_unknown_strategy_falls_back_to_primary() -> None:
    bus = AsyncEventBus()
    primary = _CountingBroker(bus, "primary")
    secondary = _CountingBroker(bus, "secondary")
    pool = BrokerPool([("primary", primary), ("secondary", secondary)])
    router = OrderRouter(bus, pool=pool, strategy_accounts={"known": "secondary"})
    # `unknown_s` has no mapping → must hit the primary broker, not secondary.
    await router.handle_signal(_signal("unknown_s"))
    await asyncio.sleep(0.02)
    assert len(primary.calls) == 1
    assert len(secondary.calls) == 0
    await bus.close()


@pytest.mark.asyncio
async def test_router_legacy_broker_param_still_works() -> None:
    """Backward compat: single-broker callers (backtests) pass `broker=`."""
    bus = AsyncEventBus()
    broker = _CountingBroker(bus, "legacy")
    router = OrderRouter(bus, broker)
    await router.handle_signal(_signal("any_strategy"))
    await asyncio.sleep(0.02)
    assert len(broker.calls) == 1
    await bus.close()


def test_router_rejects_both_broker_and_pool() -> None:
    bus = AsyncEventBus()
    broker = _CountingBroker(bus, "a")
    pool = BrokerPool([("x", _CountingBroker(bus, "x"))])
    with pytest.raises(ValueError):
        OrderRouter(bus, broker, pool=pool)


def test_router_rejects_neither_broker_nor_pool() -> None:
    with pytest.raises(ValueError):
        OrderRouter(AsyncEventBus())  # neither broker= nor pool=
