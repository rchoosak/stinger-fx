from __future__ import annotations

from datetime import UTC, datetime

import pytest

from stinger_fx.backtest.replay_broker import SimBroker
from stinger_fx.core import AsyncEventBus
from stinger_fx.domain import OrderRequest, OrderStatus, OrderType, Side


def _market_req(symbol: str = "XAUUSD") -> OrderRequest:
    return OrderRequest(
        strategy_id="s1",
        symbol=symbol,
        side=Side.BUY,
        type=OrderType.MARKET,
        volume=2.0,
        client_order_id=f"coid-{symbol}",
    )


@pytest.mark.asyncio
async def test_sim_broker_uses_default_contract_size_for_legacy_fx() -> None:
    bus = AsyncEventBus()
    broker = SimBroker(bus, initial_balance=10_000)
    try:
        broker.advance_clock(datetime(2024, 1, 1, tzinfo=UTC))
        broker.set_market("EURUSD", 1.1000)
        result = await broker.place_order(_market_req("EURUSD"))
        assert result.ok

        broker.set_market("EURUSD", 1.1010)
        close = await broker.close_position(result.ticket or 0)

        assert close.status == OrderStatus.FILLED
        assert broker.balance == pytest.approx(10_000 + 200.0)
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_sim_broker_uses_symbol_contract_size_for_mtm_and_close() -> None:
    bus = AsyncEventBus()
    broker = SimBroker(
        bus,
        initial_balance=10_000,
        symbol_contract_sizes={"XAUUSD": 100.0},
    )
    try:
        broker.advance_clock(datetime(2024, 1, 1, tzinfo=UTC))
        broker.set_market("XAUUSD", 2000.0)
        result = await broker.place_order(_market_req())
        assert result.ok

        broker.set_market("XAUUSD", 2001.0)
        snapshot = await broker.get_account_snapshot()
        assert snapshot.equity == pytest.approx(10_200.0)

        close = await broker.close_position(result.ticket or 0)
        assert close.status == OrderStatus.FILLED
        assert broker.balance == pytest.approx(10_200.0)
    finally:
        await bus.close()
