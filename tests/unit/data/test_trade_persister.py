"""TradePersister — writes a TradeRow per full close so the live `trades`
table is populated and RiskMonitor daily-loss recovery has real data to read."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from sqlmodel import select

from stinger_fx.backtest.replay_broker import SimBroker
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import PositionClosedEvent
from stinger_fx.data import TradePersister, TradeRepo, in_memory_store
from stinger_fx.data.schemas import TradeRow
from stinger_fx.domain import OrderRequest, OrderType, Position, Side


def _closed(
    *,
    ticket: int = 1,
    magic: int = 7,
    symbol: str = "XAUUSD",
    pnl: float = 10.0,
    open_price: float = 2000.0,
    close_price: float | None = 2010.0,
    volume: float = 0.1,
) -> PositionClosedEvent:
    return PositionClosedEvent(
        position=Position(
            ticket=ticket,
            symbol=symbol,
            side=Side.BUY,
            volume=volume,
            open_price=open_price,
            open_time=datetime(2024, 1, 1, 10, tzinfo=UTC),
            magic=magic,
        ),
        realized_pnl=pnl,
        close_price=close_price,
    )


def _rows(store) -> list[TradeRow]:
    with store.session() as s:
        return list(s.exec(select(TradeRow)))


@pytest.mark.asyncio
async def test_writes_one_row_with_mapped_fields() -> None:
    store = in_memory_store()
    p = TradePersister(
        AsyncEventBus(), store, strategy_for_magic=lambda m: {7: "s1"}.get(m)
    )
    await p._on_closed(_closed())
    rows = _rows(store)
    assert len(rows) == 1
    r = rows[0]
    assert r.position_id == 1
    assert r.strategy_id == "s1"
    assert r.symbol == "XAUUSD"
    assert r.side == "buy"
    assert r.pnl == 10.0
    assert r.open_price == 2000.0
    assert r.close_price == 2010.0
    assert r.volume == 0.1


@pytest.mark.asyncio
async def test_unknown_magic_yields_empty_strategy_id() -> None:
    store = in_memory_store()
    p = TradePersister(
        AsyncEventBus(), store, strategy_for_magic=lambda m: {7: "s1"}.get(m)
    )
    await p._on_closed(_closed(magic=999))
    assert _rows(store)[0].strategy_id == ""


@pytest.mark.asyncio
async def test_close_price_falls_back_to_open_price() -> None:
    store = in_memory_store()
    p = TradePersister(AsyncEventBus(), store, strategy_for_magic=lambda m: "s1")
    await p._on_closed(_closed(open_price=2000.0, close_price=None))
    assert _rows(store)[0].close_price == 2000.0


@pytest.mark.asyncio
async def test_db_error_is_swallowed(monkeypatch) -> None:
    store = in_memory_store()
    p = TradePersister(AsyncEventBus(), store, strategy_for_magic=lambda m: "s1")

    def _boom(**_kw):
        raise RuntimeError("db down")

    monkeypatch.setattr(p._repo, "add", _boom)
    # Must not raise — persistence failures can't break the trading path.
    await p._on_closed(_closed())
    assert _rows(store) == []


@pytest.mark.asyncio
async def test_realized_since_round_trip_is_the_fix() -> None:
    """The point of the PR: after persisting today's closes, the same query
    RiskMonitor uses on restart returns the real daily P&L (not zero)."""
    store = in_memory_store()
    p = TradePersister(AsyncEventBus(), store, strategy_for_magic=lambda m: "s1")
    await p._on_closed(_closed(ticket=1, symbol="XAUUSD", pnl=10.0))
    await p._on_closed(_closed(ticket=2, symbol="XAUUSD", pnl=-4.0))
    await p._on_closed(_closed(ticket=3, symbol="EURUSD", pnl=3.0))

    midnight = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    total, by_symbol = TradeRepo(store).realized_since(midnight)
    assert total == pytest.approx(9.0)
    assert by_symbol == {"XAUUSD": pytest.approx(6.0), "EURUSD": pytest.approx(3.0)}


@pytest.mark.asyncio
async def test_end_to_end_simbroker_close_persists_with_close_price() -> None:
    """Full chain: SimBroker close → PositionClosedEvent(close_price) on the bus
    → TradePersister writes a faithful TradeRow."""
    bus = AsyncEventBus()
    store = in_memory_store()
    broker = SimBroker(bus, initial_balance=10_000, symbol_contract_sizes={"XAUUSD": 100.0})
    p = TradePersister(bus, store, strategy_for_magic=lambda m: "s1")
    await p.start()
    try:
        broker.advance_clock(datetime(2024, 1, 1, 10, tzinfo=UTC))
        broker.set_market("XAUUSD", 2000.0)
        result = await broker.place_order(
            OrderRequest(
                strategy_id="s1",
                symbol="XAUUSD",
                side=Side.BUY,
                type=OrderType.MARKET,
                volume=0.1,
                client_order_id="c1",
            )
        )
        broker.set_market("XAUUSD", 2010.0)
        await broker.close_position(result.ticket or 0)
        for _ in range(3):  # let the bus deliver to the subscriber
            await asyncio.sleep(0)

        rows = _rows(store)
        assert len(rows) == 1
        assert rows[0].close_price == pytest.approx(2010.0)
        assert rows[0].pnl == pytest.approx((2010.0 - 2000.0) * 0.1 * 100.0)
    finally:
        await p.stop()
        await bus.close()


@pytest.mark.asyncio
async def test_partial_close_does_not_persist() -> None:
    """Partial closes emit PartialClosedEvent, not PositionClosedEvent — the
    persister (and RiskMonitor's daily counter) only track full closes."""
    bus = AsyncEventBus()
    store = in_memory_store()
    broker = SimBroker(bus, initial_balance=10_000, symbol_contract_sizes={"XAUUSD": 100.0})
    p = TradePersister(bus, store, strategy_for_magic=lambda m: "s1")
    await p.start()
    try:
        broker.advance_clock(datetime(2024, 1, 1, 10, tzinfo=UTC))
        broker.set_market("XAUUSD", 2000.0)
        result = await broker.place_order(
            OrderRequest(
                strategy_id="s1",
                symbol="XAUUSD",
                side=Side.BUY,
                type=OrderType.MARKET,
                volume=0.2,
                client_order_id="c1",
            )
        )
        broker.set_market("XAUUSD", 2010.0)
        await broker.close_position(result.ticket or 0, volume=0.1)  # partial
        for _ in range(3):
            await asyncio.sleep(0)
        assert _rows(store) == []  # no full close yet
    finally:
        await p.stop()
        await bus.close()
