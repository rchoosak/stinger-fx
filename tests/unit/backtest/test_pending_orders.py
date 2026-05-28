"""SimBroker pending orders (Phase 6.2.A).

Verifies all four pending types fire at the right price, with the right
fill semantics; that cancel_order works; and that an un-triggered pending
just sits there.
"""

from __future__ import annotations

import asyncio

import pytest

from stinger_fx.backtest.replay_broker import SimBroker
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import (
    OrderCancelledEvent,
    OrderFilledEvent,
    OrderSubmittedEvent,
)
from stinger_fx.domain import OrderRequest, OrderStatus, OrderType, Side


def _req(
    *,
    side: Side,
    type_: OrderType,
    price: float,
    volume: float = 0.1,
    coid: str = "coid-1",
) -> OrderRequest:
    return OrderRequest(
        strategy_id="s1",
        symbol="EURUSD",
        side=side,
        type=type_,
        volume=volume,
        price=price,
        client_order_id=coid,
    )


async def _drain(bus: AsyncEventBus, *, ticks: int = 3) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_place_pending_returns_submitted_not_filled() -> None:
    """A pending order must come back as SUBMITTED with no position created."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.set_market_tick("EURUSD", 1.1000, 1.1002)

    submitted: list[OrderSubmittedEvent] = []
    filled: list[OrderFilledEvent] = []
    sub_s = bus.subscribe(OrderSubmittedEvent, lambda e: submitted.append(e) or asyncio.sleep(0))
    sub_f = bus.subscribe(OrderFilledEvent, lambda e: filled.append(e) or asyncio.sleep(0))

    try:
        result = await sb.place_order(
            _req(side=Side.BUY, type_=OrderType.STOP, price=1.1020)
        )
        await _drain(bus)
        assert result.ok is True
        assert result.status == OrderStatus.SUBMITTED
        assert len(submitted) == 1
        assert filled == []
        assert (await sb.get_positions()) == []
        opens = await sb.get_open_orders()
        assert len(opens) == 1
        assert opens[0].type == OrderType.STOP
    finally:
        await sub_s.unsubscribe()
        await sub_f.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_buy_stop_triggers_when_ask_crosses_above() -> None:
    """BUY_STOP at 1.10 must fire when ask reaches 1.10."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.set_market_tick("EURUSD", 1.0998, 1.0999)  # below trigger

    filled: list[OrderFilledEvent] = []
    sub = bus.subscribe(OrderFilledEvent, lambda e: filled.append(e) or asyncio.sleep(0))

    try:
        await sb.place_order(_req(side=Side.BUY, type_=OrderType.STOP, price=1.10))
        # Tick below trigger — no fire
        triggered = await sb.check_pending("EURUSD", 1.0998, 1.0999)
        assert triggered == []
        # Tick reaches trigger
        triggered = await sb.check_pending("EURUSD", 1.0999, 1.10)
        await _drain(bus)
        assert len(triggered) == 1
        assert len(filled) == 1
        # Position created
        positions = await sb.get_positions()
        assert len(positions) == 1
        assert positions[0].side == Side.BUY
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_sell_stop_triggers_when_bid_falls() -> None:
    """SELL_STOP at 1.10 must fire when bid drops to 1.10."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.set_market_tick("EURUSD", 1.1005, 1.1007)

    try:
        await sb.place_order(_req(side=Side.SELL, type_=OrderType.STOP, price=1.10))
        triggered = await sb.check_pending("EURUSD", 1.1001, 1.1003)
        assert triggered == []
        triggered = await sb.check_pending("EURUSD", 1.0999, 1.10)
        assert len(triggered) == 1
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_buy_limit_triggers_when_ask_falls() -> None:
    """BUY_LIMIT at 1.10 must fire when ask drops to 1.10, fill at limit."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.set_market_tick("EURUSD", 1.1010, 1.1012)

    try:
        await sb.place_order(_req(side=Side.BUY, type_=OrderType.LIMIT, price=1.10))
        triggered = await sb.check_pending("EURUSD", 1.1009, 1.1011)
        assert triggered == []
        triggered = await sb.check_pending("EURUSD", 1.0999, 1.10)
        assert len(triggered) == 1
        # LIMIT fills at the limit price (1.10), not the market price
        assert triggered[0].fill_price == pytest.approx(1.10)
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_sell_limit_triggers_when_bid_rises() -> None:
    """SELL_LIMIT at 1.10 fires when bid rises to 1.10."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.set_market_tick("EURUSD", 1.0990, 1.0992)

    try:
        await sb.place_order(_req(side=Side.SELL, type_=OrderType.LIMIT, price=1.10))
        triggered = await sb.check_pending("EURUSD", 1.0995, 1.0997)
        assert triggered == []
        triggered = await sb.check_pending("EURUSD", 1.10, 1.1002)
        assert len(triggered) == 1
        # LIMIT fills at the limit price (1.10), not the market price
        assert triggered[0].fill_price == pytest.approx(1.10)
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_cancel_pending_removes_from_queue() -> None:
    """cancel_order on a pending ticket emits OrderCancelledEvent + removes it."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.set_market_tick("EURUSD", 1.10, 1.1002)

    cancelled: list[OrderCancelledEvent] = []
    sub = bus.subscribe(OrderCancelledEvent, lambda e: cancelled.append(e) or asyncio.sleep(0))

    try:
        result = await sb.place_order(_req(side=Side.BUY, type_=OrderType.STOP, price=1.20))
        ticket = result.ticket
        assert ticket is not None
        # Cancel
        cancel_result = await sb.cancel_order(ticket)
        await _drain(bus)
        assert cancel_result.ok is True
        assert cancel_result.status == OrderStatus.CANCELLED
        assert len(cancelled) == 1
        # Pending queue is now empty
        assert (await sb.get_open_orders()) == []
        # Trigger no longer fires (gone from queue)
        triggered = await sb.check_pending("EURUSD", 1.20, 1.2005)
        assert triggered == []
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_pending_without_price_rejected() -> None:
    """Pending order with price=None must be rejected."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.set_market_tick("EURUSD", 1.10, 1.1002)

    try:
        req = OrderRequest(
            strategy_id="s1",
            symbol="EURUSD",
            side=Side.BUY,
            type=OrderType.STOP,
            volume=0.1,
            price=None,  # missing!
            client_order_id="x",
        )
        result = await sb.place_order(req)
        assert result.ok is False
        assert "requires a price" in result.message
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_pending_wrong_symbol_ignored() -> None:
    """check_pending for symbol A must not trigger an order on symbol B."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.set_market_tick("EURUSD", 1.10, 1.1002)
    sb.set_market_tick("GBPUSD", 1.27, 1.2702)

    try:
        await sb.place_order(_req(side=Side.BUY, type_=OrderType.STOP, price=1.10))
        # GBPUSD tick crossing 1.10 must NOT fire the EURUSD pending
        triggered = await sb.check_pending("GBPUSD", 1.10, 1.1002)
        assert triggered == []
        # EURUSD tick still triggers normally
        triggered = await sb.check_pending("EURUSD", 1.10, 1.1002)
        assert len(triggered) == 1
    finally:
        await bus.close()
