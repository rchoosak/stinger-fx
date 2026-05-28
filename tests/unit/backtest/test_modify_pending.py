"""Pending order modification via SimBroker + OrderRouter (Phase 6.2.D).

Covers:
  * SimBroker.modify_order applied to a pending order updates the right
    fields (price / volume / sl / tp) and leaves un-supplied fields alone
  * After a price update, check_pending uses the NEW price
  * After a volume update, the eventual fill uses the NEW volume
  * Router routes a ModifyOrderRequestEvent for a pending ticket through
    the broker correctly (with magic ownership check)
"""

from __future__ import annotations

import asyncio

import pytest

from stinger_fx.backtest.order_router import OrderRouter
from stinger_fx.backtest.replay_broker import SimBroker
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import ModifyOrderRequestEvent
from stinger_fx.domain import OrderRequest, OrderType, Side


def _pending_req(
    *, price: float = 1.10, volume: float = 0.1, sl: float | None = None,
    side: Side = Side.BUY, coid: str = "coid-1",
) -> OrderRequest:
    return OrderRequest(
        strategy_id="s1",
        symbol="EURUSD",
        side=side,
        type=OrderType.STOP,
        volume=volume,
        price=price,
        sl=sl,
        client_order_id=coid,
        magic=1,
    )


async def _drain(bus: AsyncEventBus, *, ticks: int = 3) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


# --- SimBroker.modify_order on pending --------------------------------------


@pytest.mark.asyncio
async def test_modify_pending_updates_price_only() -> None:
    """Update price; volume/sl/tp untouched."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.set_market_tick("EURUSD", 1.0998, 1.0999)

    try:
        result = await sb.place_order(_pending_req(price=1.10, volume=0.1, sl=1.095))
        ticket = result.ticket
        assert ticket is not None
        # Modify only price
        mod_result = await sb.modify_order(ticket, price=1.1050)
        assert mod_result.ok is True
        pending_after = (await sb.get_open_orders())[0]
        assert pending_after.price == pytest.approx(1.1050)
        assert pending_after.volume == pytest.approx(0.1)
        assert pending_after.sl == pytest.approx(1.095)
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_modify_pending_updates_volume_only() -> None:
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.set_market_tick("EURUSD", 1.0998, 1.0999)
    try:
        result = await sb.place_order(_pending_req(price=1.10, volume=0.1))
        assert result.ticket is not None
        await sb.modify_order(result.ticket, volume=0.25)
        pending = (await sb.get_open_orders())[0]
        assert pending.volume == pytest.approx(0.25)
        assert pending.price == pytest.approx(1.10)
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_modify_pending_rejects_negative_volume() -> None:
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.set_market_tick("EURUSD", 1.0998, 1.0999)
    try:
        result = await sb.place_order(_pending_req(price=1.10, volume=0.1))
        assert result.ticket is not None
        mod = await sb.modify_order(result.ticket, volume=-0.5)
        assert mod.ok is False
        assert "volume" in mod.message.lower()
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_modified_pending_triggers_at_new_price() -> None:
    """After moving the trigger price, check_pending fires at the NEW price."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.set_market_tick("EURUSD", 1.0998, 1.0999)
    try:
        result = await sb.place_order(_pending_req(price=1.10))
        assert result.ticket is not None
        # Move trigger up to 1.105
        await sb.modify_order(result.ticket, price=1.105)
        # Tick crosses original (1.10) — should NOT fire
        triggered = await sb.check_pending("EURUSD", 1.10, 1.1001)
        assert triggered == []
        # Tick crosses NEW price (1.105) — fires
        triggered = await sb.check_pending("EURUSD", 1.1049, 1.105)
        assert len(triggered) == 1
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_modified_pending_fills_with_new_volume() -> None:
    """After volume update, the resulting position uses the new volume."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.set_market_tick("EURUSD", 1.0998, 1.0999)
    try:
        result = await sb.place_order(_pending_req(price=1.10, volume=0.1))
        assert result.ticket is not None
        await sb.modify_order(result.ticket, volume=0.3)
        await sb.check_pending("EURUSD", 1.10, 1.1001)
        positions = await sb.get_positions()
        assert len(positions) == 1
        assert positions[0].volume == pytest.approx(0.3)
    finally:
        await bus.close()


# --- Router → SimBroker for pending modification -----------------------------


@pytest.mark.asyncio
async def test_router_modifies_pending_via_event() -> None:
    """A strategy publishes ModifyOrderRequestEvent → router routes to
    broker.modify_order and the pending order's price updates."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.set_market_tick("EURUSD", 1.0998, 1.0999)
    router = OrderRouter(bus, sb, strategy_magic={"s1": 1})
    await router.attach()
    try:
        result = await sb.place_order(_pending_req(price=1.10))
        ticket = result.ticket
        await bus.publish(ModifyOrderRequestEvent(
            strategy_id="s1",
            ticket=ticket,
            price=1.1050,
            reason="trail",
        ))
        await _drain(bus, ticks=5)
        pending = (await sb.get_open_orders())[0]
        assert pending.price == pytest.approx(1.1050)
    finally:
        await router.detach()
        await bus.close()


@pytest.mark.asyncio
async def test_router_refuses_cross_strategy_pending_modify() -> None:
    """A strategy can't modify a pending owned by another strategy (magic check)."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.set_market_tick("EURUSD", 1.0998, 1.0999)
    # s1 owns magic=1; we'll try to modify from s2 with magic=999
    router = OrderRouter(bus, sb, strategy_magic={"s1": 1, "s2": 999})
    await router.attach()
    try:
        # Place pending as s1 (magic=1)
        result = await sb.place_order(_pending_req(price=1.10))
        ticket = result.ticket
        # s2 tries to modify — should be refused
        await bus.publish(ModifyOrderRequestEvent(
            strategy_id="s2",
            ticket=ticket,
            price=1.20,  # would never want this
            reason="hijack",
        ))
        await _drain(bus, ticks=5)
        pending = (await sb.get_open_orders())[0]
        # Unchanged
        assert pending.price == pytest.approx(1.10)
    finally:
        await router.detach()
        await bus.close()
