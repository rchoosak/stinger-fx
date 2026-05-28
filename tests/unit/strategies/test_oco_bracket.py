"""OCOGroupManager — pending-aware bracket OCO (Phase 6.2.C).

The Phase 5 D position-only tests live in test_oco_manager.py; this file
covers the new bracket scenarios: pending tickets in a group, fill-cascade
on one, and the order-cancellation auto-cleanup path.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
import structlog

from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.core.events import CancelOrderRequestEvent, ClosePositionRequestEvent
from stinger_fx.domain import (
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
    Timeframe,
)
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.managers.oco import OCOGroupManager
from stinger_fx.strategies.parameters import StrategyParams
from tests._helpers import collect_into


def _make_ctx(bus: AsyncEventBus) -> StrategyContext:
    return StrategyContext(
        strategy_id="oco_test",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        params=StrategyParams(),
        clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)),
        logger=structlog.get_logger("test"),
        magic=12345,
        signal_sink=lambda s: asyncio.sleep(0),
        bus=bus,
    )


def _pending_order(ticket: int, side: Side = Side.BUY) -> Order:
    return Order(
        ticket=ticket,
        strategy_id="oco_test",
        symbol="EURUSD",
        side=side,
        type=OrderType.STOP,
        volume=0.1,
        price=1.10,
        status=OrderStatus.SUBMITTED,
        magic=12345,
        client_order_id=f"coid-{ticket}",
    )


def _filled_order(ticket: int, side: Side = Side.BUY) -> Order:
    return Order(
        ticket=ticket,
        strategy_id="oco_test",
        symbol="EURUSD",
        side=side,
        type=OrderType.STOP,
        volume=0.1,
        filled_volume=0.1,
        price=1.10,
        fill_price=1.10,
        status=OrderStatus.FILLED,
        magic=12345,
        client_order_id=f"coid-{ticket}",
        filled_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


def _position(ticket: int) -> Position:
    return Position(
        ticket=ticket,
        symbol="EURUSD",
        side=Side.BUY,
        volume=0.1,
        open_price=1.10,
        open_time=datetime(2024, 1, 1, tzinfo=UTC),
        magic=12345,
    )


async def _drain(bus: AsyncEventBus, *, ticks: int = 3) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


# --- Pending-aware bracket cascades -----------------------------------------


@pytest.mark.asyncio
async def test_pending_fill_cancels_sibling_pending() -> None:
    """Two pending orders in a group; one fills → the other is cancelled
    via a CancelOrderRequestEvent."""
    bus = AsyncEventBus()
    cancels: list[CancelOrderRequestEvent] = []
    closes: list[ClosePositionRequestEvent] = []

    async def cap_cancel(evt: CancelOrderRequestEvent) -> None:
        cancels.append(evt)

    async def cap_close(evt: ClosePositionRequestEvent) -> None:
        closes.append(evt)

    sub_c = bus.subscribe(CancelOrderRequestEvent, cap_cancel, name="t.cancel")
    sub_cl = bus.subscribe(ClosePositionRequestEvent, cap_close, name="t.close")
    ctx = _make_ctx(bus)
    oco = OCOGroupManager(ctx)

    oco.add_bracket(101, 102, group_id="bracket_1")
    assert oco.groups == {"bracket_1": {101: "pending", 102: "pending"}}

    try:
        # Ticket 101 fills → ticket 102 should be cancelled
        await oco.on_order_filled(ctx, _filled_order(101))
        await _drain(bus)

        assert len(cancels) == 1
        assert cancels[0].ticket == 102
        assert "oco_sibling_filled:101" in cancels[0].reason
        assert closes == []  # no close requests (102 was a pending, not a position)
        # Group dissolved
        assert oco.groups == {}
    finally:
        await sub_c.unsubscribe()
        await sub_cl.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_mixed_group_pending_fills_cancels_pending_closes_position() -> None:
    """Group has 1 pending + 1 position; pending fills → close the position,
    nothing to cancel (position uses close, not cancel)."""
    bus = AsyncEventBus()
    cancels: list[CancelOrderRequestEvent] = []
    closes: list[ClosePositionRequestEvent] = []

    async def cap_cancel(evt):
        cancels.append(evt)

    async def cap_close(evt):
        closes.append(evt)

    sub_c = bus.subscribe(CancelOrderRequestEvent, cap_cancel, name="t.cancel")
    sub_cl = bus.subscribe(ClosePositionRequestEvent, cap_close, name="t.close")
    ctx = _make_ctx(bus)
    oco = OCOGroupManager(ctx)

    oco.add(50, group_id="g", kind="pending")
    oco.add(51, group_id="g", kind="position")

    try:
        await oco.on_order_filled(ctx, _filled_order(50))
        await _drain(bus)

        # Sibling 51 is a position → should be closed (not cancelled)
        assert len(closes) == 1
        assert closes[0].ticket == 51
        assert cancels == []
        assert oco.groups == {}
    finally:
        await sub_c.unsubscribe()
        await sub_cl.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_position_close_cancels_pending_sibling() -> None:
    """Group has 1 position + 1 pending; position closes → cancel pending."""
    bus = AsyncEventBus()
    cancels: list[CancelOrderRequestEvent] = []
    closes: list[ClosePositionRequestEvent] = []
    sub_c = bus.subscribe(CancelOrderRequestEvent, collect_into(cancels))
    sub_cl = bus.subscribe(ClosePositionRequestEvent, collect_into(closes))

    ctx = _make_ctx(bus)
    oco = OCOGroupManager(ctx)
    oco.add(60, group_id="g", kind="position")
    oco.add(61, group_id="g", kind="pending")

    try:
        await oco.on_position_closed(ctx, _position(60))
        await _drain(bus)

        # Ticket 61 was pending → cancel, not close
        assert len(cancels) == 1
        assert cancels[0].ticket == 61
        assert closes == []
        assert oco.groups == {}
    finally:
        await sub_c.unsubscribe()
        await sub_cl.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_three_member_bracket_cancels_two_siblings() -> None:
    """3 pendings in a group; one fills → both others cancelled."""
    bus = AsyncEventBus()
    cancels: list[CancelOrderRequestEvent] = []
    sub = bus.subscribe(CancelOrderRequestEvent, collect_into(cancels))

    ctx = _make_ctx(bus)
    oco = OCOGroupManager(ctx)
    oco.add_bracket(1, 2, 3, group_id="g")

    try:
        await oco.on_order_filled(ctx, _filled_order(2))
        await _drain(bus)

        cancelled_tickets = sorted(e.ticket for e in cancels)
        assert cancelled_tickets == [1, 3]
        assert oco.groups == {}
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_external_cancellation_just_cleans_up_no_cascade() -> None:
    """OrderCancelledEvent for a group member should drop it from the group
    without triggering any cascade — the cancel was manual, not a 'win'."""
    bus = AsyncEventBus()
    cancels: list[CancelOrderRequestEvent] = []
    closes: list[ClosePositionRequestEvent] = []
    sub_c = bus.subscribe(CancelOrderRequestEvent, collect_into(cancels))
    sub_cl = bus.subscribe(ClosePositionRequestEvent, collect_into(closes))

    ctx = _make_ctx(bus)
    oco = OCOGroupManager(ctx)
    oco.add_bracket(10, 11, group_id="g")

    try:
        # Simulate ticket 10 being cancelled externally (operator pressed
        # cancel in the UI, or the strategy decided to abort one leg)
        await oco.on_order_cancelled(ctx, _pending_order(10))
        await _drain(bus)

        # No cascade requests fired
        assert cancels == []
        assert closes == []
        # Ticket 10 is gone from the group; ticket 11 still there
        assert oco.groups == {"g": {11: "pending"}}
    finally:
        await sub_c.unsubscribe()
        await sub_cl.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_unrelated_fill_is_ignored() -> None:
    """A fill for a ticket that isn't in any group must not cascade."""
    bus = AsyncEventBus()
    cancels: list[CancelOrderRequestEvent] = []
    sub = bus.subscribe(CancelOrderRequestEvent, collect_into(cancels))

    ctx = _make_ctx(bus)
    oco = OCOGroupManager(ctx)
    oco.add_bracket(100, 101, group_id="g")

    try:
        # Ticket 999 isn't part of any group
        await oco.on_order_filled(ctx, _filled_order(999))
        await _drain(bus)

        assert cancels == []
        # Group untouched
        assert oco.groups == {"g": {100: "pending", 101: "pending"}}
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_phase5_position_only_path_still_works() -> None:
    """Backward-compat: Phase 5 D's position-only group still cascades on close."""
    bus = AsyncEventBus()
    closes: list[ClosePositionRequestEvent] = []
    sub = bus.subscribe(ClosePositionRequestEvent, collect_into(closes))

    ctx = _make_ctx(bus)
    oco = OCOGroupManager(ctx)
    # Default kind="position" — pre-Phase-6.2.C API
    oco.add(200, "g")
    oco.add(201, "g")

    try:
        await oco.on_position_closed(ctx, _position(200))
        await _drain(bus)

        assert len(closes) == 1
        assert closes[0].ticket == 201
    finally:
        await sub.unsubscribe()
        await bus.close()
