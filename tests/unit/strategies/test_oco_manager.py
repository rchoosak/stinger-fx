"""OCOGroupManager — closing one member cancels its siblings."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import pytest

from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.core.events import ClosePositionRequestEvent
from stinger_fx.domain import Position, Side, Timeframe
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.managers.oco import OCOGroupManager
from stinger_fx.strategies.parameters import StrategyParams


def _make_ctx(bus: AsyncEventBus) -> StrategyContext:
    return StrategyContext(
        strategy_id="oco_test",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        params=StrategyParams(),
        clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)),
        logger=logging.getLogger("test"),
        magic=12345,
        signal_sink=lambda s: asyncio.sleep(0),
        bus=bus,
    )


def _make_position(ticket: int) -> Position:
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


@pytest.mark.asyncio
async def test_oco_closes_sibling_when_member_closes() -> None:
    """Closing one member of an OCO group must publish a close request
    for every other member of the same group."""
    bus = AsyncEventBus()
    close_events: list[ClosePositionRequestEvent] = []

    async def collect(evt: ClosePositionRequestEvent) -> None:
        close_events.append(evt)

    sub = bus.subscribe(ClosePositionRequestEvent, collect, name="t.close")
    ctx = _make_ctx(bus)
    oco = OCOGroupManager(ctx)

    # Two tickets in the same OCO group
    oco.add(101, group_id="bracket_1")
    oco.add(102, group_id="bracket_1")

    try:
        # Ticket 101 closes — sibling 102 should be cancelled
        await oco.on_position_closed(ctx, _make_position(101))
        await _drain(bus)

        assert len(close_events) == 1
        assert close_events[0].ticket == 102
        assert "oco_sibling_closed:101" in close_events[0].reason
        # Group should be cleaned up
        assert oco.groups == {}
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_oco_three_member_group_cancels_two() -> None:
    """A group of three: closing one must cancel the other two."""
    bus = AsyncEventBus()
    close_events: list[ClosePositionRequestEvent] = []

    async def collect(evt: ClosePositionRequestEvent) -> None:
        close_events.append(evt)

    sub = bus.subscribe(ClosePositionRequestEvent, collect, name="t.close")
    ctx = _make_ctx(bus)
    oco = OCOGroupManager(ctx)

    oco.add(1, group_id="g")
    oco.add(2, group_id="g")
    oco.add(3, group_id="g")

    try:
        await oco.on_position_closed(ctx, _make_position(2))
        await _drain(bus)

        closed_tickets = sorted(e.ticket for e in close_events)
        assert closed_tickets == [1, 3]
        assert oco.groups == {}
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_oco_unrelated_close_is_noop() -> None:
    """Closing a position that isn't in any group must not trigger anything."""
    bus = AsyncEventBus()
    close_events: list[ClosePositionRequestEvent] = []

    async def collect(evt: ClosePositionRequestEvent) -> None:
        close_events.append(evt)

    sub = bus.subscribe(ClosePositionRequestEvent, collect, name="t.close")
    ctx = _make_ctx(bus)
    oco = OCOGroupManager(ctx)

    oco.add(1, group_id="g")
    oco.add(2, group_id="g")

    try:
        # Ticket 99 isn't in the group
        await oco.on_position_closed(ctx, _make_position(99))
        await _drain(bus)

        assert close_events == []
        # Group untouched
        assert oco.groups == {"g": {1, 2}}
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_oco_two_groups_independent() -> None:
    """Closing a ticket in group A must not affect group B."""
    bus = AsyncEventBus()
    close_events: list[ClosePositionRequestEvent] = []

    async def collect(evt: ClosePositionRequestEvent) -> None:
        close_events.append(evt)

    sub = bus.subscribe(ClosePositionRequestEvent, collect, name="t.close")
    ctx = _make_ctx(bus)
    oco = OCOGroupManager(ctx)

    oco.add(1, group_id="A")
    oco.add(2, group_id="A")
    oco.add(3, group_id="B")
    oco.add(4, group_id="B")

    try:
        await oco.on_position_closed(ctx, _make_position(1))
        await _drain(bus)

        # Only ticket 2 (group A sibling) should be cancelled
        assert [e.ticket for e in close_events] == [2]
        # Group A gone, group B intact
        assert oco.groups == {"B": {3, 4}}
    finally:
        await sub.unsubscribe()
        await bus.close()


@pytest.mark.asyncio
async def test_oco_no_re_entry_cascade() -> None:
    """When the sibling's close event arrives back at the manager, it must
    NOT trigger another cancellation of the original member (or itself).
    """
    bus = AsyncEventBus()
    close_events: list[ClosePositionRequestEvent] = []

    async def collect(evt: ClosePositionRequestEvent) -> None:
        close_events.append(evt)

    sub = bus.subscribe(ClosePositionRequestEvent, collect, name="t.close")
    ctx = _make_ctx(bus)
    oco = OCOGroupManager(ctx)

    oco.add(1, group_id="g")
    oco.add(2, group_id="g")

    try:
        # First close — triggers cancellation of 2
        await oco.on_position_closed(ctx, _make_position(1))
        await _drain(bus)
        # Simulate the broker firing on_position_closed for ticket 2 after it
        # got cancelled. The manager must treat this as a no-op (group already
        # cleaned up).
        await oco.on_position_closed(ctx, _make_position(2))
        await _drain(bus)

        # Exactly one close request from the original cancellation; no further
        # close events from the cascade-back.
        assert len(close_events) == 1
        assert close_events[0].ticket == 2
    finally:
        await sub.unsubscribe()
        await bus.close()


def test_oco_reject_reassignment() -> None:
    """Adding a ticket to a different group must raise ValueError."""
    bus = AsyncEventBus()
    ctx = _make_ctx(bus)
    oco = OCOGroupManager(ctx)
    oco.add(1, group_id="A")
    with pytest.raises(ValueError, match="already in group"):
        oco.add(1, group_id="B")


def test_oco_remove_manual() -> None:
    """Explicit remove() must drop the ticket from its group."""
    bus = AsyncEventBus()
    ctx = _make_ctx(bus)
    oco = OCOGroupManager(ctx)
    oco.add(1, group_id="A")
    oco.add(2, group_id="A")
    oco.remove(1)
    assert oco.groups == {"A": {2}}
    # Removing the last member empties the group entry
    oco.remove(2)
    assert oco.groups == {}
