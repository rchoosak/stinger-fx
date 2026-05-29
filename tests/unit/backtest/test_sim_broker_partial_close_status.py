"""Regression tests for the SimBroker partial-close status hotfix.

Pre-fix bug
===========

``SimBroker.close_position`` (replay_broker.py) correctly emitted
``PartialClosedEvent`` when ``volume < pos.volume`` but always returned
``OrderResult(ok=True, ticket=ticket, status=OrderStatus.FILLED)``
regardless — see replay_broker.py:510.

PR #59 fixed the same bug on the live side (``MT5Broker.close_position``
returned FILLED on DONE_PARTIAL). After that, MT5 and Sim had divergent
contracts: live close on a partial → PARTIALLY_FILLED, sim close on a
partial → FILLED. Any caller / audit / SQLite mirror / CLI that reads
``result.status`` (rather than subscribing to the bus) couldn't tell
full vs partial in backtests, and code that ran cleanly against the
live contract would behave differently against the sim contract.

Fix
===

Use the same ``full_close`` flag the event-emission branch already
computes to pick ``FILLED`` vs ``PARTIALLY_FILLED`` for the returned
``OrderResult``. Mirrors MT5Broker.close_position.

These tests pin:

  1. Partial close → ``status=PARTIALLY_FILLED``. (THE bug fix.)
  2. Full close → ``status=FILLED`` (regression preserved).
  3. Volume omitted / >= pos.volume → full close, FILLED.
  4. The event side still matches: PartialClosedEvent for partial,
     PositionClosedEvent for full.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from stinger_fx.backtest.replay_broker import SimBroker
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import PartialClosedEvent, PositionClosedEvent
from stinger_fx.domain import OrderRequest, OrderStatus, OrderType, Side
from tests._helpers import collect_into


async def _drain(bus: AsyncEventBus, *, ticks: int = 3) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


def _market_buy_req(*, volume: float = 0.1, coid: str = "m1") -> OrderRequest:
    return OrderRequest(
        strategy_id="s1", symbol="EURUSD", side=Side.BUY,
        type=OrderType.MARKET, volume=volume, client_order_id=coid,
    )


async def _open_position(sb: SimBroker, *, volume: float = 0.2) -> int:
    """Open a BUY position at 1.1001 ask (set in caller's set_market_tick)
    so subsequent partial closes have something to chew through."""
    sb.advance_clock(datetime(2024, 1, 1, tzinfo=UTC))
    result = await sb.place_order(_market_buy_req(volume=volume, coid="open"))
    assert result.ok and result.ticket is not None
    return result.ticket


# --- 1. THE bug fix: partial close → PARTIALLY_FILLED ---------------------


@pytest.mark.asyncio
async def test_close_position_partial_returns_partially_filled_status() -> None:
    """Regression: pre-fix ``OrderResult.status`` was hard-coded to FILLED
    even when ``close_qty < pos.volume``. Sim contract diverged from
    live MT5 contract after PR #59."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.set_market_tick("EURUSD", 1.0999, 1.1001)

    try:
        ticket = await _open_position(sb, volume=0.2)
        # Close 0.05 out of 0.2 → 0.15 remaining
        result = await sb.close_position(ticket, volume=0.05)
        assert result.ok is True
        assert result.status is OrderStatus.PARTIALLY_FILLED, (
            f"partial close must surface as PARTIALLY_FILLED in "
            f"OrderResult.status. Pre-fix this was hard-coded FILLED. "
            f"Got {result.status}."
        )
    finally:
        await bus.close()


# --- 2. Full close still FILLED -------------------------------------------


@pytest.mark.asyncio
async def test_close_position_full_returns_filled_status() -> None:
    """Regression guard: full-close path's status unchanged by the fix."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.set_market_tick("EURUSD", 1.0999, 1.1001)

    try:
        ticket = await _open_position(sb, volume=0.1)
        result = await sb.close_position(ticket)  # close everything
        assert result.ok is True
        assert result.status is OrderStatus.FILLED, (
            f"full close must remain FILLED; got {result.status}"
        )
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_close_position_over_volume_collapses_to_full_close() -> None:
    """When the caller passes volume >= pos.volume, the broker treats it
    as a full close (existing semantic) → status=FILLED, not
    PARTIALLY_FILLED."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.set_market_tick("EURUSD", 1.0999, 1.1001)

    try:
        ticket = await _open_position(sb, volume=0.1)
        # Caller asks to close more than the position — broker clamps
        # to the position volume and emits a full close.
        result = await sb.close_position(ticket, volume=0.2)
        assert result.ok is True
        assert result.status is OrderStatus.FILLED, (
            f"over-volume close collapses to full close → FILLED; "
            f"got {result.status}"
        )
    finally:
        await bus.close()


# --- 3. Event side still matches ------------------------------------------


@pytest.mark.asyncio
async def test_close_position_event_and_status_stay_in_sync_on_partial() -> None:
    """The fix must keep event-side and status-side in agreement: when
    PartialClosedEvent fires, OrderResult.status must be
    PARTIALLY_FILLED (and vice versa for full close). Pre-fix the two
    sides disagreed."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.set_market_tick("EURUSD", 1.0999, 1.1001)

    closed: list[PositionClosedEvent] = []
    partials: list[PartialClosedEvent] = []
    bus.subscribe(PositionClosedEvent, collect_into(closed))
    bus.subscribe(PartialClosedEvent, collect_into(partials))

    try:
        ticket = await _open_position(sb, volume=0.2)
        result = await sb.close_position(ticket, volume=0.07)
        await _drain(bus)

        assert result.status is OrderStatus.PARTIALLY_FILLED
        assert len(partials) == 1, f"expected one PartialClosedEvent, got {partials}"
        assert closed == [], "PositionClosedEvent must NOT fire on partial close"
        assert partials[0].closed_volume == pytest.approx(0.07)
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_close_position_event_and_status_stay_in_sync_on_full() -> None:
    """The full-close path's event/status agreement preserved."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.set_market_tick("EURUSD", 1.0999, 1.1001)

    closed: list[PositionClosedEvent] = []
    partials: list[PartialClosedEvent] = []
    bus.subscribe(PositionClosedEvent, collect_into(closed))
    bus.subscribe(PartialClosedEvent, collect_into(partials))

    try:
        ticket = await _open_position(sb, volume=0.1)
        result = await sb.close_position(ticket)
        await _drain(bus)

        assert result.status is OrderStatus.FILLED
        assert len(closed) == 1, f"expected one PositionClosedEvent, got {closed}"
        assert partials == [], "PartialClosedEvent must NOT fire on full close"
    finally:
        await bus.close()
