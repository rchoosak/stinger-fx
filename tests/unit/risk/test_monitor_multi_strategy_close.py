"""Regression tests for the RiskMonitor multi-strategy decrement hotfix.

Pre-fix bug
===========

On ``PositionClosedEvent``, ``RiskMonitor._on_closed`` decremented the
**first non-zero** per-strategy bucket it found, regardless of which
strategy actually owns the closed position::

    for sid in list(self._open_positions.keys()):
        if self._open_positions[sid] > 0:
            self._open_positions[sid] -= 1
            break

With two or more strategies running, that "first non-zero" was just
dict insertion order. If strategy ``A`` opens a position and then
strategy ``B`` closes one, the monitor would decrement ``A``'s bucket
even though ``A`` is still actually holding its position. Net effect:

  * ``A`` reports fewer open positions than reality → allowed to open
    *more* than its cap (silent over-trading).
  * ``B`` reports more open positions than reality → could be wrongly
    blocked from new signals (silent under-trading).

This is exactly the kind of bug that's invisible in a single-strategy
test (where there's only one bucket so "first" always matches) and only
surfaces in multi-strategy production.

Fix
===

Track ``ticket → strategy_id`` from ``OrderFilledEvent`` (which carries
both fields). On ``PositionClosedEvent``, look up by ticket to decrement
the correct strategy's bucket and pop the mapping. Unknown ticket
(e.g. position opened before RiskMonitor started) logs a warning and
skips the per-strategy decrement rather than corrupting an innocent
bucket.

These tests pin:

  1. Two strategies; close ``A``'s ticket; only ``A``'s bucket
     decrements, ``B``'s is unchanged. (The headline regression.)
  2. The dict-insertion-order angle: even if ``A`` filled first and is
     therefore "first non-zero" in the dict, closing ``B``'s ticket
     still decrements ``B`` — proving the lookup is by ticket and not
     by iteration order.
  3. Unknown ticket falls back to skip + warn, never corrupts another
     strategy's count.
  4. After close, the ticket→strategy map is cleaned up (no unbounded
     growth across an engine lifetime).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from stinger_fx.config.models import RiskConfig
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import OrderFilledEvent, PositionClosedEvent
from stinger_fx.domain import (
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
)
from stinger_fx.risk import RiskMonitor


def _filled(strategy_id: str, ticket: int, symbol: str = "EURUSD") -> OrderFilledEvent:
    return OrderFilledEvent(
        order=Order(
            ticket=ticket,
            strategy_id=strategy_id,
            symbol=symbol,
            side=Side.BUY,
            type=OrderType.MARKET,
            volume=0.1,
            status=OrderStatus.FILLED,
        )
    )


def _closed(ticket: int, *, symbol: str = "EURUSD", magic: int = 0, pnl: float = 0.0) -> PositionClosedEvent:
    return PositionClosedEvent(
        position=Position(
            ticket=ticket,
            symbol=symbol,
            side=Side.BUY,
            volume=0.1,
            open_price=1.10,
            open_time=datetime.now(UTC),
            magic=magic,
        ),
        realized_pnl=pnl,
    )


# --- 1. The headline regression -------------------------------------------


@pytest.mark.asyncio
async def test_close_decrements_owning_strategy_not_first_bucket() -> None:
    """Two strategies, each with one open position. Closing strategy
    ``A``'s ticket must decrement ``A``'s bucket only — pre-fix this
    decremented whichever bucket appeared first in ``_open_positions``,
    silently moving the count between strategies."""
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, RiskConfig(max_open_positions_per_strategy=1))
    await rm.start()
    try:
        # s1 fills ticket 100, s2 fills ticket 200.
        await rm._on_filled(_filled("s1", ticket=100))
        await rm._on_filled(_filled("s2", ticket=200))
        snap_before = rm.snapshot()
        assert snap_before["open_positions"] == {"s1": 1, "s2": 1}

        # Close s1's ticket. s1 must go to 0, s2 must STAY at 1.
        await rm._on_closed(_closed(ticket=100))
        snap = rm.snapshot()
        assert snap["open_positions"] == {"s1": 0, "s2": 1}, (
            f"close of s1's ticket=100 should decrement s1 alone, "
            f"got {snap['open_positions']}"
        )

        # And the cap semantics confirm it: s1 may open again, s2 may not.
        from stinger_fx.domain import Signal, SignalStrength
        s1_sig = Signal(
            strategy_id="s1", time=datetime.now(UTC), symbol="EURUSD",
            side=Side.BUY, strength=SignalStrength.NORMAL, suggested_volume=0.1,
        )
        s2_sig = Signal(
            strategy_id="s2", time=datetime.now(UTC), symbol="EURUSD",
            side=Side.BUY, strength=SignalStrength.NORMAL, suggested_volume=0.1,
        )
        assert rm.check_signal(s1_sig).allowed is True, "s1 closed, should be allowed"
        assert rm.check_signal(s2_sig).allowed is False, "s2 still at cap, should be blocked"
    finally:
        await rm.stop()
        await bus.close()


# --- 2. Dict-iteration-order angle ----------------------------------------


@pytest.mark.asyncio
async def test_close_b_does_not_decrement_a_just_because_a_filled_first() -> None:
    """Pre-fix bug was specifically "first non-zero bucket". Even when
    ``s1`` fills first (so it appears first in dict iteration order),
    closing ``s2``'s ticket must decrement ``s2`` — not ``s1``."""
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, RiskConfig(max_open_positions_per_strategy=3))
    await rm.start()
    try:
        # Establish insertion order: s1 first, then s2.
        await rm._on_filled(_filled("s1", ticket=100))
        await rm._on_filled(_filled("s2", ticket=200))
        await rm._on_filled(_filled("s2", ticket=201))  # s2 now has 2 open

        assert rm.snapshot()["open_positions"] == {"s1": 1, "s2": 2}

        # Close s2's ticket 200. s1 must remain at 1, s2 must drop to 1.
        await rm._on_closed(_closed(ticket=200))
        snap = rm.snapshot()
        assert snap["open_positions"] == {"s1": 1, "s2": 1}, (
            f"closing s2's ticket should not touch s1 — got {snap['open_positions']}. "
            f"This is the exact pre-fix bug: 'first non-zero bucket' is s1."
        )
    finally:
        await rm.stop()
        await bus.close()


# --- 3. Unknown ticket falls back gracefully ------------------------------


@pytest.mark.asyncio
async def test_unknown_ticket_close_does_not_corrupt_other_strategies(caplog) -> None:
    """A close for a ticket we never saw filled (e.g. position opened
    before RiskMonitor started). The fix must NOT decrement someone
    else's bucket — that would be the same kind of cross-strategy
    corruption the headline test guards against."""
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, RiskConfig(max_open_positions_per_strategy=2))
    await rm.start()
    try:
        # s1 has a known open position.
        await rm._on_filled(_filled("s1", ticket=100))
        before = dict(rm.snapshot()["open_positions"])  # type: ignore[arg-type]
        assert before == {"s1": 1}

        # A close fires for an unknown ticket.
        with caplog.at_level(logging.WARNING, logger="stinger.risk"):
            await rm._on_closed(_closed(ticket=9999))

        after = dict(rm.snapshot()["open_positions"])  # type: ignore[arg-type]
        assert after == {"s1": 1}, (
            f"unknown-ticket close must not decrement any strategy bucket "
            f"— got {after}"
        )
        # And the operator should know about it.
        assert any(
            "risk_close_ticket_unknown" in rec.message for rec in caplog.records
        ), "expected a warning log about the unknown ticket"
    finally:
        await rm.stop()
        await bus.close()


# --- 4. Ticket map cleanup ------------------------------------------------


@pytest.mark.asyncio
async def test_ticket_map_drains_on_close() -> None:
    """The ticket→strategy mapping must shrink as positions close —
    otherwise it grows unbounded over the engine's lifetime."""
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, RiskConfig())
    await rm.start()
    try:
        await rm._on_filled(_filled("s1", ticket=100))
        await rm._on_filled(_filled("s2", ticket=200))
        assert rm._ticket_to_strategy == {100: "s1", 200: "s2"}

        await rm._on_closed(_closed(ticket=100))
        assert rm._ticket_to_strategy == {200: "s2"}, (
            f"ticket 100 should have been popped — got {rm._ticket_to_strategy}"
        )

        await rm._on_closed(_closed(ticket=200))
        assert rm._ticket_to_strategy == {}, (
            f"all tickets should be drained — got {rm._ticket_to_strategy}"
        )
    finally:
        await rm.stop()
        await bus.close()


# --- 5. Symbol-level decrement still works --------------------------------


@pytest.mark.asyncio
async def test_per_symbol_counter_decrements_correctly() -> None:
    """Per-symbol counter uses ``evt.position.symbol`` (already correct
    pre-fix), but make sure the fix didn't accidentally break it."""
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, RiskConfig())
    await rm.start()
    try:
        await rm._on_filled(_filled("s1", ticket=100, symbol="EURUSD"))
        await rm._on_filled(_filled("s2", ticket=200, symbol="GBPUSD"))
        assert rm.snapshot()["open_positions_by_symbol"] == {"EURUSD": 1, "GBPUSD": 1}

        await rm._on_closed(_closed(ticket=100, symbol="EURUSD"))
        assert rm.snapshot()["open_positions_by_symbol"] == {"EURUSD": 0, "GBPUSD": 1}
    finally:
        await rm.stop()
        await bus.close()
