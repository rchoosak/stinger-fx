"""OCOGroupManager — one-cancels-other position and pending-order groups.

Two complementary use cases are supported:

  1. **Position-only OCO** (Phase 5 D, the original): two or more open
     positions in the same group; when one closes, close the rest.
     Example: a hedged long/short pair where exiting one half means
     exiting the other.

  2. **Pending-aware OCO bracket** (Phase 6.2.C): two or more pending
     orders in the same group; when one fills, cancel the rest. After
     the fill, the surviving position is solo (out of the group).
     Example: BUY_STOP above resistance + SELL_STOP below support —
     a breakout in either direction triggers one leg and cancels
     the other.

The manager is reactive to three event types (each routed via the runner):

  * ``on_order_filled``     — a pending member triggered → cancel siblings
  * ``on_position_closed``  — a position member closed → close siblings
  * ``on_order_cancelled``  — a pending member cancelled externally →
                              just remove from group (no cascade)

Each ticket in a group has a ``kind`` of ``"pending"`` or ``"position"``.
The kind transitions ``pending → position`` automatically on
``on_order_filled``. The cascade picks the right primitive for each
sibling based on its kind: ``ctx.cancel_order()`` for pendings,
``ctx.close()`` for positions.

The ``_dissolving`` set prevents re-entrant cascades: when the manager
cancels/closes siblings, their own ``on_order_cancelled`` /
``on_position_closed`` events arrive back at the manager and would
otherwise re-trigger the cascade.
"""

from __future__ import annotations

import logging
from typing import Literal

from stinger_fx.domain import Order, Position, Tick
from stinger_fx.strategies.context import StrategyContext

logger = logging.getLogger("stinger.strategies.managers.oco")

Kind = Literal["pending", "position"]


class OCOGroupManager:
    """Cancel/close siblings when any member of an OCO group transitions out."""

    def __init__(self, ctx: StrategyContext) -> None:
        self._ctx = ctx
        # group_id -> {ticket: kind}
        self._groups: dict[str, dict[int, Kind]] = {}
        # ticket -> group_id (O(1) reverse lookup)
        self._ticket_to_group: dict[int, str] = {}
        # Tickets currently in flight via the cascade — when their close /
        # cancel event arrives back at the manager, skip the cascade so we
        # don't trigger it twice for the same group dissolution.
        self._dissolving: set[int] = set()

    # --- Public API ---------------------------------------------------------

    def add(
        self,
        ticket: int,
        group_id: str,
        *,
        kind: Kind = "position",
    ) -> None:
        """Register ``ticket`` as a member of OCO group ``group_id``.

        ``kind`` should be ``"pending"`` for orders awaiting trigger,
        ``"position"`` for already-open positions. Default ``"position"``
        keeps Phase 5 D's API unchanged.

        Re-adding a ticket to its existing group is idempotent; assigning
        a ticket to a *different* group raises.
        """
        if kind not in ("pending", "position"):
            raise ValueError(f"kind must be 'pending' or 'position', got {kind!r}")
        existing = self._ticket_to_group.get(ticket)
        if existing is not None and existing != group_id:
            raise ValueError(
                f"ticket {ticket} is already in group {existing!r}; "
                f"can't reassign to {group_id!r}"
            )
        self._groups.setdefault(group_id, {})[ticket] = kind
        self._ticket_to_group[ticket] = group_id
        logger.debug("oco_add ticket=%s group=%s kind=%s", ticket, group_id, kind)

    def add_bracket(self, *tickets: int, group_id: str) -> None:
        """Convenience for the typical breakout pattern: add multiple pending
        tickets to the same OCO group at once.

        Example::

            buy_result = await broker.place_order(buy_stop_req)
            sell_result = await broker.place_order(sell_stop_req)
            self._oco.add_bracket(
                buy_result.ticket, sell_result.ticket, group_id="breakout_1"
            )
        """
        for t in tickets:
            self.add(t, group_id, kind="pending")

    def remove(self, ticket: int) -> None:
        """Drop a ticket from its group (manual cleanup, rare)."""
        group_id = self._ticket_to_group.pop(ticket, None)
        if group_id is None:
            return
        members = self._groups.get(group_id)
        if members is not None:
            members.pop(ticket, None)
            if not members:
                self._groups.pop(group_id, None)

    @property
    def groups(self) -> dict[str, dict[int, Kind]]:
        """Read-only snapshot — useful for tests / debugging."""
        return {gid: dict(tickets) for gid, tickets in self._groups.items()}

    # --- Hooks --------------------------------------------------------------

    async def on_tick(self, ctx: StrategyContext, tick: Tick) -> None:
        """OCO is event-driven; nothing to do per-tick."""
        return None

    async def on_order_filled(self, ctx: StrategyContext, order: Order) -> None:
        """A pending in the group just triggered → cancel sibling pendings,
        close sibling positions. The triggered ticket transitions to
        ``"position"`` kind and remains in the group only briefly while we
        cascade; afterwards the group is dissolved entirely.
        """
        group_id = self._ticket_to_group.get(order.ticket)
        if group_id is None:
            return
        # Bump kind so any concurrent _cascade pass sees the new state.
        members = self._groups.get(group_id, {})
        if order.ticket in members:
            members[order.ticket] = "position"
        await self._cascade(order.ticket, reason_prefix="oco_sibling_filled")

    async def on_position_closed(
        self, ctx: StrategyContext, position: Position
    ) -> None:
        """A position in the group just closed → cancel/close siblings."""
        if position.ticket not in self._ticket_to_group:
            return
        await self._cascade(position.ticket, reason_prefix="oco_sibling_closed")

    async def on_order_cancelled(
        self, ctx: StrategyContext, order: Order
    ) -> None:
        """An external cancel landed on one of our tickets — just clean up
        bookkeeping; cancellation is not a "win" so we don't cascade."""
        self.remove(order.ticket)

    # --- Cascade engine -----------------------------------------------------

    async def _cascade(self, trigger_ticket: int, *, reason_prefix: str) -> None:
        if trigger_ticket in self._dissolving:
            return  # already being handled — avoid re-entry on bounced events
        group_id = self._ticket_to_group.get(trigger_ticket)
        if group_id is None:
            return
        members = self._groups.get(group_id, {})
        # Snapshot siblings before we start dispatching so concurrent edits
        # don't change the set under our feet.
        sibling_items = [
            (t, k) for t, k in members.items() if t != trigger_ticket and t not in self._dissolving
        ]
        for t, _ in sibling_items:
            self._dissolving.add(t)
        logger.info(
            "oco_trigger ticket=%s group=%s siblings=%s reason=%s",
            trigger_ticket, group_id,
            sorted(t for t, _ in sibling_items),
            reason_prefix,
        )
        try:
            reason = f"{reason_prefix}:{trigger_ticket}"
            for sib_ticket, kind in sorted(sibling_items):
                if kind == "pending":
                    await self._ctx.cancel_order(sib_ticket, reason=reason)
                else:
                    await self._ctx.close(sib_ticket, reason=reason)
        finally:
            # Whatever happened, the group is over. Clean up all members
            # (including the trigger and any siblings) so subsequent
            # bounced events fall through quietly.
            for t in list(members.keys()):
                self._ticket_to_group.pop(t, None)
                self._dissolving.discard(t)
            self._groups.pop(group_id, None)
