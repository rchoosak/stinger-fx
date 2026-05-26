"""OCOGroupManager — one-cancels-other position groups.

When a strategy opens two or more positions that should "compete" — i.e.
exiting one means cancelling the others — wire them into an OCO group
through this manager.  As soon as any member of the group closes
(SL fires, TP fires, strategy calls ``ctx.close()``, or another manager
closes it), every other member of the same group is closed automatically.

This is a Python-side, position-level OCO. It does **not** require pending
orders or broker-side OCO support — it works on already-open positions.

Typical use case: a bracket trade where you've opened both a "take profit"
and a "stop loss" position (perhaps with different volumes or stops) and
want the surviving one to vanish once the other has paid out.

Usage::

    async def on_start(self, ctx):
        self._oco = OCOGroupManager(ctx)
        ctx.attach_manager(self._oco)

    async def on_bar(self, ctx, bar):
        # Open two positions that should compete
        await ctx.buy(0.1, sl=1.10, tp=1.12)
        ticket_a = next(p.ticket for p in ctx.position.all() if ...)
        await ctx.sell(0.1, sl=1.13, tp=1.11)
        ticket_b = next(...)
        self._oco.add(ticket_a, group_id="bracket_1")
        self._oco.add(ticket_b, group_id="bracket_1")

The manager exposes:

* ``add(ticket, group_id)`` — register a ticket as a member of an OCO group
* ``remove(ticket)`` — explicit removal (rare; auto-cleanup happens on close)
* ``groups`` — read-only view of current groups (dict[group_id, set[ticket]])
"""

from __future__ import annotations

import logging

from stinger_fx.domain import Position, Tick
from stinger_fx.strategies.context import StrategyContext

logger = logging.getLogger("stinger.strategies.managers.oco")


class OCOGroupManager:
    """Cancel siblings when any member of an OCO group closes."""

    def __init__(self, ctx: StrategyContext) -> None:
        self._ctx = ctx
        # group_id -> set of tickets in that group
        self._groups: dict[str, set[int]] = {}
        # ticket -> group_id (reverse lookup for O(1) on close)
        self._ticket_to_group: dict[int, str] = {}
        # Tickets currently being cancelled — avoids re-entrant chain when
        # the cascade closes trigger their own on_position_closed dispatch.
        self._cancelling: set[int] = set()

    # --- Public API ---------------------------------------------------------

    def add(self, ticket: int, group_id: str) -> None:
        """Register ``ticket`` as a member of OCO group ``group_id``.

        If the ticket already belongs to a different group, raise — silent
        re-assignment would be a footgun.
        """
        existing = self._ticket_to_group.get(ticket)
        if existing is not None and existing != group_id:
            raise ValueError(
                f"ticket {ticket} is already in group {existing!r}; "
                f"can't reassign to {group_id!r}"
            )
        self._groups.setdefault(group_id, set()).add(ticket)
        self._ticket_to_group[ticket] = group_id
        logger.debug("oco_add ticket=%s group=%s", ticket, group_id)

    def remove(self, ticket: int) -> None:
        """Drop a ticket from its OCO group (manual cleanup, rare)."""
        group_id = self._ticket_to_group.pop(ticket, None)
        if group_id is None:
            return
        members = self._groups.get(group_id)
        if members is not None:
            members.discard(ticket)
            if not members:
                self._groups.pop(group_id, None)

    @property
    def groups(self) -> dict[str, set[int]]:
        """Read-only snapshot of current groups → ticket membership."""
        return {gid: set(tickets) for gid, tickets in self._groups.items()}

    # --- Hooks --------------------------------------------------------------

    async def on_tick(self, ctx: StrategyContext, tick: Tick) -> None:
        """OCO is reactive (close-driven) — no per-tick work needed."""
        return None

    async def on_position_closed(self, ctx: StrategyContext, position: Position) -> None:
        """When a member of an OCO group closes, cancel every sibling."""
        group_id = self._ticket_to_group.get(position.ticket)
        if group_id is None:
            return

        # Pull siblings (all members of the group except the one that just closed).
        members = self._groups.get(group_id, set())
        siblings = members - {position.ticket} - self._cancelling

        # Mark them as "being cancelled" so their own on_position_closed
        # dispatch doesn't re-trigger a cascade in the wrong direction.
        self._cancelling.update(siblings)

        logger.info(
            "oco_trigger closed_ticket=%s group=%s siblings=%s",
            position.ticket, group_id, sorted(siblings),
        )

        try:
            for sib_ticket in sorted(siblings):
                await ctx.close(sib_ticket, reason=f"oco_sibling_closed:{position.ticket}")
        finally:
            # Whatever happens, clean up bookkeeping for the whole group.
            for t in members:
                self._ticket_to_group.pop(t, None)
                self._cancelling.discard(t)
            self._groups.pop(group_id, None)
