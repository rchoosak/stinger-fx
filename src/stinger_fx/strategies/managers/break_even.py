"""BreakEvenMover — pushes SL to (or just past) the open price once enough
profit has accrued.

Common usage: a 20-pip trade with an SL 10 pips below entry. When +15 pips
runs in your favour, you'd like to lock in at least flat — set
`trigger_pips=15, lock_pips=1` to ratchet SL to open_price + 1 pip on the
first tick that crosses the trigger.

The mover is idempotent — once it has fired for a ticket it ignores further
ticks for that position. Closing and reopening (new ticket) resets state.
"""

from __future__ import annotations

import logging

from stinger_fx.domain import Side, Tick
from stinger_fx.strategies.context import StrategyContext

logger = logging.getLogger("stinger.strategy.break_even")


class BreakEvenMover:
    """One instance per (strategy, symbol-or-all)."""

    def __init__(
        self,
        ctx: StrategyContext,
        *,
        trigger_pips: float,
        lock_pips: float = 0.0,
        symbol: str | None = None,
        point: float = 0.0001,
    ) -> None:
        if trigger_pips <= 0:
            raise ValueError(f"trigger_pips must be > 0, got {trigger_pips!r}")
        self._ctx = ctx
        self._trigger = trigger_pips * point
        self._lock = lock_pips * point
        self._symbol = symbol
        self._point = point
        self._fired: set[int] = set()

    async def on_tick(self, ctx: StrategyContext, tick: Tick) -> None:
        if self._symbol is not None and tick.symbol != self._symbol:
            return
        for pos in ctx.position.for_symbol(tick.symbol):
            if pos.ticket in self._fired:
                continue
            if pos.side is Side.BUY:
                if tick.bid - pos.open_price >= self._trigger:
                    new_sl = pos.open_price + self._lock
                    await self._fire(ctx, pos.ticket, new_sl, tick)
            else:  # SELL
                if pos.open_price - tick.ask >= self._trigger:
                    new_sl = pos.open_price - self._lock
                    await self._fire(ctx, pos.ticket, new_sl, tick)

    async def _fire(
        self, ctx: StrategyContext, ticket: int, new_sl: float, tick: Tick
    ) -> None:
        self._fired.add(ticket)
        await ctx.move_stop(ticket, sl=new_sl, reason="break_even")
        logger.debug(
            "break_even.fire ticket=%s new_sl=%s tick_bid=%s tick_ask=%s",
            ticket, new_sl, tick.bid, tick.ask,
        )
