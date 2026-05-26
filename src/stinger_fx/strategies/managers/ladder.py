"""LadderManager — adds positions in layers as price moves favourably.

When a strategy opens an initial BUY position the manager watches bid
price. Every time bid rises by `step_pips` above the *last entry price*
for that ticket, it opens a new `level_volume` BUY, up to `max_levels`
additional entries.  For SELL positions it mirrors the logic on ask.

Usage::

    async def on_start(self, ctx):
        ctx.attach_manager(
            LadderManager(ctx, step_pips=10, max_levels=3, level_volume=0.05)
        )

The manager is deliberately passive — it never opens the *first* position.
That remains the strategy's responsibility. If no position exists the
manager is idle.

Parameters
----------
ctx:
    StrategyContext — used to call ``ctx.buy()`` / ``ctx.sell()``.
step_pips:
    Distance in pips between ladder rungs.
max_levels:
    Maximum number of *additional* entries per ticket.
level_volume:
    Volume (lots) to trade at each rung.
symbol:
    Filter to a specific symbol. Defaults to ``ctx.symbol``.
point:
    One pip in price units (default 0.0001 for 5-digit FX).
"""

from __future__ import annotations

from stinger_fx.domain import Side, Tick
from stinger_fx.strategies.context import StrategyContext


class LadderManager:
    """Pyramid into an existing position as price advances."""

    def __init__(
        self,
        ctx: StrategyContext,
        *,
        step_pips: float,
        max_levels: int,
        level_volume: float,
        symbol: str | None = None,
        point: float = 0.0001,
    ) -> None:
        if step_pips <= 0:
            raise ValueError(f"step_pips must be > 0, got {step_pips!r}")
        if max_levels < 1:
            raise ValueError(f"max_levels must be >= 1, got {max_levels!r}")
        if level_volume <= 0:
            raise ValueError(f"level_volume must be > 0, got {level_volume!r}")
        self._ctx = ctx
        self._step = step_pips * point
        self._max = max_levels
        self._volume = level_volume
        self._symbol = symbol or ctx.symbol
        # Per-ticket state
        self._levels: dict[int, int] = {}      # levels added so far
        self._last_price: dict[int, float] = {}  # price at last entry

    async def on_tick(self, ctx: StrategyContext, tick: Tick) -> None:
        if tick.symbol != self._symbol:
            return

        positions = ctx.position.for_symbol(self._symbol)
        live_tickets = set()

        for pos in positions:
            live_tickets.add(pos.ticket)

            # Register newly seen positions — last_price = open_price so
            # the first rung fires when price has moved step_pips from open.
            if pos.ticket not in self._levels:
                self._levels[pos.ticket] = 0
                self._last_price[pos.ticket] = pos.open_price

            if self._levels[pos.ticket] >= self._max:
                continue

            if pos.side is Side.BUY:
                trigger = self._last_price[pos.ticket] + self._step
                if tick.bid >= trigger:
                    self._levels[pos.ticket] += 1
                    self._last_price[pos.ticket] = tick.bid
                    await ctx.buy(self._volume)
            else:  # SELL
                trigger = self._last_price[pos.ticket] - self._step
                if tick.ask <= trigger:
                    self._levels[pos.ticket] += 1
                    self._last_price[pos.ticket] = tick.ask
                    await ctx.sell(self._volume)

        # Purge closed tickets to avoid memory leak on long runs.
        for ticket in list(self._levels.keys()):
            if ticket not in live_tickets:
                del self._levels[ticket]
                del self._last_price[ticket]
