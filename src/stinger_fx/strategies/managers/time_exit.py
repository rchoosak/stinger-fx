"""TimeExitManager — closes positions that have been open too long.

Supports two (mutually exclusive) exit modes:

* **max_seconds** — closes when ``tick.time - pos.open_time`` exceeds the
  limit. Checked on every tick so resolution is as fine as the tick stream.
* **max_bars** — closes when the number of *closed bars* since the position
  opened exceeds the limit. The runner dispatches ``on_bar`` to the manager
  before the strategy's own ``on_bar``, so this works on any timeframe.

You may supply *either* ``max_seconds`` *or* ``max_bars``, not both.

Usage::

    async def on_start(self, ctx):
        # Time-based: close after 4 hours
        ctx.attach_manager(TimeExitManager(ctx, max_seconds=4 * 3600))

        # Bar-based: close after 8 closed bars on the primary feed
        ctx.attach_manager(TimeExitManager(ctx, max_bars=8))

Parameters
----------
ctx:
    StrategyContext — used for ``ctx.close(ticket)``.
max_seconds:
    Maximum trade duration in seconds.  Mutually exclusive with max_bars.
max_bars:
    Maximum number of closed bars while the position is open.
symbol:
    Filter to a specific symbol. Defaults to ``ctx.symbol``.
"""

from __future__ import annotations

from datetime import timedelta

from stinger_fx.domain import Bar, Tick
from stinger_fx.strategies.context import StrategyContext


class TimeExitManager:
    """Close positions that exceed a time or bar duration limit."""

    def __init__(
        self,
        ctx: StrategyContext,
        *,
        max_seconds: float | None = None,
        max_bars: int | None = None,
        symbol: str | None = None,
    ) -> None:
        if max_seconds is None and max_bars is None:
            raise ValueError("TimeExitManager requires max_seconds or max_bars")
        if max_seconds is not None and max_bars is not None:
            raise ValueError("TimeExitManager accepts max_seconds XOR max_bars, not both")
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError(f"max_seconds must be > 0, got {max_seconds!r}")
        if max_bars is not None and max_bars < 1:
            raise ValueError(f"max_bars must be >= 1, got {max_bars!r}")
        self._max_seconds = max_seconds
        self._max_bars = max_bars
        self._symbol = symbol or ctx.symbol
        # per-ticket bar count (only used in max_bars mode)
        self._bars: dict[int, int] = {}
        # set of tickets already scheduled for close (avoid double-close)
        self._closing: set[int] = set()

    async def on_tick(self, ctx: StrategyContext, tick: Tick) -> None:
        if tick.symbol != self._symbol or self._max_seconds is None:
            return

        limit = timedelta(seconds=self._max_seconds)
        positions = ctx.position.for_symbol(self._symbol)
        live_tickets = {p.ticket for p in positions}

        for pos in positions:
            if pos.ticket in self._closing:
                continue
            elapsed = tick.time - pos.open_time
            if elapsed >= limit:
                self._closing.add(pos.ticket)
                await ctx.close(pos.ticket, reason="time_exit")

        # Purge closed tickets
        for ticket in list(self._closing):
            if ticket not in live_tickets:
                self._closing.discard(ticket)

    async def on_bar(self, ctx: StrategyContext, bar: Bar) -> None:
        if bar.symbol != self._symbol or self._max_bars is None:
            return

        positions = ctx.position.for_symbol(self._symbol)
        live_tickets = {p.ticket for p in positions}

        for pos in positions:
            if pos.ticket in self._closing:
                continue
            # Only count bars that closed *after* the position opened.
            if bar.time >= pos.open_time:
                self._bars[pos.ticket] = self._bars.get(pos.ticket, 0) + 1

            if self._bars.get(pos.ticket, 0) >= self._max_bars:
                self._closing.add(pos.ticket)
                await ctx.close(pos.ticket, reason="time_exit_bars")

        # Purge closed tickets
        for ticket in list(self._bars.keys()):
            if ticket not in live_tickets:
                del self._bars[ticket]
        for ticket in list(self._closing):
            if ticket not in live_tickets:
                self._closing.discard(ticket)
