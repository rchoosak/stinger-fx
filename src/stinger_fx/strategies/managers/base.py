"""Base protocol for position managers.

A manager is a small piece of stateful logic that watches every tick (and
optionally every closed bar) for positions tagged with the host strategy's
magic number. Managers can move SL, close positions, or open new layers.

Examples: trailing stop, break-even mover, time-based exits, ladder entry.

The protocol is intentionally minimal — `on_tick` is required, `on_bar` is
optional (default no-op). Managers that only care about time can implement
`on_tick` alone; managers that count bars implement both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from stinger_fx.domain import Bar, Tick
    from stinger_fx.strategies.context import StrategyContext


@runtime_checkable
class PositionManager(Protocol):
    """Anything with an async `on_tick(ctx, tick)` method qualifies.

    Optional hooks (the runner uses `hasattr` to guard dispatch, so they
    aren't required by the Protocol and existing tick-only managers stay
    conformant):

    * ``on_bar(ctx, bar)`` — runs before the strategy's ``on_bar`` for any
      closed bar on a declared feed. Useful for bar-counting (e.g.
      :class:`TimeExitManager` with ``max_bars``).
    * ``on_position_closed(ctx, position)`` — runs before the strategy's
      ``on_position_closed`` whenever a position owned by this strategy
      closes (SL/TP fired, manager closed it, manual close). Useful for
      OCO groups, where closing one member cancels the others.
    """

    async def on_tick(self, ctx: StrategyContext, tick: Tick) -> None: ...

    # async def on_bar(self, ctx: StrategyContext, bar: Bar) -> None: ...
    # async def on_position_closed(self, ctx: StrategyContext, position: Position) -> None: ...
