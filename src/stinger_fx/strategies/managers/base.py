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

    `on_bar(ctx, bar)` is optional — implement it when you need to react to
    closed bars (e.g. counting bars-in-trade). The runner dispatches it to
    all managers *before* the strategy's own `on_bar`.
    """

    async def on_tick(self, ctx: StrategyContext, tick: Tick) -> None: ...

    # `on_bar` is intentionally not part of the runtime-checkable Protocol so
    # that existing managers that only implement `on_tick` remain conformant.
    # The runner uses `hasattr` to guard dispatch.
    # async def on_bar(self, ctx: StrategyContext, bar: Bar) -> None: ...
