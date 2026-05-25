"""Base protocol for position managers.

A manager is a small piece of stateful logic that watches every tick for
positions tagged with the host strategy's magic number and (optionally)
moves SL via the strategy context. Examples: trailing stop, break-even
mover, time-based exits.

The contract is intentionally tiny — `on_tick(ctx, tick)` only — so a
strategy can compose multiple managers without arguing about lifecycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from stinger_fx.domain import Tick
    from stinger_fx.strategies.context import StrategyContext


@runtime_checkable
class PositionManager(Protocol):
    """Anything with an async `on_tick(ctx, tick)` method qualifies."""

    async def on_tick(self, ctx: StrategyContext, tick: Tick) -> None: ...
