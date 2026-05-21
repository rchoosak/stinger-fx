"""BaseStrategy — the contract every strategy implements.

A strategy class:
  1. Sets `name`, optional `version`, and `Params` (subclass of StrategyParams).
  2. Defines `subscriptions(params)` returning (symbol, timeframe) pairs.
  3. Overrides one or more `on_*` lifecycle hooks.

Hooks are async so they can `await ctx.buy()` etc. Synchronous strategies can
just write `async def on_bar(...): ...` — there's no penalty.
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

from stinger_fx.domain import (
    Bar,
    Order,
    Position,
    Subscription,
    Tick,
    Timeframe,
)
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.parameters import StrategyParams


class BaseStrategy(ABC):
    """All strategies subclass this and override the relevant lifecycle hooks."""

    name: ClassVar[str] = ""
    version: ClassVar[str] = "0.1.0"
    Params: ClassVar[type[StrategyParams]] = StrategyParams

    @classmethod
    def subscriptions(cls, params: StrategyParams) -> list[Subscription]:
        """Override to declare which (symbol, timeframe) feeds the strategy needs."""
        return []

    # --- Lifecycle hooks (all optional) ------------------------------------

    async def on_start(self, ctx: StrategyContext) -> None: ...
    async def on_stop(self, ctx: StrategyContext) -> None: ...
    async def on_tick(self, ctx: StrategyContext, tick: Tick) -> None: ...
    async def on_bar(self, ctx: StrategyContext, bar: Bar) -> None: ...
    async def on_order_filled(self, ctx: StrategyContext, order: Order) -> None: ...
    async def on_order_rejected(
        self, ctx: StrategyContext, order: Order, reason: str
    ) -> None: ...
    async def on_position_closed(
        self, ctx: StrategyContext, position: Position
    ) -> None: ...
    async def on_params_reloaded(
        self,
        ctx: StrategyContext,
        old: StrategyParams,
        new: StrategyParams,
    ) -> None: ...

    # --- Helpers for subclasses --------------------------------------------

    @classmethod
    def primary_subscription(cls, params: StrategyParams) -> Subscription:
        subs = cls.subscriptions(params)
        if not subs:
            raise ValueError(f"strategy {cls.name!r} declared no subscriptions")
        return subs[0]

    @staticmethod
    def make_subscription(symbol: str, timeframe: Timeframe) -> Subscription:
        return Subscription(symbol=symbol, timeframe=timeframe)
