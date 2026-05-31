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


class BaseStrategy(ABC):  # noqa: B024 — lifecycle hooks are intentionally all optional; ABC reserves the name even without abstract members
    """All strategies subclass this and override the relevant lifecycle hooks."""

    name: ClassVar[str] = ""
    version: ClassVar[str] = "0.1.0"
    Params: ClassVar[type[StrategyParams]] = StrategyParams

    @classmethod
    def subscriptions(cls, params: StrategyParams) -> list[Subscription]:
        """Override to declare which (symbol, timeframe) feeds the strategy needs."""
        return []

    @classmethod
    def warmup_bars(
        cls, params: StrategyParams,
    ) -> dict[Subscription, int] | None:
        """Override to declare per-feed warmup requirements for live mode.

        Return a dict mapping each declared subscription to the number of
        **closed bars** the strategy needs before its indicators / state
        are warm.  The live runtime translates this to a tick-history
        window (``bars × tf.seconds``) and backfills the BarAggregator
        via ``MT5Broker.get_history_ticks`` before opening the live tick
        pump — so the strategy's first ``on_bar()`` lands on already-warm
        history rather than a cold-start partial bar.

        Return ``None`` (default) to accept the conservative 48-hour
        backfill applied to every subscription.  48h covers:
          * Indicators up to ``EMA(96)`` on H1
          * Session-anchored indicators (VWAP session, daily-open levels)
          * Most multi-day swing patterns

        Strategies that need longer (e.g. ``EMA(200)`` on H4) should
        override; strategies that want faster startup (ORB needs only
        ~2h) should override too to avoid over-fetching.

        The dict only needs entries for subscriptions whose default 48h
        would be *too small* OR where the strategy wants a tighter
        window than the default.  Missing entries fall back to the 48h
        default.

        Example::

            @classmethod
            def warmup_bars(cls, params):
                return {
                    Subscription(params.symbol, Timeframe.M1): 1,
                    Subscription(params.symbol, Timeframe.M5):
                        max(params.range_lookback_bars, params.atr_period + 1),
                    Subscription(params.symbol, Timeframe.M15):
                        2 * params.adx_period,
                }
        """
        return None

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
