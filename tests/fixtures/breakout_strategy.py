"""Minimal breakout strategy for the bar-mode pending-orders regression test.

Places a single BUY_STOP at a fixed price on the very first bar it sees,
then does nothing. The point isn't to be a useful strategy — it's to
exercise the end-to-end path: ctx.buy_stop → SignalEvent → OrderRouter
→ SimBroker.place_order (queued as pending) → bar-mode replay's
check_pending_bar → OrderFilledEvent.

Pre-fix this whole chain ran to "queued as pending" and then stopped,
because bar-mode replay never called the equivalent of check_pending.
The strategy's on_order_filled never fired.
"""

from __future__ import annotations

from pydantic import Field

from stinger_fx.domain import Subscription, Timeframe
from stinger_fx.strategies.base import BaseStrategy, StrategyParams
from stinger_fx.strategies.context import StrategyContext


class BreakoutOnceParams(StrategyParams):
    symbol: str = "EURUSD"
    timeframe: Timeframe = Timeframe.M15
    trigger_price: float = Field(1.1010, gt=0)
    volume: float = Field(0.1, gt=0)


class BreakoutOnce(BaseStrategy):
    """Place a BUY_STOP on the first bar, then idle."""

    name = "breakout_once"
    Params = BreakoutOnceParams

    @classmethod
    def subscriptions(cls, params):
        return [Subscription(symbol=params.symbol, timeframe=params.timeframe)]

    def __init__(self) -> None:
        super().__init__()
        self._placed = False

    async def on_bar(self, ctx: StrategyContext, bar) -> None:
        if self._placed:
            return
        self._placed = True
        # ctx.params is typed as the StrategyParams base; we set Params on this
        # class to BreakoutOnceParams so the runtime instance has the extra
        # fields. assert keeps mypy honest.
        params = ctx.params
        assert isinstance(params, BreakoutOnceParams)
        await ctx.buy_stop(price=params.trigger_price, volume=params.volume)

    async def on_order_filled(self, ctx: StrategyContext, order) -> None:
        # Mark on the strategy instance so the test can observe the fill
        # arrived through the runner's dispatch (not just the bus).
        self._last_filled_ticket = order.ticket
