"""Moving-average crossover example strategy.

Buys on fast-SMA crossing up over slow-SMA; sells on the opposite.
"""

from __future__ import annotations

from pydantic import Field

from stinger_fx.domain import Bar, Subscription, Timeframe
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.indicators import sma
from stinger_fx.strategies.parameters import StrategyParams


class MACrossoverParams(StrategyParams):
    symbol: str = "EURUSD"
    timeframe: Timeframe = Timeframe.M15
    fast: int = Field(10, ge=2, le=200)
    slow: int = Field(30, ge=5, le=500)
    volume: float = Field(0.01, gt=0)


class MACrossover(BaseStrategy):
    name = "ma_crossover"
    Params = MACrossoverParams

    @classmethod
    def subscriptions(cls, params: StrategyParams) -> list[Subscription]:
        assert isinstance(params, MACrossoverParams)
        return [Subscription(symbol=params.symbol, timeframe=params.timeframe)]

    async def on_bar(self, ctx: StrategyContext, bar: Bar) -> None:
        params = ctx.params
        assert isinstance(params, MACrossoverParams)

        closes = ctx.history.closes()
        if len(closes) < params.slow + 2:
            return

        fast_now = sma(closes, params.fast)
        slow_now = sma(closes, params.slow)
        fast_prev = sma(closes[:-1], params.fast)
        slow_prev = sma(closes[:-1], params.slow)
        if None in (fast_now, slow_now, fast_prev, slow_prev):
            return

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now  # type: ignore[operator]
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now  # type: ignore[operator]

        # Close opposite position first
        open_positions = ctx.position.for_symbol(params.symbol)

        if crossed_up:
            ctx.logger.info("signal_buy", fast=fast_now, slow=slow_now, n=len(closes))
            await ctx.buy(params.volume, comment="ma_x_up")
        elif crossed_down:
            ctx.logger.info("signal_sell", fast=fast_now, slow=slow_now, n=len(closes))
            await ctx.sell(params.volume, comment="ma_x_down")
        else:
            ctx.logger.debug(
                "no_cross",
                fast=fast_now,
                slow=slow_now,
                positions=len(open_positions),
            )
