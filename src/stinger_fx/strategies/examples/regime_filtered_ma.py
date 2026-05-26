"""MA-crossover with a TrendingFilter — trade only in trending markets.

Same logic as :class:`MACrossover` (the base example), but every
crossover signal goes through a `TrendingFilter` gate first. When ADX
is below the configured threshold the strategy stays in cash — chop
kills MA crossover strategies, so this filter routinely turns a
break-even backtest into a positive-edge one.

Useful as a demo of the regime-filter pattern: wrap any signal-emitting
call site in `if self._filter.allows(bars): ...`.
"""

from __future__ import annotations

from pydantic import Field

from stinger_fx.domain import Bar, Subscription, Timeframe
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.indicators import sma
from stinger_fx.strategies.parameters import StrategyParams
from stinger_fx.strategies.regime import TrendingFilter


class RegimeFilteredMAParams(StrategyParams):
    symbol: str = "EURUSD"
    timeframe: Timeframe = Timeframe.M15
    fast: int = Field(10, ge=2, le=200)
    slow: int = Field(30, ge=5, le=500)
    volume: float = Field(0.01, gt=0)
    # Regime filter knobs
    adx_period: int = Field(14, ge=2)
    adx_threshold: float = Field(25.0, gt=0)


class RegimeFilteredMA(BaseStrategy):
    name = "regime_filtered_ma"
    Params = RegimeFilteredMAParams

    @classmethod
    def subscriptions(cls, params: StrategyParams) -> list[Subscription]:
        assert isinstance(params, RegimeFilteredMAParams)
        return [Subscription(symbol=params.symbol, timeframe=params.timeframe)]

    async def on_start(self, ctx: StrategyContext) -> None:
        params = ctx.params
        assert isinstance(params, RegimeFilteredMAParams)
        self._filter = TrendingFilter(
            adx_period=params.adx_period, threshold=params.adx_threshold
        )

    async def on_bar(self, ctx: StrategyContext, bar: Bar) -> None:
        params = ctx.params
        assert isinstance(params, RegimeFilteredMAParams)
        bars = ctx.history.bars()
        if len(bars) < max(params.slow + 2, 2 * params.adx_period):
            return

        # --- Regime gate --------------------------------------------------
        # Only emit signals when ADX confirms a trend. Below threshold the
        # strategy stays flat for this bar.
        if not self._filter.allows(bars):
            return

        # --- MA-crossover logic (same as MACrossover) ---------------------
        closes = ctx.history.closes()
        fast_now = sma(closes, params.fast)
        slow_now = sma(closes, params.slow)
        fast_prev = sma(closes[:-1], params.fast)
        slow_prev = sma(closes[:-1], params.slow)
        if None in (fast_now, slow_now, fast_prev, slow_prev):
            return

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now  # type: ignore[operator]
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now  # type: ignore[operator]
        if crossed_up:
            await ctx.buy(params.volume)
        elif crossed_down:
            await ctx.sell(params.volume)
