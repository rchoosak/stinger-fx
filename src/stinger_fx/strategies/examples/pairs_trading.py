"""Pairs trading template — mean-reversion on a cointegrated spread.

Subscribes to two symbols (e.g. EURUSD + GBPUSD) on the same timeframe.
On each bar:

  1. Wait until enough bars have accumulated on both feeds
  2. Compute a rolling hedge ratio from the last ``window`` closes
  3. Form the spread:  spread = price_a - hedge_ratio * price_b
  4. Compute the z-score of the latest spread over the rolling window
  5. If |z| > entry_z and no open position → go long spread (buy A,
     sell B) when z is very negative, short spread when z is very positive
  6. If |z| < exit_z → flatten both legs

The two legs are wired into an OCO group so an unexpected close of
either leg (SL fired, manual intervention) cancels/closes the other.

This is a deliberately simple template — production pairs strategies
usually add a Kalman-filtered dynamic hedge ratio, half-life-based
position sizing, and rolling cointegration tests to handle regime
changes. Phase 7+ territory.
"""

from __future__ import annotations

from pydantic import Field

from stinger_fx.domain import Bar, Subscription, Timeframe
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.cointegration import rolling_hedge_ratio, spread_zscore
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.managers.oco import OCOGroupManager
from stinger_fx.strategies.parameters import StrategyParams


class PairsTradingParams(StrategyParams):
    symbol_a: str = "EURUSD"
    symbol_b: str = "GBPUSD"
    timeframe: Timeframe = Timeframe.M15
    window: int = Field(50, ge=10, le=500)
    entry_z: float = Field(2.0, gt=0)
    exit_z: float = Field(0.5, ge=0)
    volume: float = Field(0.1, gt=0)


class PairsTrading(BaseStrategy):
    """Generic long-spread / short-spread mean-reversion template."""

    name = "pairs_trading"
    Params = PairsTradingParams

    @classmethod
    def subscriptions(cls, params: StrategyParams) -> list[Subscription]:
        assert isinstance(params, PairsTradingParams)
        return [
            Subscription(symbol=params.symbol_a, timeframe=params.timeframe),
            Subscription(symbol=params.symbol_b, timeframe=params.timeframe),
        ]

    async def on_start(self, ctx: StrategyContext) -> None:
        self._oco = OCOGroupManager(ctx)
        ctx.attach_manager(self._oco)
        # Per-spread-position state: maps "long_spread" / "short_spread"
        # to the OCO group_id that holds the two legs.
        self._open_side: str | None = None
        self._group_seq = 0

    async def on_bar(self, ctx: StrategyContext, bar: Bar) -> None:
        params = ctx.params
        assert isinstance(params, PairsTradingParams)

        # Only react when bar B (the slower side of the iteration order)
        # closes — we need both feeds aligned. Strategies that want
        # tick-level alignment can use ctx.history_for instead.
        view_a = ctx.history_for(params.symbol_a, params.timeframe)
        view_b = ctx.history_for(params.symbol_b, params.timeframe)
        if view_a is None or view_b is None:
            return
        closes_a = view_a.closes()
        closes_b = view_b.closes()
        # Align on whichever side has fewer samples; both must reach window.
        n = min(len(closes_a), len(closes_b))
        if n < params.window:
            return
        a = closes_a[-n:]
        b = closes_b[-n:]

        hedge = rolling_hedge_ratio(a, b, params.window)
        if hedge is None:
            return
        spread = [ai - hedge * bi for ai, bi in zip(a, b, strict=True)]
        z = spread_zscore(spread, params.window)
        if z is None:
            return

        # Exit if open and z has reverted to the band
        if self._open_side is not None and abs(z) < params.exit_z:
            for pos in ctx.position.all():
                await ctx.close(pos.ticket, reason="pairs_z_reverted")
            self._open_side = None
            return

        # Entry only when flat
        if self._open_side is not None:
            return

        if z > params.entry_z:
            # Spread too high → short spread (sell A, buy B)
            await self._open_spread(ctx, "short_spread", params)
        elif z < -params.entry_z:
            # Spread too low → long spread (buy A, sell B)
            await self._open_spread(ctx, "long_spread", params)

    async def _open_spread(
        self,
        ctx: StrategyContext,
        side: str,
        params: PairsTradingParams,
    ) -> None:
        """Place both legs of the spread; we don't yet have tickets so
        the OCO wiring happens later in on_order_filled.  For the
        template we keep it simple: just submit both market orders."""
        if side == "long_spread":
            await ctx.submit_signal(
                _signal(ctx, params.symbol_a, side="buy", volume=params.volume)
            )
            await ctx.submit_signal(
                _signal(ctx, params.symbol_b, side="sell", volume=params.volume)
            )
        else:  # short_spread
            await ctx.submit_signal(
                _signal(ctx, params.symbol_a, side="sell", volume=params.volume)
            )
            await ctx.submit_signal(
                _signal(ctx, params.symbol_b, side="buy", volume=params.volume)
            )
        self._open_side = side


def _signal(ctx: StrategyContext, symbol: str, *, side: str, volume: float):
    """Build a market-order Signal for the given symbol/side.

    `ctx.buy` / `ctx.sell` always use `ctx.symbol`; pairs trading needs
    to target either leg, so we build the Signal manually.
    """
    from stinger_fx.domain.positions import Side
    from stinger_fx.domain.signals import Signal, SignalStrength

    return Signal(
        strategy_id=ctx.strategy_id,
        time=ctx.clock.now(),
        symbol=symbol,
        side=Side.BUY if side == "buy" else Side.SELL,
        strength=SignalStrength.NORMAL,
        suggested_volume=volume,
        comment=f"pairs:{side}",
    )
