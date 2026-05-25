"""StrategyContext — facade passed into every strategy lifecycle hook.

Holds the *current* params (atomically swapped on hot-reload), a logger bound
with the strategy id, the clock, and views over market history / positions /
account. Strategy code calls `ctx.buy()` / `ctx.sell()` / `ctx.submit_signal()`
to interact with the broker — never the broker directly.

The OrderRouter (injected via `signal_sink`) converts signals into orders and
runs risk checks.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from stinger_fx.core.clock import Clock
from stinger_fx.domain import (
    Bar,
    OrderRequest,
    Position,
    Side,
    Signal,
    SignalStrength,
    Subscription,
    Tick,
    Timeframe,
)

if TYPE_CHECKING:
    from stinger_fx.strategies.parameters import StrategyParams

SignalSink = Callable[[Signal], Awaitable[None]]
OrderSink = Callable[[OrderRequest], Awaitable[None]]


class HistoryView:
    """Rolling buffer of bars and the latest tick for a (symbol, timeframe).

    The runner appends to this buffer as BarEvent/TickEvent arrive for the
    strategy's subscriptions.
    """

    def __init__(self, symbol: str, timeframe: Timeframe, capacity: int = 2000) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self._bars: deque[Bar] = deque(maxlen=capacity)
        self._last_tick: Tick | None = None

    def append_bar(self, bar: Bar) -> None:
        if not bar.is_closed:
            return
        self._bars.append(bar)

    def update_tick(self, tick: Tick) -> None:
        self._last_tick = tick

    def bars(self, n: int | None = None) -> tuple[Bar, ...]:
        if n is None:
            return tuple(self._bars)
        return tuple(list(self._bars)[-n:])

    def closes(self, n: int | None = None) -> list[float]:
        return [b.close for b in self.bars(n)]

    def last_tick(self) -> Tick | None:
        return self._last_tick


class PositionView:
    """View into the engine's position cache, filtered by magic number."""

    def __init__(self, magic: int) -> None:
        self.magic = magic
        self._positions: list[Position] = []

    def update(self, positions: list[Position]) -> None:
        self._positions = [p for p in positions if p.magic == self.magic]

    def all(self) -> list[Position]:
        return list(self._positions)

    def for_symbol(self, symbol: str) -> list[Position]:
        return [p for p in self._positions if p.symbol == symbol]

    def net_volume(self, symbol: str) -> float:
        return sum(p.volume * p.side.sign for p in self.for_symbol(symbol))


class StrategyContext:
    """Handed to every strategy hook. One per strategy instance.

    `symbol` / `timeframe` / `history` reference the *primary* feed (first
    subscription returned by `strategy.subscriptions(params)`). Multi-feed
    strategies access additional feeds via `ctx.history_for(symbol, tf)`
    or by iterating `ctx.histories`.
    """

    def __init__(
        self,
        *,
        strategy_id: str,
        symbol: str,
        timeframe: Timeframe,
        params: StrategyParams,
        clock: Clock,
        logger: logging.Logger,
        magic: int,
        signal_sink: SignalSink,
        history_capacity: int = 2000,
        subscriptions: list[Subscription] | None = None,
    ) -> None:
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.params = params
        self.clock = clock
        self.logger = logger
        self.magic = magic
        self._signal_sink = signal_sink
        self.position = PositionView(magic)
        # Build a HistoryView per declared subscription so multi-feed strategies
        # can read each feed independently. The primary view stays accessible
        # via `.history` for back-compat with single-feed strategies.
        subs = subscriptions or [Subscription(symbol=symbol, timeframe=timeframe)]
        self.histories: dict[Subscription, HistoryView] = {
            sub: HistoryView(sub.symbol, sub.timeframe, capacity=history_capacity)
            for sub in subs
        }
        primary_sub = Subscription(symbol=symbol, timeframe=timeframe)
        self.history = self.histories.get(
            primary_sub,
            HistoryView(symbol, timeframe, capacity=history_capacity),
        )
        # If the primary sub wasn't in the declared subscriptions list, also
        # register it so `_route_bar` and direct primary-feed access still work.
        if primary_sub not in self.histories:
            self.histories[primary_sub] = self.history

    def history_for(self, symbol: str, timeframe: Timeframe) -> HistoryView | None:
        """Look up the HistoryView for any declared (symbol, timeframe)."""
        return self.histories.get(Subscription(symbol=symbol, timeframe=timeframe))

    # --- Trading helpers ----------------------------------------------------

    async def buy(
        self,
        volume: float,
        *,
        sl: float | None = None,
        tp: float | None = None,
        comment: str = "",
    ) -> None:
        await self.submit_signal(
            Signal(
                strategy_id=self.strategy_id,
                time=self.clock.now(),
                symbol=self.symbol,
                side=Side.BUY,
                strength=SignalStrength.NORMAL,
                suggested_volume=volume,
                suggested_sl=sl,
                suggested_tp=tp,
                comment=comment,
            )
        )

    async def sell(
        self,
        volume: float,
        *,
        sl: float | None = None,
        tp: float | None = None,
        comment: str = "",
    ) -> None:
        await self.submit_signal(
            Signal(
                strategy_id=self.strategy_id,
                time=self.clock.now(),
                symbol=self.symbol,
                side=Side.SELL,
                strength=SignalStrength.NORMAL,
                suggested_volume=volume,
                suggested_sl=sl,
                suggested_tp=tp,
                comment=comment,
            )
        )

    async def submit_signal(self, signal: Signal) -> None:
        await self._signal_sink(signal)

    # Used by the reloader to swap params atomically
    def _replace_params(self, new_params: StrategyParams) -> None:
        self.params = new_params
