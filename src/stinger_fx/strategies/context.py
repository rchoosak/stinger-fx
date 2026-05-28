"""StrategyContext — facade passed into every strategy lifecycle hook.

Holds the *current* params (atomically swapped on hot-reload), a logger bound
with the strategy id, the clock, and views over market history / positions /
account. Strategy code calls `ctx.buy()` / `ctx.sell()` / `ctx.submit_signal()`
to interact with the broker — never the broker directly.

The OrderRouter (injected via `signal_sink`) converts signals into orders and
runs risk checks.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import structlog

from stinger_fx.core.clock import Clock
from stinger_fx.core.events import (
    CancelOrderRequestEvent,
    ClosePositionRequestEvent,
    ModifyOrderRequestEvent,
    PartialCloseRequestEvent,
)
from stinger_fx.domain import (
    Bar,
    OrderRequest,
    OrderType,
    Position,
    Side,
    Signal,
    SignalStrength,
    Subscription,
    Tick,
    Timeframe,
)

if TYPE_CHECKING:
    from stinger_fx.core.event_bus import AsyncEventBus
    from stinger_fx.strategies.managers.base import PositionManager
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
        logger: structlog.stdlib.BoundLogger,
        magic: int,
        signal_sink: SignalSink,
        history_capacity: int = 2000,
        subscriptions: list[Subscription] | None = None,
        bus: AsyncEventBus | None = None,
    ) -> None:
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.params = params
        self.clock = clock
        self.logger = logger
        self.magic = magic
        self._signal_sink = signal_sink
        # Bus is needed to publish modify / partial-close requests. Optional
        # so older callers (tests that build a context manually) keep working.
        self._bus = bus
        # Managers attached via ctx.attach_manager() — the runner dispatches
        # on_tick to each one before the strategy's own on_tick.
        self._managers: list[PositionManager] = []
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

    # --- Pending-order helpers (Phase 6.2.B) --------------------------------

    async def buy_stop(
        self,
        price: float,
        volume: float,
        *,
        sl: float | None = None,
        tp: float | None = None,
        comment: str = "",
    ) -> None:
        """Place a BUY STOP — fills when ask reaches/exceeds ``price``.

        Used for breakout entries: buy above resistance once price confirms
        the move.  The pending order survives across ticks until either
        ``price`` is crossed or the strategy cancels it.
        """
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
                order_type=OrderType.STOP,
                suggested_price=price,
                comment=comment,
            )
        )

    async def sell_stop(
        self,
        price: float,
        volume: float,
        *,
        sl: float | None = None,
        tp: float | None = None,
        comment: str = "",
    ) -> None:
        """Place a SELL STOP — fills when bid falls to/below ``price``.

        Used for breakdown entries: sell below support once price confirms.
        """
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
                order_type=OrderType.STOP,
                suggested_price=price,
                comment=comment,
            )
        )

    async def buy_limit(
        self,
        price: float,
        volume: float,
        *,
        sl: float | None = None,
        tp: float | None = None,
        comment: str = "",
    ) -> None:
        """Place a BUY LIMIT — fills when ask falls to/below ``price``.

        Used for pullback entries: buy at a target price below the
        current market.  Fills exactly at ``price`` (limit guarantees
        worst-case execution).
        """
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
                order_type=OrderType.LIMIT,
                suggested_price=price,
                comment=comment,
            )
        )

    async def sell_limit(
        self,
        price: float,
        volume: float,
        *,
        sl: float | None = None,
        tp: float | None = None,
        comment: str = "",
    ) -> None:
        """Place a SELL LIMIT — fills when bid rises to/exceeds ``price``.

        Used for sell-the-rally entries: sell at a target price above
        the current market.  Fills exactly at ``price``.
        """
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
                order_type=OrderType.LIMIT,
                suggested_price=price,
                comment=comment,
            )
        )

    # --- Position management (Phase 4 — modify / partial close) ---------------

    async def move_stop(
        self,
        ticket: int,
        *,
        sl: float | None = None,
        tp: float | None = None,
        reason: str = "",
    ) -> None:
        """Request the broker to update SL / TP on an existing position.

        Ownership is enforced by the router — a strategy may only modify its
        own tickets (matched by `position.magic == ctx.magic`). Cross-strategy
        modifications are rejected without touching the broker.

        Pass `sl=None` (or `tp=None`) to leave that stop unchanged. If both
        are `None` the request is a no-op and is silently dropped here so
        the bus stays quiet.
        """
        if sl is None and tp is None:
            return
        if self._bus is None:
            raise RuntimeError(
                "StrategyContext was built without a bus — modify primitives "
                "require ctx.bus= to be set"
            )
        await self._bus.publish(
            ModifyOrderRequestEvent(
                strategy_id=self.strategy_id,
                ticket=ticket,
                sl=sl,
                tp=tp,
                reason=reason,
            )
        )

    async def move_pending(
        self,
        ticket: int,
        *,
        price: float | None = None,
        stop_price: float | None = None,
        volume: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
        reason: str = "",
    ) -> None:
        """Adjust a pending order's trigger price, volume, SL, or TP.

        Common use cases:
          * Chase a moving level — e.g. trail a BUY_STOP higher as the
            underlying trend confirms
          * Scale up/down the size of a yet-to-trigger order based on
            updated risk budget
          * Tighten a pending order's protective SL before it triggers

        Has no effect on positions that have already filled — use
        :meth:`move_stop` for SL/TP adjustments to open positions.

        Pass any subset of fields; ``None`` means "leave unchanged".
        Raises :class:`ValueError` when no field is supplied so the bus
        stays quiet.
        """
        if all(v is None for v in (price, stop_price, volume, sl, tp)):
            raise ValueError(
                "move_pending requires at least one of: price, stop_price, "
                "volume, sl, tp"
            )
        if self._bus is None:
            raise RuntimeError(
                "StrategyContext was built without a bus — move_pending "
                "requires ctx.bus= to be set"
            )
        await self._bus.publish(
            ModifyOrderRequestEvent(
                strategy_id=self.strategy_id,
                ticket=ticket,
                price=price,
                stop_price=stop_price,
                volume=volume,
                sl=sl,
                tp=tp,
                reason=reason,
            )
        )

    async def partial_close(
        self,
        ticket: int,
        volume: float,
        *,
        reason: str = "",
    ) -> None:
        """Request the broker to reduce an existing position by `volume`.

        Volume must be positive and ≤ the current position volume; the
        router refuses both under- and over-close requests. As with
        `move_stop`, ownership is enforced by magic-number match — a
        strategy can only shrink its own tickets.
        """
        if volume <= 0:
            raise ValueError(f"partial_close volume must be > 0, got {volume!r}")
        if self._bus is None:
            raise RuntimeError(
                "StrategyContext was built without a bus — partial_close "
                "requires ctx.bus= to be set"
            )
        await self._bus.publish(
            PartialCloseRequestEvent(
                strategy_id=self.strategy_id,
                ticket=ticket,
                volume=volume,
                reason=reason,
            )
        )

    async def cancel_order(
        self,
        ticket: int,
        *,
        reason: str = "",
    ) -> None:
        """Request the broker to cancel a pending order.

        Only affects orders still in the SUBMITTED state (i.e. not yet
        triggered into a position). For closing an already-open position,
        use :meth:`close`.

        Ownership is enforced by the router — a strategy can only cancel
        its own tickets (matched by magic).
        """
        if self._bus is None:
            raise RuntimeError(
                "StrategyContext was built without a bus — cancel_order "
                "requires ctx.bus= to be set"
            )
        await self._bus.publish(
            CancelOrderRequestEvent(
                strategy_id=self.strategy_id,
                ticket=ticket,
                reason=reason,
            )
        )

    async def close(
        self,
        ticket: int,
        *,
        reason: str = "",
    ) -> None:
        """Request the broker to fully close an existing position.

        Ownership is enforced by the router — a strategy may only close its
        own tickets (matched by magic). The router routes to `handle_close`
        which calls `broker.close_position(ticket)` (full volume).

        This is distinct from `partial_close` which reduces by a specified
        volume. Use this when you want to exit the whole position immediately
        (e.g. time-based exits, emergency stop).
        """
        if self._bus is None:
            raise RuntimeError(
                "StrategyContext was built without a bus — close "
                "requires ctx.bus= to be set"
            )
        await self._bus.publish(
            ClosePositionRequestEvent(
                strategy_id=self.strategy_id,
                ticket=ticket,
                reason=reason,
            )
        )

    # --- Position managers (Phase 4 C.2) --------------------------------------

    def attach_manager(self, manager: PositionManager) -> None:
        """Attach a position manager (trailing stop, break-even, …) to this
        context. The runner forwards every `on_tick` to every attached manager
        in attachment order *before* the strategy's own `on_tick`, so a manager
        can move SL via `ctx.move_stop()` before the strategy sees the tick.
        """
        self._managers.append(manager)

    @property
    def managers(self) -> list[PositionManager]:
        return list(self._managers)

    # Used by the reloader to swap params atomically
    def _replace_params(self, new_params: StrategyParams) -> None:
        self.params = new_params
