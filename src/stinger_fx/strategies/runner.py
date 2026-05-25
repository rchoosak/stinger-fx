"""StrategyRunner — per-strategy isolation around a `BaseStrategy` instance.

Responsibilities:
  • Subscribe to TickEvent/BarEvent filtered by this strategy's symbol/timeframe.
  • Maintain a rolling history buffer in `ctx.history`.
  • Dispatch events to the user's `on_*` hooks under the engine's reload lock.
  • Track an error budget and quarantine the strategy when it's exceeded.
  • Provide `update_params()`, `pause()`, `resume()`, `stop()` for the reloader.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import deque

from stinger_fx.core.clock import Clock
from stinger_fx.core.errors import StrategyQuarantinedError
from stinger_fx.core.event_bus import AsyncEventBus
from stinger_fx.core.event_bus import Subscription as BusSubscription
from stinger_fx.core.events import (
    BarEvent,
    OrderFilledEvent,
    OrderModifiedEvent,
    OrderRejectedEvent,
    PartialClosedEvent,
    PositionClosedEvent,
    StrategyStateChangedEvent,
    TickEvent,
)
from stinger_fx.domain import Position, Subscription
from stinger_fx.log import get_logger
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.context import SignalSink, StrategyContext
from stinger_fx.strategies.parameters import StrategyParams

# Error budget — if a strategy raises more than this many times within
# the window, it gets quarantined.
MAX_ERRORS_PER_MINUTE = 5
QUARANTINE_WINDOW_SECONDS = 60.0


def derive_magic(strategy_id: str) -> int:
    """Deterministic 63-bit magic-number from a strategy id."""
    h = hashlib.sha256(strategy_id.encode("utf-8")).digest()
    # 63-bit positive (avoid sign bit on platforms that treat int as signed)
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


class StrategyRunner:
    """One per active strategy instance."""

    def __init__(
        self,
        *,
        strategy_id: str,
        strategy: BaseStrategy,
        params: StrategyParams,
        bus: AsyncEventBus,
        clock: Clock,
        reload_lock: asyncio.Lock,
        signal_sink: SignalSink,
    ) -> None:
        self.id = strategy_id
        self.strategy = strategy
        self.bus = bus
        self.clock = clock
        self._reload_lock = reload_lock
        self._params = params
        self._magic = derive_magic(strategy_id)
        self._logger = get_logger(f"stinger.strategy.{strategy_id}")
        self._subscriptions: list[BusSubscription] = []
        self._errors: deque[float] = deque()
        self._paused = False
        self._stopped = False
        self._quarantined = False
        self._ctx: StrategyContext | None = None
        self._signal_sink = signal_sink

    # --- Lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        # The strategy can declare multiple subscriptions; the first is primary
        # and drives ctx.symbol/timeframe. Additional subs get their own
        # HistoryView so the strategy can read each feed via ctx.history_for().
        subs = self.strategy.subscriptions(self._params)
        if not subs:
            self._logger.warning("strategy declared no subscriptions; nothing to do")
            return
        primary = subs[0]
        self._ctx = StrategyContext(
            strategy_id=self.id,
            symbol=primary.symbol,
            timeframe=primary.timeframe,
            params=self._params,
            clock=self.clock,
            logger=self._logger,
            magic=self._magic,
            signal_sink=self._signal_sink,
            subscriptions=list(subs),
            bus=self.bus,
        )

        # Subscribe to every needed feed; one bus subscription per event type
        # is enough because the handler filters by (symbol, timeframe).
        self._subscriptions.append(self.bus.subscribe(TickEvent, self._on_tick, name=f"{self.id}.tick"))
        self._subscriptions.append(self.bus.subscribe(BarEvent, self._on_bar, name=f"{self.id}.bar"))
        self._subscriptions.append(
            self.bus.subscribe(OrderFilledEvent, self._on_order_filled, name=f"{self.id}.fill")
        )
        self._subscriptions.append(
            self.bus.subscribe(OrderRejectedEvent, self._on_order_rejected, name=f"{self.id}.reject")
        )
        self._subscriptions.append(
            self.bus.subscribe(PositionClosedEvent, self._on_position_closed, name=f"{self.id}.close")
        )
        # Position-state tracking for ctx.position / position managers (Phase 4)
        self._subscriptions.append(
            self.bus.subscribe(OrderModifiedEvent, self._on_order_modified, name=f"{self.id}.modify")
        )
        self._subscriptions.append(
            self.bus.subscribe(PartialClosedEvent, self._on_partial_closed, name=f"{self.id}.partial")
        )

        # User hook
        await self._guarded(self.strategy.on_start, self._ctx)
        await self.bus.publish(
            StrategyStateChangedEvent(strategy_id=self.id, state="started")
        )

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        for sub in self._subscriptions:
            await sub.unsubscribe()
        self._subscriptions.clear()
        if self._ctx is not None:
            await self._guarded(self.strategy.on_stop, self._ctx)
        await self.bus.publish(
            StrategyStateChangedEvent(strategy_id=self.id, state="stopped")
        )

    async def pause(self) -> None:
        self._paused = True
        await self.bus.publish(
            StrategyStateChangedEvent(strategy_id=self.id, state="paused")
        )

    async def resume(self) -> None:
        self._paused = False
        await self.bus.publish(
            StrategyStateChangedEvent(strategy_id=self.id, state="started")
        )

    async def update_params(self, new_params: StrategyParams) -> None:
        """Atomically swap params and call on_params_reloaded."""
        old = self._params
        async with self._reload_lock:
            self._params = new_params
            if self._ctx is not None:
                self._ctx._replace_params(new_params)
        if self._ctx is not None:
            await self._guarded(self.strategy.on_params_reloaded, self._ctx, old, new_params)

    # --- Event dispatch -----------------------------------------------------

    async def _on_tick(self, evt: TickEvent) -> None:
        if not self._active() or self._ctx is None:
            return
        # Update any HistoryView whose symbol matches this tick. Multi-feed
        # strategies with several timeframes for the same symbol get every
        # view's last_tick refreshed in one event.
        any_match = False
        for sub, view in self._ctx.histories.items():
            if sub.symbol == evt.tick.symbol:
                view.update_tick(evt.tick)
                any_match = True
        if not any_match:
            return
        # Position managers (trailing stop, break-even, …) run BEFORE the
        # strategy's own on_tick so they can move SL before the strategy
        # reacts to this tick. They get every same-symbol tick regardless
        # of whether the strategy's `on_tick` would fire — managers may
        # care about non-primary symbols (e.g. trailing on a secondary
        # feed where the entry signal came from a different symbol).
        for manager in self._ctx.managers:
            await self._guarded(manager.on_tick, self._ctx, evt.tick)
        # on_tick fires only for the primary feed to preserve single-feed
        # strategy ergonomics; multi-feed strategies that need every tick can
        # iterate ctx.histories themselves on each bar.
        if evt.tick.symbol == self._ctx.symbol:
            await self._guarded(self.strategy.on_tick, self._ctx, evt.tick)

    async def _on_bar(self, evt: BarEvent) -> None:
        if not self._active() or self._ctx is None:
            return
        sub = Subscription(symbol=evt.bar.symbol, timeframe=evt.bar.timeframe)
        view = self._ctx.histories.get(sub)
        if view is None:
            # Not a feed this strategy declared — ignore. (Other strategies
            # on the same bus may care.)
            return
        view.append_bar(evt.bar)
        if not evt.bar.is_closed:
            return
        # Position managers that implement on_bar (e.g. TimeExitManager with
        # bar-counting) run before the strategy's own on_bar.
        for manager in self._ctx.managers:
            if hasattr(manager, "on_bar"):
                await self._guarded(manager.on_bar, self._ctx, evt.bar)
        # Fire on_bar for ANY of the strategy's declared feeds. The strategy
        # is responsible for routing by (bar.symbol, bar.timeframe) when it
        # cares. Back-compat for single-feed strategies is preserved because
        # only the primary feed's bars ever arrive.
        await self._guarded(self.strategy.on_bar, self._ctx, evt.bar)

    async def _on_order_filled(self, evt: OrderFilledEvent) -> None:
        if not self._active() or evt.order.strategy_id != self.id:
            return
        if self._ctx is None:
            return
        # Materialise an open position from the fill so ctx.position
        # (and any attached managers) can see it.
        o = evt.order
        if o.fill_price is not None and o.filled_at is not None:
            self._track_open(
                Position(
                    ticket=o.ticket,
                    symbol=o.symbol,
                    side=o.side,
                    volume=o.filled_volume or o.volume,
                    open_price=o.fill_price,
                    open_time=o.filled_at,
                    sl=o.sl,
                    tp=o.tp,
                    comment=o.comment,
                    magic=o.magic,
                )
            )
        await self._guarded(self.strategy.on_order_filled, self._ctx, evt.order)

    async def _on_order_rejected(self, evt: OrderRejectedEvent) -> None:
        if not self._active() or evt.order.strategy_id != self.id:
            return
        if self._ctx is None:
            return
        await self._guarded(self.strategy.on_order_rejected, self._ctx, evt.order, evt.reason)

    async def _on_position_closed(self, evt: PositionClosedEvent) -> None:
        if not self._active() or evt.position.magic != self._magic:
            return
        if self._ctx is None:
            return
        self._track_remove(evt.position.ticket)
        await self._guarded(self.strategy.on_position_closed, self._ctx, evt.position)

    async def _on_order_modified(self, evt: OrderModifiedEvent) -> None:
        if not self._active() or evt.position.magic != self._magic:
            return
        if self._ctx is None:
            return
        # Refresh the tracked snapshot so trailing-stop comparisons see the
        # new SL on the next tick.
        self._track_open(evt.position)

    async def _on_partial_closed(self, evt: PartialClosedEvent) -> None:
        if not self._active() or evt.position.magic != self._magic:
            return
        if self._ctx is None:
            return
        # `evt.position` is the remaining leg with the shrunken volume.
        self._track_open(evt.position)

    # --- Position tracking --------------------------------------------------

    def _track_open(self, pos: Position) -> None:
        """Add or refresh a tracked position. Filters by magic — only the
        strategy's own positions go into ctx.position."""
        if self._ctx is None or pos.magic != self._magic:
            return
        existing = list(self._ctx.position.all())
        # Replace any existing entry with the same ticket
        replaced = [p for p in existing if p.ticket != pos.ticket]
        replaced.append(pos)
        self._ctx.position.update(replaced)

    def _track_remove(self, ticket: int) -> None:
        if self._ctx is None:
            return
        existing = list(self._ctx.position.all())
        self._ctx.position.update([p for p in existing if p.ticket != ticket])

    # --- Internals ----------------------------------------------------------

    def _active(self) -> bool:
        return not (self._stopped or self._paused or self._quarantined)

    async def _guarded(self, fn, *args, **kwargs) -> None:
        """Run a user hook under the reload lock with error budget tracking."""
        async with self._reload_lock:
            try:
                await fn(*args, **kwargs)
                return
            except StrategyQuarantinedError:
                raise
            except Exception as e:
                self._logger.exception("strategy hook raised", hook=fn.__name__)
                self._record_error()
                if self._quarantined:
                    await self.bus.publish(
                        StrategyStateChangedEvent(
                            strategy_id=self.id,
                            state="quarantined",
                            reason=str(e),
                        )
                    )

    def _record_error(self) -> None:
        now = time.monotonic()
        self._errors.append(now)
        cutoff = now - QUARANTINE_WINDOW_SECONDS
        while self._errors and self._errors[0] < cutoff:
            self._errors.popleft()
        if len(self._errors) > MAX_ERRORS_PER_MINUTE:
            self._quarantined = True
            self._logger.error(
                "strategy quarantined: too many errors",
                count=len(self._errors),
                window_s=QUARANTINE_WINDOW_SECONDS,
            )
