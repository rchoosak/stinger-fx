"""Normal-mode UI — plain stdout status lines + tail of structured logs.

Console output from `stinger_fx.log` already streams through the structlog
console renderer; this runner only adds periodic one-line status summaries
so an operator watching the terminal can see the engine is alive.
"""

from __future__ import annotations

import asyncio
import logging

from stinger_fx.core.event_bus import AsyncEventBus, Subscription
from stinger_fx.core.events import (
    AccountSnapshotEvent,
    BarEvent,
    EngineHeartbeatEvent,
    OrderFilledEvent,
    OrderRejectedEvent,
    StrategyStateChangedEvent,
    TickEvent,
)
from stinger_fx.log import get_logger
from stinger_fx.ui.handle import EngineHandle

logger = get_logger("stinger.ui.normal")
stdlogger = logging.getLogger("stinger.ui.normal")


class NormalUI:
    """Lifecycle component: registers bus subscriptions on start."""

    def __init__(self, handle: EngineHandle, *, heartbeat_seconds: float = 10.0) -> None:
        self._handle = handle
        self._bus: AsyncEventBus = handle.bus
        self._heartbeat = heartbeat_seconds
        self._subs: list[Subscription] = []
        self._hb_task: asyncio.Task[None] | None = None
        self._latest_equity: float | None = None
        self._latest_balance: float | None = None
        self._latest_profit: float | None = None
        # Flow counters reset each heartbeat — so the operator can see at a
        # glance whether ticks/bars are actually arriving.
        self._ticks_since_heartbeat: int = 0
        self._bars_since_heartbeat: int = 0
        self._last_tick_symbol: str | None = None
        self._last_tick_bid: float | None = None
        self._last_tick_ask: float | None = None

    async def start(self) -> None:
        self._subs.append(self._bus.subscribe(OrderFilledEvent, self._on_filled, name="ui.normal.fill"))
        self._subs.append(self._bus.subscribe(OrderRejectedEvent, self._on_rejected, name="ui.normal.reject"))
        self._subs.append(
            self._bus.subscribe(StrategyStateChangedEvent, self._on_strategy, name="ui.normal.strategy")
        )
        self._subs.append(
            self._bus.subscribe(AccountSnapshotEvent, self._on_snapshot, name="ui.normal.snapshot")
        )
        self._subs.append(
            self._bus.subscribe(TickEvent, self._on_tick, name="ui.normal.tick")
        )
        self._subs.append(
            self._bus.subscribe(BarEvent, self._on_bar, name="ui.normal.bar")
        )
        self._hb_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("normal_ui_started")

    async def stop(self) -> None:
        if self._hb_task is not None:
            self._hb_task.cancel()
            try:
                await self._hb_task
            except asyncio.CancelledError:
                pass
            self._hb_task = None
        for sub in self._subs:
            await sub.unsubscribe()
        self._subs.clear()
        logger.info("normal_ui_stopped")

    # --- Handlers -----------------------------------------------------------

    async def _on_filled(self, evt: OrderFilledEvent) -> None:
        o = evt.order
        logger.info(
            "order_filled",
            strategy=o.strategy_id,
            symbol=o.symbol,
            side=o.side.value,
            volume=o.volume,
            price=o.fill_price,
            ticket=o.ticket,
        )

    async def _on_rejected(self, evt: OrderRejectedEvent) -> None:
        o = evt.order
        logger.warning(
            "order_rejected",
            strategy=o.strategy_id,
            symbol=o.symbol,
            side=o.side.value,
            volume=o.volume,
            reason=evt.reason,
        )

    async def _on_strategy(self, evt: StrategyStateChangedEvent) -> None:
        logger.info("strategy_state", id=evt.strategy_id, state=evt.state, reason=evt.reason)

    async def _on_snapshot(self, evt: AccountSnapshotEvent) -> None:
        self._latest_equity = evt.snapshot.equity
        self._latest_balance = evt.snapshot.balance
        self._latest_profit = evt.snapshot.profit

    async def _on_tick(self, evt: TickEvent) -> None:
        self._ticks_since_heartbeat += 1
        self._last_tick_symbol = evt.tick.symbol
        self._last_tick_bid = evt.tick.bid
        self._last_tick_ask = evt.tick.ask

    async def _on_bar(self, evt: BarEvent) -> None:
        if evt.bar.is_closed:
            self._bars_since_heartbeat += 1
            logger.info(
                "bar_closed",
                symbol=evt.bar.symbol,
                timeframe=evt.bar.timeframe.value,
                time=evt.bar.time.isoformat(),
                close=evt.bar.close,
            )

    async def _heartbeat_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._heartbeat)
            except asyncio.CancelledError:
                return
            strategies = await self._handle.list_strategies()
            positions = await self._handle.get_positions()
            logger.info(
                "status",
                strategies=len(strategies),
                positions=len(positions),
                balance=self._latest_balance,
                equity=self._latest_equity,
                profit=self._latest_profit,
                ticks=self._ticks_since_heartbeat,
                bars=self._bars_since_heartbeat,
                last_symbol=self._last_tick_symbol,
                last_bid=self._last_tick_bid,
                last_ask=self._last_tick_ask,
            )
            self._ticks_since_heartbeat = 0
            self._bars_since_heartbeat = 0
            await self._bus.publish(EngineHeartbeatEvent(interval_seconds=self._heartbeat))
