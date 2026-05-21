"""TradingEngine — the orchestrator that wires bus, broker, strategies, and scheduler.

This is a thin coordinator: it owns the lifecycle of subcomponents and the
reload lock, but business logic lives in those subcomponents (brokers, strategy
runners, order router, etc.).

The engine is intentionally constructed without the broker, strategies, or UI
already instantiated — the entry-point (CLI) assembles those and hands them in.
That keeps `core/` free of broker- and UI-layer imports.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from stinger_fx.core.clock import Clock, LiveClock
from stinger_fx.core.event_bus import AsyncEventBus
from stinger_fx.core.events import EngineStartedEvent, EngineStoppedEvent
from stinger_fx.core.lifecycle import Lifecycle
from stinger_fx.core.scheduler import Scheduler

logger = logging.getLogger("stinger.engine")


class TradingEngine:
    """Owns the event bus and the start/stop sequence of all subcomponents."""

    def __init__(
        self,
        *,
        bus: AsyncEventBus | None = None,
        clock: Clock | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        self.bus: AsyncEventBus = bus or AsyncEventBus()
        self.clock: Clock = clock or LiveClock()
        self.scheduler: Scheduler = scheduler or Scheduler()

        self._components: list[Lifecycle] = []
        self._started = False
        self._stopped = False

        # Reload lock — runners must hold it briefly while dispatching an event
        # so that config-reload diffs cannot swap params mid-handler. See
        # config/reload.py and strategies/runner.py.
        self.reload_lock: asyncio.Lock = asyncio.Lock()

    # --- Component registration ---------------------------------------------

    def register(self, component: Lifecycle) -> None:
        """Register a component that must start/stop with the engine.

        Order matters: start runs in registration order; stop runs in reverse.
        Typical order: broker → data writers → strategies → UI.
        """
        if self._started:
            raise RuntimeError("cannot register after start")
        self._components.append(component)

    # --- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        logger.info("engine starting components=%d", len(self._components))
        for c in self._components:
            await c.start()
        await self.scheduler.start()
        await self.bus.publish(EngineStartedEvent())
        logger.info("engine started")

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        logger.info("engine stopping")
        await self.scheduler.stop()
        # Stop in reverse registration order
        for c in reversed(self._components):
            try:
                await c.stop()
            except Exception:
                logger.exception("component stop failed component=%s", type(c).__name__)
        await self.bus.publish(EngineStoppedEvent())
        await self.bus.close()
        logger.info("engine stopped")

    # --- Convenience ---------------------------------------------------------

    async def run_until(self, ready: Callable[[], Awaitable[bool]] | None = None) -> None:
        """Start the engine and block until SIGINT/SIGTERM or `ready()` returns True.

        The CLI entrypoint usually composes this with a signal handler.
        """
        await self.start()
        try:
            stop_evt = asyncio.Event()
            if ready is None:
                await stop_evt.wait()
            else:
                while not await ready():
                    await asyncio.sleep(0.1)
        finally:
            await self.stop()
