"""In-process async pub/sub.

Each subscriber owns an `asyncio.Queue` so a slow subscriber cannot stall the
publisher. Different overflow policies apply per event type:

  • drop_oldest — used for high-volume telemetry (TickEvent) where the freshest
    value is what matters; older queued items are evicted when the queue is full.
  • block       — used for order-flow events; we never silently drop trades.

Subscribers run as long-lived tasks consuming their queue. Errors raised by a
handler are logged but do not propagate to the bus or to other subscribers.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from stinger_fx.core.events import Event

E = TypeVar("E", bound=Event)
Handler = Callable[[E], Awaitable[None]]

logger = logging.getLogger("stinger.engine.bus")

DROP_OLDEST = "drop_oldest"
BLOCK = "block"


@dataclass
class Subscription[E: Event]:
    """Handle returned by `subscribe()` — call `unsubscribe()` to stop."""

    event_type: type[E]
    handler: Handler[E]
    queue: asyncio.Queue[E]
    task: asyncio.Task[None] = field(repr=False)
    overflow: str = DROP_OLDEST
    closed: bool = False
    name: str = ""

    async def unsubscribe(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass


class AsyncEventBus:
    """Typed in-process event bus.

    Subscriptions are matched by `isinstance(event, sub.event_type)` so that
    subscribing to `Event` catches everything and subscribing to a parent class
    catches its subclasses.
    """

    def __init__(self) -> None:
        self._subs: list[Subscription] = []
        self._closed = False

    def subscribe(
        self,
        event_type: type[E],
        handler: Handler[E],
        *,
        maxsize: int = 1024,
        overflow: str = DROP_OLDEST,
        name: str = "",
    ) -> Subscription[E]:
        if self._closed:
            raise RuntimeError("bus is closed")
        if overflow not in (DROP_OLDEST, BLOCK):
            raise ValueError(f"unknown overflow policy: {overflow}")

        queue: asyncio.Queue[E] = asyncio.Queue(maxsize=maxsize)
        sub = Subscription(
            event_type=event_type,
            handler=handler,
            queue=queue,
            task=asyncio.create_task(self._consumer_loop(queue, handler, name or event_type.__name__)),
            overflow=overflow,
            name=name or event_type.__name__,
        )
        self._subs.append(sub)
        return sub

    async def publish(self, event: Event) -> None:
        if self._closed:
            return
        for sub in self._subs:
            if sub.closed:
                continue
            if not isinstance(event, sub.event_type):
                continue
            if sub.overflow == BLOCK:
                await sub.queue.put(event)
            else:  # DROP_OLDEST
                if sub.queue.full():
                    try:
                        sub.queue.get_nowait()        # evict oldest
                    except asyncio.QueueEmpty:
                        pass
                try:
                    sub.queue.put_nowait(event)
                except asyncio.QueueFull:
                    # Race: another concurrent publisher refilled the queue.
                    # Drop this event silently — that's the policy.
                    logger.debug("drop_oldest: queue full after eviction sub=%s", sub.name)

    async def _consumer_loop(
        self,
        queue: asyncio.Queue[Event],
        handler: Handler[Event],
        name: str,
    ) -> None:
        while True:
            event = await queue.get()
            try:
                await handler(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("bus handler raised sub=%s event=%s", name, type(event).__name__)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for sub in self._subs:
            sub.closed = True
            sub.task.cancel()
        # Drain cancellations
        await asyncio.gather(*(self._await_cancelled(s.task) for s in self._subs))
        self._subs.clear()

    @staticmethod
    async def _await_cancelled(task: asyncio.Task[None]) -> None:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("bus consumer raised on shutdown")
