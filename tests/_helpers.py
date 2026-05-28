"""Typed helpers shared across the test suite.

In particular ``collect_into`` replaces the ``lambda e: events.append(e)
or asyncio.sleep(0)`` idiom that mypy flagged with `func-returns-value`
(``list.append`` returns None, and the ``or`` trick only works because
of that). The helper returns a real async coroutine the bus subscribe
API will await.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable


def collect_into[E](bucket: list[E]) -> Callable[[E], Awaitable[None]]:
    """Return an async handler that appends each event to ``bucket``.

    Usage::

        events: list[FooEvent] = []
        sub = bus.subscribe(FooEvent, collect_into(events))

    Replaces the chained-append-or-sleep idiom that mypy can't narrow.
    """

    async def _handler(evt: E) -> None:
        bucket.append(evt)

    return _handler
