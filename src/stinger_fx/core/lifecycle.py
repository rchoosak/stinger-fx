"""Component lifecycle protocol — anything the engine starts/stops implements it."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Lifecycle(Protocol):
    """Anything that has an async start/stop pair the engine can orchestrate."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
