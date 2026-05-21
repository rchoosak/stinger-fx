"""Periodic task scheduler used for heartbeats and account snapshots.

This is intentionally minimal — bar-close detection lives in the bar aggregator,
not here. The scheduler only owns wall-clock-driven periodic work.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

logger = logging.getLogger("stinger.engine.scheduler")


@dataclass
class _Job:
    name: str
    interval: float
    handler: Callable[[], Awaitable[None]]
    task: asyncio.Task[None] | None = field(default=None, repr=False)


class Scheduler:
    def __init__(self) -> None:
        self._jobs: list[_Job] = []
        self._running = False

    def every(self, seconds: float, handler: Callable[[], Awaitable[None]], *, name: str = "") -> None:
        if seconds <= 0:
            raise ValueError("interval must be positive")
        self._jobs.append(_Job(name=name or handler.__name__, interval=seconds, handler=handler))

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        for job in self._jobs:
            job.task = asyncio.create_task(self._run_job(job))

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for job in self._jobs:
            if job.task is not None:
                job.task.cancel()
        await asyncio.gather(
            *(self._await_cancelled(j.task) for j in self._jobs if j.task is not None),
            return_exceptions=True,
        )

    async def _run_job(self, job: _Job) -> None:
        while self._running:
            try:
                await job.handler()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("scheduled job raised name=%s", job.name)
            try:
                await asyncio.sleep(job.interval)
            except asyncio.CancelledError:
                raise

    @staticmethod
    async def _await_cancelled(task: asyncio.Task[None]) -> None:
        try:
            await task
        except asyncio.CancelledError:
            pass
