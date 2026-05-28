"""Filesystem watcher → triggers config reload on YAML changes.

Uses `watchdog` for OS-native events. Editors often write a file 2–3 times in
quick succession (rename, write, sync) so we debounce by 300ms.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger("stinger.config.watcher")

DEBOUNCE_SECONDS = 0.3


class _Handler(FileSystemEventHandler):
    """Bridges sync watchdog callbacks into the asyncio loop."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        callback: Callable[[], None],
        tracked: set[str],
    ) -> None:
        self._loop = loop
        self._callback = callback
        self._tracked = tracked  # filenames we care about, e.g. {"app.yaml", "strategies.yaml"}

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        # watchdog types src_path as bytes | str depending on the OS; coerce.
        src = event.src_path
        if isinstance(src, bytes):
            src = src.decode()
        name = Path(src).name
        if name not in self._tracked:
            return
        self._loop.call_soon_threadsafe(self._callback)


class ConfigWatcher:
    """Watches `config_dir` and invokes `on_change()` after debounce."""

    def __init__(
        self,
        config_dir: Path,
        on_change: Callable[[], Awaitable[None]],
        *,
        tracked: set[str] | None = None,
    ) -> None:
        self._dir = config_dir
        self._on_change = on_change
        self._tracked = tracked or {"app.yaml", "strategies.yaml", "backtest.yaml"}
        # Observer is a factory function in watchdog (not a class), so mypy
        # can't use it as a type; Any keeps the runtime behaviour intact.
        self._observer: Any = None
        self._debounce_task: asyncio.Task[None] | None = None
        self._pending = False

    async def start(self) -> None:
        if self._observer is not None:
            return
        loop = asyncio.get_running_loop()
        handler = _Handler(loop, self._schedule, self._tracked)
        self._observer = Observer()
        self._observer.schedule(handler, str(self._dir), recursive=False)
        self._observer.start()
        logger.info("config watcher started dir=%s tracked=%s", self._dir, sorted(self._tracked))

    async def stop(self) -> None:
        if self._observer is None:
            return
        self._observer.stop()
        self._observer.join(timeout=2)
        self._observer = None
        if self._debounce_task is not None:
            self._debounce_task.cancel()
            try:
                await self._debounce_task
            except asyncio.CancelledError:
                pass
            self._debounce_task = None

    def _schedule(self) -> None:
        """Called from the watchdog thread via call_soon_threadsafe."""
        self._pending = True
        if self._debounce_task is None or self._debounce_task.done():
            self._debounce_task = asyncio.create_task(self._debounce_and_fire())

    async def _debounce_and_fire(self) -> None:
        # Coalesce a burst of events into one callback
        while True:
            self._pending = False
            await asyncio.sleep(DEBOUNCE_SECONDS)
            if not self._pending:
                break
        try:
            await self._on_change()
        except Exception:
            logger.exception("on_change callback raised")
