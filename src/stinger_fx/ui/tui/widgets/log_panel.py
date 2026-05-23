"""Scrolling event log — fills, rejections, decisions, strategy state changes."""

from __future__ import annotations

from collections import deque
from datetime import datetime

from rich.text import Text
from textual.widgets import RichLog


class LogPanel(RichLog):
    BORDER_TITLE = "Events"

    DEFAULT_CSS = """
    LogPanel {
        height: 1fr;
        border: round $primary;
    }
    """

    MAX_LINES = 500

    def on_mount(self) -> None:
        self._buf: deque[Text] = deque(maxlen=self.MAX_LINES)
        self.markup = True
        self.wrap = False

    def push(self, *, level: str, message: str, ts: datetime | None = None) -> None:
        ts = ts or datetime.now()
        prefix = ts.strftime("%H:%M:%S")
        color = {"info": "cyan", "warning": "yellow", "error": "red"}.get(level.lower(), "white")
        line = Text.from_markup(f"[dim]{prefix}[/] [{color}]{level.upper():7}[/] {message}")
        self._buf.append(line)
        self.write(line)
