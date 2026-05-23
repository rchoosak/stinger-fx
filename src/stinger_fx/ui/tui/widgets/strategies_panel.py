"""Strategies table — id, name, state, symbol, timeframe."""

from __future__ import annotations

from textual.widgets import DataTable

from stinger_fx.ui.handle import StrategyState


class StrategiesPanel(DataTable):
    BORDER_TITLE = "Strategies"

    DEFAULT_CSS = """
    StrategiesPanel {
        height: auto;
        max-height: 10;
        border: round $primary;
    }
    """

    def on_mount(self) -> None:
        self.cursor_type = "row"
        self.add_columns("id", "name", "symbol", "timeframe", "state")

    def refresh_rows(self, strategies: list[StrategyState]) -> None:
        self.clear()
        for s in strategies:
            state_str = _state_color(s.state)
            self.add_row(s.id, s.name, s.symbol, s.timeframe, state_str)


def _state_color(state: str) -> str:
    if state == "started":
        return "[green]started[/]"
    if state == "paused":
        return "[yellow]paused[/]"
    if state == "quarantined":
        return "[red]quarantined[/]"
    return state
