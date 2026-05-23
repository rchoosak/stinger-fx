"""Positions table — refreshed on every broker poll."""

from __future__ import annotations

from textual.widgets import DataTable

from stinger_fx.domain import Position


class PositionsPanel(DataTable):
    BORDER_TITLE = "Positions"

    DEFAULT_CSS = """
    PositionsPanel {
        height: auto;
        max-height: 12;
        border: round $primary;
    }
    """

    def on_mount(self) -> None:
        self.cursor_type = "none"
        self.add_columns("ticket", "symbol", "side", "vol", "open", "sl", "tp", "P/L")

    def refresh_rows(self, positions: list[Position]) -> None:
        self.clear()
        for p in positions:
            pnl = f"{p.profit:+.2f}"
            color = "green" if p.profit > 0 else "red" if p.profit < 0 else ""
            self.add_row(
                str(p.ticket),
                p.symbol,
                p.side.value.upper(),
                f"{p.volume:.2f}",
                f"{p.open_price:.5f}",
                f"{p.sl:.5f}" if p.sl else "—",
                f"{p.tp:.5f}" if p.tp else "—",
                f"[{color}]{pnl}[/]" if color else pnl,
            )
