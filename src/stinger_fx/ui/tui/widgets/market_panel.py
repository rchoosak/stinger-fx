"""Market panel — last tick + tick/bar counters since the panel started."""

from __future__ import annotations

from rich.table import Table
from textual.reactive import reactive
from textual.widgets import Static


class MarketPanel(Static):
    """Last tick snapshot + cumulative flow counters."""

    DEFAULT_CSS = """
    MarketPanel {
        height: auto;
        padding: 0 1;
        border: round $primary;
    }
    """

    symbol: reactive[str | None] = reactive(None)
    bid: reactive[float | None] = reactive(None)
    ask: reactive[float | None] = reactive(None)
    last_bar_close: reactive[float | None] = reactive(None)
    last_bar_time: reactive[str] = reactive("—")
    last_bar_tf: reactive[str] = reactive("—")
    total_ticks: reactive[int] = reactive(0)
    total_bars: reactive[int] = reactive(0)

    def render(self) -> Table:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold cyan", justify="right")
        table.add_column()
        table.add_row("Symbol", self.symbol or "—")
        spread = "—"
        if self.bid is not None and self.ask is not None:
            spread = f"{(self.ask - self.bid) * 1e5:.1f} pip"
        table.add_row("Bid / Ask", _fmt_price(self.bid) + " / " + _fmt_price(self.ask))
        table.add_row("Spread", spread)
        table.add_row("Last bar tf", self.last_bar_tf)
        table.add_row("Last bar close", _fmt_price(self.last_bar_close))
        table.add_row("Last bar time", self.last_bar_time)
        table.add_row("Total ticks", str(self.total_ticks))
        table.add_row("Total bars", str(self.total_bars))
        return table


def _fmt_price(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.5f}"
