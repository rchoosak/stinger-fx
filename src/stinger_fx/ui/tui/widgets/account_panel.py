"""Account panel — balance, equity, drawdown, kill-switch state."""

from __future__ import annotations

from rich.table import Table
from textual.reactive import reactive
from textual.widgets import Static


class AccountPanel(Static):
    """Live account snapshot. Updated from AccountSnapshotEvent."""

    DEFAULT_CSS = """
    AccountPanel {
        height: auto;
        padding: 0 1;
        border: round $primary;
    }
    """

    balance: reactive[float | None] = reactive(None)
    equity: reactive[float | None] = reactive(None)
    profit: reactive[float | None] = reactive(None)
    peak_equity: reactive[float | None] = reactive(None)
    drawdown_pct: reactive[float] = reactive(0.0)
    kill_switch: reactive[bool] = reactive(False)
    broker: reactive[str] = reactive("—")
    currency: reactive[str] = reactive("USD")

    def render(self) -> Table:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold cyan", justify="right")
        table.add_column()
        table.add_row("Broker", f"{self.broker}")
        table.add_row("Balance", _fmt_money(self.balance, self.currency))
        table.add_row("Equity", _fmt_money(self.equity, self.currency))
        profit_str = _fmt_money(self.profit, self.currency)
        if self.profit is not None and self.profit < 0:
            profit_str = f"[red]{profit_str}[/]"
        elif self.profit is not None and self.profit > 0:
            profit_str = f"[green]{profit_str}[/]"
        table.add_row("Floating P/L", profit_str)
        table.add_row("Peak equity", _fmt_money(self.peak_equity, self.currency))
        dd_color = "red" if self.drawdown_pct >= 10 else "yellow" if self.drawdown_pct >= 5 else "green"
        table.add_row("Drawdown", f"[{dd_color}]{self.drawdown_pct:.2f}%[/]")
        ks = "[red on white]TRIPPED[/]" if self.kill_switch else "[green]armed[/]"
        table.add_row("Kill switch", ks)
        return table


def _fmt_money(v: float | None, ccy: str) -> str:
    if v is None:
        return "—"
    return f"{v:,.2f} {ccy}"
