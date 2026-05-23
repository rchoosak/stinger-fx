"""Backtest metrics + equity-curve output."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import sqrt
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class TradeRecord:
    open_ts: datetime
    close_ts: datetime
    side: str
    open_price: float
    close_price: float
    volume: float
    pnl: float


@dataclass
class BacktestReport:
    run_id: str
    strategy_id: str
    started_at: datetime
    finished_at: datetime
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    initial_balance: float = 10_000.0
    final_balance: float = 0.0

    # --- Metrics ------------------------------------------------------------

    @property
    def net_pnl(self) -> float:
        return self.final_balance - self.initial_balance

    @property
    def gross_profit(self) -> float:
        return sum(t.pnl for t in self.trades if t.pnl > 0)

    @property
    def gross_loss(self) -> float:
        return sum(t.pnl for t in self.trades if t.pnl < 0)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    @property
    def profit_factor(self) -> float:
        loss = -self.gross_loss
        if loss == 0:
            return float("inf") if self.gross_profit > 0 else 0.0
        return self.gross_profit / loss

    @property
    def expectancy(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.pnl for t in self.trades) / len(self.trades)

    @property
    def max_drawdown(self) -> float:
        """Peak-to-trough drawdown on the equity curve (in account currency)."""
        peak = self.initial_balance
        max_dd = 0.0
        for _, eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def sharpe(self) -> float:
        """Period-returns Sharpe assuming each step is one bar. Risk-free = 0."""
        if len(self.equity_curve) < 2:
            return 0.0
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev = self.equity_curve[i - 1][1]
            cur = self.equity_curve[i][1]
            if prev == 0:
                continue
            returns.append((cur - prev) / prev)
        if not returns:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        sd = sqrt(var)
        if sd == 0:
            return 0.0
        return mean / sd * sqrt(len(returns))

    def to_metrics_dict(self) -> dict:
        return {
            "trades": len(self.trades),
            "net_pnl": round(self.net_pnl, 2),
            "gross_profit": round(self.gross_profit, 2),
            "gross_loss": round(self.gross_loss, 2),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4) if self.profit_factor != float("inf") else None,
            "expectancy": round(self.expectancy, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "sharpe": round(self.sharpe, 4),
            "initial_balance": self.initial_balance,
            "final_balance": round(self.final_balance, 2),
        }

    def write_equity_curve(self, path: Path) -> None:
        if not self.equity_curve:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tbl = pa.Table.from_pylist(
            [{"time": ts, "equity": eq} for ts, eq in self.equity_curve],
            schema=pa.schema([("time", pa.timestamp("ns", tz="UTC")), ("equity", pa.float64())]),
        )
        pq.write_table(tbl, path, compression="zstd")

    def trades_to_jsonable(self) -> list[dict]:
        """List-of-dicts representation suitable for JSON serialization.

        Used by the web trade-replay view so the browser can plot entry/exit
        markers on top of the equity curve.
        """
        return [
            {
                "open_ts": t.open_ts.isoformat(),
                "close_ts": t.close_ts.isoformat(),
                "side": t.side,
                "open_price": t.open_price,
                "close_price": t.close_price,
                "volume": t.volume,
                "pnl": round(t.pnl, 2),
            }
            for t in self.trades
        ]
