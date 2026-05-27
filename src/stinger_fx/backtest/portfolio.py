"""Portfolio aggregation — combine multiple backtest reports into one view.

A single strategy's backtest tells you whether *that* strategy works. A
real allocation typically runs several at once, and the questions
change: how do the strategies correlate? Which contributes most? Does
combining them improve risk-adjusted return, or do they bleed into
each other?

This module aggregates ``list[BacktestReport]`` into a ``PortfolioReport``:

  * **Combined equity curve** — sum the per-strategy curves on the
    union of timestamps (forward-fill missing samples per strategy
    so an idle strategy contributes a flat line, not zero)
  * **Combined metrics** — net P&L, max drawdown, Sharpe over the
    portfolio equity curve (not naive averages of per-strategy values)
  * **Cross-strategy correlation matrix** — pairwise Pearson on
    per-step P&L deltas (low correlation = better diversification)
  * **Contribution attribution** — each strategy's net P&L as a fraction
    of the portfolio total, with a directional flag (positive vs drag)

Use ``aggregate_portfolio([r1, r2, r3])`` from anywhere. The Web UI's
upcoming portfolio page consumes ``PortfolioReport.to_summary()``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import Any

from stinger_fx.backtest.reports import BacktestReport


@dataclass
class StrategyContribution:
    """How much one strategy contributed to the portfolio."""

    strategy_id: str
    run_id: str
    net_pnl: float
    share_pct: float            # net_pnl / portfolio_total_pnl × 100
    final_balance: float
    max_drawdown_pct: float
    sharpe: float
    trade_count: int


@dataclass
class PortfolioReport:
    """Aggregate of multiple BacktestReports."""

    id: str
    component_ids: list[str]
    initial_balance: float       # sum of per-strategy initial balances
    final_balance: float
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    contributions: list[StrategyContribution] = field(default_factory=list)
    correlation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)

    # --- Portfolio-level metrics --------------------------------------------

    @property
    def net_pnl(self) -> float:
        return self.final_balance - self.initial_balance

    @property
    def max_drawdown_pct(self) -> float:
        """Max drawdown as a percentage of the portfolio peak equity."""
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0][1]
        dd_pct = 0.0
        for _, eq in self.equity_curve:
            if eq > peak:
                peak = eq
            if peak > 0:
                drop = (peak - eq) / peak * 100
                if drop > dd_pct:
                    dd_pct = drop
        return dd_pct

    @property
    def sharpe(self) -> float:
        """Sharpe over the portfolio equity curve (per-step returns)."""
        if len(self.equity_curve) < 2:
            return 0.0
        returns: list[float] = []
        prev = self.equity_curve[0][1]
        for _, eq in self.equity_curve[1:]:
            if prev != 0:
                returns.append((eq - prev) / prev)
            prev = eq
        if not returns:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        sd = sqrt(var)
        if sd == 0:
            return 0.0
        return mean / sd * sqrt(len(returns))

    # --- Serialisation ------------------------------------------------------

    def to_summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "component_ids": list(self.component_ids),
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "net_pnl": round(self.net_pnl, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe": round(self.sharpe, 4),
            "contributions": [
                {
                    "strategy_id": c.strategy_id,
                    "run_id": c.run_id,
                    "net_pnl": round(c.net_pnl, 2),
                    "share_pct": round(c.share_pct, 2),
                    "final_balance": round(c.final_balance, 2),
                    "max_drawdown_pct": round(c.max_drawdown_pct, 2),
                    "sharpe": round(c.sharpe, 4),
                    "trade_count": c.trade_count,
                }
                for c in self.contributions
            ],
            "correlation_matrix": self.correlation_matrix,
            "equity_curve": [
                {"time": ts.isoformat(), "equity": round(eq, 2)}
                for ts, eq in self.equity_curve
            ],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_summary(), indent=2, default=str))


# --- Aggregation entry point ------------------------------------------------


def aggregate_portfolio(
    reports: list[BacktestReport],
    *,
    portfolio_id: str = "portfolio",
) -> PortfolioReport:
    """Combine N backtest reports into a single PortfolioReport.

    Equity-curve merge strategy: take the union of all timestamps across
    reports. For each timestamp, forward-fill each strategy's last known
    equity (initial_balance until the strategy first produces a sample),
    then sum across strategies for the combined equity. This gives the
    real "what would my account have looked like running all of these in
    parallel" curve.
    """
    if not reports:
        raise ValueError("aggregate_portfolio requires at least one report")

    initial_balance = sum(r.initial_balance for r in reports)
    combined_curve = _merge_equity_curves(reports)
    final_balance = (
        combined_curve[-1][1] if combined_curve else initial_balance
    )

    # Per-strategy contributions
    total_pnl = final_balance - initial_balance
    contributions: list[StrategyContribution] = []
    for r in reports:
        per_pnl = r.final_balance - r.initial_balance
        share = (per_pnl / total_pnl * 100) if total_pnl != 0 else 0.0
        contributions.append(
            StrategyContribution(
                strategy_id=r.strategy_id,
                run_id=r.run_id,
                net_pnl=per_pnl,
                share_pct=share,
                final_balance=r.final_balance,
                max_drawdown_pct=r.max_drawdown,
                sharpe=r.sharpe,
                trade_count=len(r.trades),
            )
        )

    # Correlation matrix on per-step P&L deltas
    correlation = _correlation_matrix(reports)

    return PortfolioReport(
        id=portfolio_id,
        component_ids=[r.run_id for r in reports],
        initial_balance=initial_balance,
        final_balance=final_balance,
        equity_curve=combined_curve,
        contributions=contributions,
        correlation_matrix=correlation,
    )


# --- Internals --------------------------------------------------------------


def _merge_equity_curves(
    reports: list[BacktestReport],
) -> list[tuple[datetime, float]]:
    """Forward-fill + sum equity across reports on the union of timestamps."""
    # Build a sorted union of all timestamps
    timestamps: set[datetime] = set()
    for r in reports:
        for ts, _ in r.equity_curve:
            timestamps.add(ts)
    if not timestamps:
        return []
    sorted_ts = sorted(timestamps)

    # Per-strategy index into its own sorted curve
    per_strategy: list[tuple[BacktestReport, list[tuple[datetime, float]], int]] = [
        (r, sorted(r.equity_curve, key=lambda x: x[0]), 0) for r in reports
    ]

    out: list[tuple[datetime, float]] = []
    last_equity = [r.initial_balance for r in reports]
    for ts in sorted_ts:
        # Advance each strategy's pointer up to ts; remember the last equity
        for i, (r, curve, idx) in enumerate(per_strategy):
            while idx < len(curve) and curve[idx][0] <= ts:
                last_equity[i] = curve[idx][1]
                idx += 1
            per_strategy[i] = (r, curve, idx)
        out.append((ts, sum(last_equity)))
    return out


def _correlation_matrix(
    reports: list[BacktestReport],
) -> dict[str, dict[str, float]]:
    """Pearson correlation matrix on per-step P&L deltas.

    Returns ``{strategy_id: {other_strategy_id: rho}}``. Strategies with
    identical run_ids would collide; we key on run_id to disambiguate.
    """
    if len(reports) < 2:
        return {r.run_id: {r.run_id: 1.0} for r in reports}

    # Build per-strategy delta series aligned on the union of timestamps
    timestamps: set[datetime] = set()
    for r in reports:
        for ts, _ in r.equity_curve:
            timestamps.add(ts)
    sorted_ts = sorted(timestamps)
    deltas: dict[str, list[float]] = {}
    for r in reports:
        # Forward-fill the curve onto sorted_ts, then compute deltas
        curve_map = dict(r.equity_curve)
        filled: list[float] = []
        last = r.initial_balance
        for ts in sorted_ts:
            if ts in curve_map:
                last = curve_map[ts]
            filled.append(last)
        d = [filled[i] - filled[i - 1] for i in range(1, len(filled))]
        deltas[r.run_id] = d

    out: dict[str, dict[str, float]] = {}
    keys = list(deltas.keys())
    for a in keys:
        out[a] = {}
        for b in keys:
            if a == b:
                out[a][b] = 1.0
            else:
                out[a][b] = _pearson(deltas[a], deltas[b])
    return out


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = sqrt(vx * vy)
    if denom == 0:
        return 0.0
    return num / denom
