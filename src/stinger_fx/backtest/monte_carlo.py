"""Monte Carlo simulation — bootstrap the trade list for confidence bands.

Backtest metrics from a single deterministic run only tell you what
happened against THIS exact sequence of trades. They don't say anything
about whether the strategy is lucky or robust — if the trades had landed
in a different order, would the equity curve still have looked OK?

Monte Carlo answers that by resampling the trade list with replacement
N times. For each resample we compute a fresh equity curve and the usual
risk metrics, then report percentile bands (5th, 50th, 95th by default).

Equity envelope: at each step k we collect the equity value across all
N simulations and take the same percentile bands, producing three
parallel curves the UI can plot as a shaded confidence band around the
median.

This is a non-parametric resampling — we don't assume any particular
distribution of returns. Just shuffle the past and see what could have
been.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MonteCarloResult:
    """Summary of a bootstrap simulation over a closed trade list."""

    n_simulations: int
    n_trades: int
    initial_balance: float
    percentile_low: int       # e.g. 5
    percentile_high: int      # e.g. 95

    # Per-metric distribution dicts: {"p5": .., "p50": .., "p95": .., "mean": ..}
    net_pnl: dict[str, float] = field(default_factory=dict)
    max_drawdown: dict[str, float] = field(default_factory=dict)
    sharpe: dict[str, float] = field(default_factory=dict)

    # Equity-curve envelope — three lists of length n_trades giving the
    # low/median/high percentile of equity at each step across simulations
    equity_envelope_low: list[float] = field(default_factory=list)
    equity_envelope_mid: list[float] = field(default_factory=list)
    equity_envelope_high: list[float] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "n_simulations": self.n_simulations,
            "n_trades": self.n_trades,
            "initial_balance": self.initial_balance,
            "percentile_low": self.percentile_low,
            "percentile_high": self.percentile_high,
            "net_pnl": self.net_pnl,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "equity_envelope": {
                "low": self.equity_envelope_low,
                "mid": self.equity_envelope_mid,
                "high": self.equity_envelope_high,
            },
        }


def run_monte_carlo(
    trade_pnls: list[float],
    *,
    n_simulations: int = 1000,
    initial_balance: float = 10_000.0,
    percentile_low: int = 5,
    percentile_high: int = 95,
    sharpe_periods_per_year: int = 252,
    random_seed: int | None = None,
) -> MonteCarloResult:
    """Bootstrap ``trade_pnls`` ``n_simulations`` times and return percentile bands.

    ``trade_pnls`` is the per-trade P&L from a backtest report; you can
    extract it via ``[t.pnl for t in report.trades]``. The Sharpe
    approximation uses per-trade P&L mean/std and annualises by
    ``sqrt(sharpe_periods_per_year)`` — for daily-ish FX backtests 252
    is the convention, swap it for the actual trade frequency if you want
    a more faithful number.
    """
    if not trade_pnls:
        raise ValueError("trade_pnls must be non-empty")
    if n_simulations <= 0:
        raise ValueError(f"n_simulations must be > 0, got {n_simulations}")
    if not (0 <= percentile_low < percentile_high <= 100):
        raise ValueError(
            f"need 0 <= percentile_low < percentile_high <= 100, "
            f"got low={percentile_low} high={percentile_high}"
        )

    rng = random.Random(random_seed)
    n_trades = len(trade_pnls)

    final_pnls: list[float] = []
    max_dds: list[float] = []
    sharpes: list[float] = []
    equity_curves: list[list[float]] = []

    for _ in range(n_simulations):
        sample = [rng.choice(trade_pnls) for _ in range(n_trades)]
        equity = initial_balance
        curve: list[float] = []
        peak = equity
        max_dd_pct = 0.0
        for p in sample:
            equity += p
            curve.append(equity)
            if equity > peak:
                peak = equity
            dd_pct = (peak - equity) / peak * 100 if peak > 0 else 0.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        final_pnls.append(curve[-1] - initial_balance)
        max_dds.append(max_dd_pct)
        sharpes.append(_sharpe(sample, sharpe_periods_per_year))
        equity_curves.append(curve)

    # Percentile bands per scalar metric
    net_pnl = _distribution(final_pnls, percentile_low, percentile_high)
    max_drawdown = _distribution(max_dds, percentile_low, percentile_high)
    sharpe = _distribution(sharpes, percentile_low, percentile_high)

    # Envelope: at each step, take percentiles across simulations
    envelope_low: list[float] = []
    envelope_mid: list[float] = []
    envelope_high: list[float] = []
    for step in range(n_trades):
        col = [c[step] for c in equity_curves]
        col_sorted = sorted(col)
        envelope_low.append(_percentile(col_sorted, percentile_low))
        envelope_mid.append(_percentile(col_sorted, 50))
        envelope_high.append(_percentile(col_sorted, percentile_high))

    return MonteCarloResult(
        n_simulations=n_simulations,
        n_trades=n_trades,
        initial_balance=initial_balance,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        net_pnl=net_pnl,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
        equity_envelope_low=envelope_low,
        equity_envelope_mid=envelope_mid,
        equity_envelope_high=envelope_high,
    )


# --- Helpers ----------------------------------------------------------------


def _distribution(values: list[float], low_p: int, high_p: int) -> dict[str, float]:
    """Return p_low / p_50 / p_high / mean summary."""
    values_sorted = sorted(values)
    return {
        f"p{low_p}": _percentile(values_sorted, low_p),
        "p50": _percentile(values_sorted, 50),
        f"p{high_p}": _percentile(values_sorted, high_p),
        "mean": sum(values) / len(values) if values else 0.0,
    }


def _percentile(sorted_values: list[float], p: int) -> float:
    """Pick the p-th percentile from an already-sorted list (linear interp)."""
    if not sorted_values:
        return 0.0
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    n = len(sorted_values)
    rank = (p / 100) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _sharpe(pnls: list[float], periods_per_year: int) -> float:
    """Per-trade Sharpe approximation; 0 when fewer than 2 trades or zero std."""
    n = len(pnls)
    if n < 2:
        return 0.0
    mean = sum(pnls) / n
    var = sum((p - mean) ** 2 for p in pnls) / (n - 1)
    std = math.sqrt(var)
    if std <= 0:
        return 0.0
    return (mean / std) * math.sqrt(periods_per_year)
