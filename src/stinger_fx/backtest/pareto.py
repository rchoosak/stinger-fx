"""Pareto frontier extraction for multi-objective optimisation.

Single-metric ranking ("best Sharpe", "best net P&L") is convenient but
misleading: a strategy with Sharpe 3.0 and 50% max drawdown isn't
obviously better than one with Sharpe 2.5 and 10% drawdown. Multi-objective
optimisation answers this by reporting the *Pareto frontier*: the set of
parameter combinations where no other combination dominates them on
every metric simultaneously.

Definitions:

  * A cell ``X`` **dominates** cell ``Y`` if ``X`` is at least as good
    as ``Y`` on every metric and strictly better on at least one.
  * A cell is **Pareto-optimal** (on the frontier) if no other cell
    dominates it.

This module operates on a list of ``(params, metrics)`` tuples and an
``Objective`` spec for each metric: name + direction (maximise or
minimise). It returns the same list annotated with an ``is_pareto`` flag
so the caller can render both the full result set and the frontier
without re-sorting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# Direction "max" → larger is better (net P&L, Sharpe, win rate).
# Direction "min" → smaller is better (max drawdown).
Direction = Literal["max", "min"]


@dataclass
class Objective:
    """One axis of a multi-objective ranking."""

    metric: str
    direction: Direction = "max"


@dataclass
class ParetoPoint:
    """A cell tagged with whether it lies on the frontier."""

    params: dict[str, Any]
    metrics: dict[str, float]
    is_pareto: bool = False


@dataclass
class ParetoResult:
    """All cells + just the frontier subset, plus the objectives used."""

    objectives: list[Objective] = field(default_factory=list)
    points: list[ParetoPoint] = field(default_factory=list)

    @property
    def frontier(self) -> list[ParetoPoint]:
        return [p for p in self.points if p.is_pareto]

    def to_summary(self) -> dict[str, Any]:
        return {
            "objectives": [
                {"metric": o.metric, "direction": o.direction}
                for o in self.objectives
            ],
            "points": [
                {
                    "params": p.params,
                    "metrics": p.metrics,
                    "is_pareto": p.is_pareto,
                }
                for p in self.points
            ],
            "frontier_size": len(self.frontier),
            "total_points": len(self.points),
        }


def extract_pareto_frontier(
    cells: list[tuple[dict[str, Any], dict[str, float]]],
    objectives: list[Objective],
) -> ParetoResult:
    """Return all cells with ``is_pareto`` annotation.

    ``cells`` is a list of ``(params, metrics)`` tuples. ``objectives``
    declares which metrics to consider and which direction is "better"
    for each. A cell is on the frontier iff no other cell dominates it
    across all objectives.

    Complexity is O(N²) in the number of cells — fine for typical sweep
    sizes (≤10k cells). For very large sweeps a sort-based algorithm
    (Kung's) would be faster but adds complexity not justified here.
    """
    if not objectives:
        raise ValueError("at least one Objective required")
    if not cells:
        return ParetoResult(objectives=objectives, points=[])

    points = [
        ParetoPoint(params=dict(params), metrics=dict(metrics))
        for params, metrics in cells
    ]

    for i, point in enumerate(points):
        dominated = False
        for j, other in enumerate(points):
            if i == j:
                continue
            if _dominates(other.metrics, point.metrics, objectives):
                dominated = True
                break
        point.is_pareto = not dominated

    return ParetoResult(objectives=objectives, points=points)


def _dominates(
    a: dict[str, float],
    b: dict[str, float],
    objectives: list[Objective],
) -> bool:
    """Return True iff a dominates b: at least as good on every objective,
    strictly better on at least one. Missing metrics are treated as the
    worst possible value for the direction (so missing data never causes
    a false positive)."""
    strictly_better = False
    for obj in objectives:
        a_val = _coerce(a.get(obj.metric), obj.direction)
        b_val = _coerce(b.get(obj.metric), obj.direction)
        if obj.direction == "max":
            if a_val < b_val:
                return False  # a worse on this axis → not dominating
            if a_val > b_val:
                strictly_better = True
        else:  # min
            if a_val > b_val:
                return False
            if a_val < b_val:
                strictly_better = True
    return strictly_better


def _coerce(value: float | None, direction: Direction) -> float:
    """Coerce missing/NaN to the worst possible value for the direction."""
    if value is None:
        return float("-inf") if direction == "max" else float("inf")
    if isinstance(value, float) and value != value:  # NaN check
        return float("-inf") if direction == "max" else float("inf")
    return float(value)
