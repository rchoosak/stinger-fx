"""Pareto frontier extraction — domination, frontier ID, mixed directions."""

from __future__ import annotations

import pytest

from stinger_fx.backtest.pareto import (
    Objective,
    ParetoPoint,
    extract_pareto_frontier,
)


def _cell(name: str, **metrics: float) -> tuple[dict, dict]:
    """Build a (params, metrics) tuple; name → params for identifiability."""
    return ({"name": name}, metrics)


# --- Basic ------------------------------------------------------------------


def test_single_objective_max_picks_one_winner() -> None:
    """When the only objective is to maximise X, the cell with highest X is
    Pareto-optimal; all others are dominated."""
    cells = [
        _cell("a", net_pnl=10.0),
        _cell("b", net_pnl=20.0),
        _cell("c", net_pnl=15.0),
    ]
    result = extract_pareto_frontier(cells, [Objective("net_pnl", "max")])
    frontier = [p.params["name"] for p in result.points if p.is_pareto]
    assert frontier == ["b"]


def test_single_objective_min_picks_lowest() -> None:
    """min direction → smaller is better."""
    cells = [
        _cell("a", max_drawdown=10.0),
        _cell("b", max_drawdown=5.0),
        _cell("c", max_drawdown=15.0),
    ]
    result = extract_pareto_frontier(cells, [Objective("max_drawdown", "min")])
    frontier = [p.params["name"] for p in result.points if p.is_pareto]
    assert frontier == ["b"]


def test_two_objective_classic_frontier() -> None:
    """Maximise both Sharpe and net P&L. A cell that's best on one is Pareto.
    A cell that beats both is the only true winner."""
    cells = [
        _cell("a", sharpe=1.0, net_pnl=200.0),   # high P&L, low sharpe
        _cell("b", sharpe=2.0, net_pnl=100.0),   # high sharpe, low P&L
        _cell("c", sharpe=2.5, net_pnl=250.0),   # best on both → dominates all
        _cell("d", sharpe=1.5, net_pnl=150.0),   # dominated by c
    ]
    result = extract_pareto_frontier(
        cells, [Objective("sharpe", "max"), Objective("net_pnl", "max")]
    )
    frontier = {p.params["name"] for p in result.points if p.is_pareto}
    # Only c is Pareto-optimal (dominates all others)
    assert frontier == {"c"}


def test_two_objective_genuine_tradeoff() -> None:
    """Without a clear winner, multiple cells stay on the frontier."""
    cells = [
        _cell("a", sharpe=1.0, net_pnl=300.0),   # high P&L, low Sharpe
        _cell("b", sharpe=2.0, net_pnl=200.0),   # mid both
        _cell("c", sharpe=3.0, net_pnl=100.0),   # high Sharpe, low P&L
        _cell("d", sharpe=1.5, net_pnl=150.0),   # dominated by b on Sharpe AND
                                                 # by a on net_pnl... but b doesn't
                                                 # dominate d on net_pnl. Actually d is
                                                 # dominated by b since 2.0>=1.5 and 200>=150
    ]
    result = extract_pareto_frontier(
        cells, [Objective("sharpe", "max"), Objective("net_pnl", "max")]
    )
    frontier = {p.params["name"] for p in result.points if p.is_pareto}
    # a, b, c are on the frontier (each best on at least one trade-off);
    # d is strictly dominated by b
    assert frontier == {"a", "b", "c"}
    assert "d" not in frontier


def test_mixed_directions_max_sharpe_min_drawdown() -> None:
    """Realistic case: maximise Sharpe, minimise max drawdown."""
    cells = [
        _cell("a", sharpe=2.0, max_drawdown=10.0),
        _cell("b", sharpe=3.0, max_drawdown=20.0),
        _cell("c", sharpe=2.0, max_drawdown=20.0),  # dominated by a
        _cell("d", sharpe=1.5, max_drawdown=5.0),   # frontier: lowest DD
    ]
    result = extract_pareto_frontier(
        cells,
        [Objective("sharpe", "max"), Objective("max_drawdown", "min")],
    )
    frontier = {p.params["name"] for p in result.points if p.is_pareto}
    assert frontier == {"a", "b", "d"}


def test_empty_cells_returns_empty_result() -> None:
    result = extract_pareto_frontier([], [Objective("net_pnl")])
    assert result.points == []
    assert result.frontier == []


def test_no_objectives_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        extract_pareto_frontier([_cell("a", net_pnl=5.0)], [])


# --- Edge cases -------------------------------------------------------------


def test_identical_cells_both_pareto() -> None:
    """Two cells with identical metrics are both on the frontier (neither
    dominates strictly)."""
    cells = [
        _cell("a", net_pnl=10.0, sharpe=1.0),
        _cell("b", net_pnl=10.0, sharpe=1.0),
    ]
    result = extract_pareto_frontier(
        cells, [Objective("net_pnl"), Objective("sharpe")]
    )
    assert len([p for p in result.points if p.is_pareto]) == 2


def test_missing_metric_treated_as_worst() -> None:
    """A cell missing a metric is considered worst on that axis — never
    dominates anything, gets dominated if the others have a value."""
    cells = [
        _cell("a", net_pnl=10.0, sharpe=1.0),
        ({"name": "b"}, {"net_pnl": 5.0}),  # missing sharpe entirely
    ]
    result = extract_pareto_frontier(
        cells, [Objective("net_pnl"), Objective("sharpe")]
    )
    # a beats b on sharpe (b's sharpe = -inf); a is also higher on net_pnl
    # → a dominates b. b is NOT on the frontier.
    frontier = {p.params["name"] for p in result.points if p.is_pareto}
    assert frontier == {"a"}


def test_pareto_result_summary_serialises() -> None:
    """to_summary() returns JSON-friendly types."""
    import json

    cells = [_cell("a", x=1.0, y=2.0), _cell("b", x=2.0, y=1.0)]
    result = extract_pareto_frontier(
        cells, [Objective("x", "max"), Objective("y", "max")]
    )
    payload = result.to_summary()
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["frontier_size"] == 2
    assert decoded["total_points"] == 2
    assert len(decoded["objectives"]) == 2
