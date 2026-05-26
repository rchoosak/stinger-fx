"""Monte Carlo bootstrap simulator — percentile bands + equity envelope."""

from __future__ import annotations

import pytest

from stinger_fx.backtest.monte_carlo import (
    MonteCarloResult,
    _percentile,
    run_monte_carlo,
)


def test_returns_result_with_expected_shape() -> None:
    """A short bootstrap returns a MonteCarloResult with all the fields populated."""
    pnls = [10.0, -5.0, 8.0, -3.0, 12.0, 7.0, -2.0]
    result = run_monte_carlo(pnls, n_simulations=50, random_seed=42)
    assert isinstance(result, MonteCarloResult)
    assert result.n_simulations == 50
    assert result.n_trades == 7
    assert set(result.net_pnl.keys()) == {"p5", "p50", "p95", "mean"}
    assert set(result.max_drawdown.keys()) == {"p5", "p50", "p95", "mean"}
    assert set(result.sharpe.keys()) == {"p5", "p50", "p95", "mean"}
    assert len(result.equity_envelope_low) == 7
    assert len(result.equity_envelope_mid) == 7
    assert len(result.equity_envelope_high) == 7


def test_envelope_ordering_low_mid_high() -> None:
    """For every step, low ≤ mid ≤ high in the equity envelope."""
    pnls = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.5, -0.5, 1.5, -1.5]
    result = run_monte_carlo(pnls, n_simulations=100, random_seed=7)
    for lo, mid, hi in zip(
        result.equity_envelope_low,
        result.equity_envelope_mid,
        result.equity_envelope_high,
        strict=True,
    ):
        assert lo <= mid <= hi


def test_percentile_ordering_in_scalar_metrics() -> None:
    """p5 ≤ p50 ≤ p95 in every scalar metric distribution."""
    pnls = [5.0, -2.0, 7.0, -1.0, 3.0]
    result = run_monte_carlo(pnls, n_simulations=200, random_seed=0)
    for metric in (result.net_pnl, result.max_drawdown):
        assert metric["p5"] <= metric["p50"] <= metric["p95"]


def test_reproducible_with_seed() -> None:
    pnls = [10.0, -5.0, 7.0, -3.0]
    a = run_monte_carlo(pnls, n_simulations=100, random_seed=42)
    b = run_monte_carlo(pnls, n_simulations=100, random_seed=42)
    assert a.net_pnl == b.net_pnl
    assert a.equity_envelope_mid == b.equity_envelope_mid


def test_custom_percentiles() -> None:
    """Non-default percentile bands work and the result dict keys reflect them."""
    pnls = [3.0, -1.0, 2.0, -2.0, 4.0]
    result = run_monte_carlo(pnls, n_simulations=50, percentile_low=10, percentile_high=90, random_seed=1)
    assert "p10" in result.net_pnl
    assert "p90" in result.net_pnl
    assert "p5" not in result.net_pnl


def test_rejects_empty_trades() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        run_monte_carlo([], n_simulations=10)


def test_rejects_bad_n_simulations() -> None:
    with pytest.raises(ValueError, match="n_simulations"):
        run_monte_carlo([1.0], n_simulations=0)


def test_rejects_bad_percentile_bounds() -> None:
    with pytest.raises(ValueError, match="percentile"):
        run_monte_carlo([1.0, 2.0], n_simulations=10, percentile_low=95, percentile_high=5)


def test_to_json_serializable() -> None:
    """to_json() returns dicts/lists/primitives — no dataclasses or extras."""
    import json

    pnls = [1.0, -0.5, 0.5]
    result = run_monte_carlo(pnls, n_simulations=20, random_seed=0)
    payload = result.to_json()
    # Round-trip through JSON to prove it's serializable
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["n_simulations"] == 20
    assert "equity_envelope" in decoded


def test_percentile_helper_handles_extremes() -> None:
    """Direct test of the percentile interpolation helper for edge cases."""
    sorted_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(sorted_vals, 0) == 1.0
    assert _percentile(sorted_vals, 100) == 5.0
    assert _percentile(sorted_vals, 50) == 3.0
    assert _percentile([], 50) == 0.0
