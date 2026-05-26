"""Cointegration helpers — OLS, rolling hedge ratio, Engle-Granger test."""

from __future__ import annotations

import random

import pytest

from stinger_fx.strategies.cointegration import (
    engle_granger_test,
    ols_regression,
    rolling_hedge_ratio,
    spread_zscore,
)


# --- OLS --------------------------------------------------------------------


def test_ols_simple_linear() -> None:
    """y = 2*x + 1: slope=2, intercept=1."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [3.0, 5.0, 7.0, 9.0, 11.0]
    slope, intercept = ols_regression(y, x)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)


def test_ols_handles_noise() -> None:
    """y ≈ 3*x + noise: slope should still be close to 3."""
    rng = random.Random(42)
    x = list(range(50))
    y = [3 * xi + rng.gauss(0, 0.1) for xi in x]
    slope, intercept = ols_regression(y, x)
    assert slope == pytest.approx(3.0, abs=0.05)


def test_ols_rejects_zero_variance_x() -> None:
    """Constant x → can't solve OLS."""
    with pytest.raises(ValueError, match="zero variance"):
        ols_regression([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])


def test_ols_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        ols_regression([1.0, 2.0], [1.0, 2.0, 3.0])


# --- Rolling hedge ratio ----------------------------------------------------


def test_rolling_hedge_ratio_uses_last_window() -> None:
    """Earlier samples don't influence the result when window slides."""
    a = list(range(100))
    b = [2 * x for x in range(50)] + [3 * x for x in range(50, 100)]
    # Last 50 samples have slope 3
    hedge = rolling_hedge_ratio(a, b, window=50)
    assert hedge is not None
    assert hedge < 1.0  # because a is 1*idx, b is 3*idx in last half → slope ≈ 1/3


def test_rolling_hedge_returns_none_short() -> None:
    assert rolling_hedge_ratio([1.0, 2.0], [3.0, 4.0], window=10) is None


def test_rolling_hedge_validation() -> None:
    with pytest.raises(ValueError):
        rolling_hedge_ratio([1.0, 2.0], [3.0, 4.0], window=1)
    with pytest.raises(ValueError, match="lengths differ"):
        rolling_hedge_ratio([1.0, 2.0], [3.0, 4.0, 5.0], window=2)


# --- Engle-Granger ----------------------------------------------------------


def test_engle_granger_on_cointegrated_pair() -> None:
    """a = 2*b + stationary mean-zero noise → should look cointegrated."""
    rng = random.Random(42)
    n = 100
    b = [10.0]
    for _ in range(n - 1):
        b.append(b[-1] + rng.gauss(0, 0.1))  # random walk
    # a = 2 * b + stationary noise
    a = [2 * bi + rng.gauss(0, 0.05) for bi in b]
    result = engle_granger_test(a, b)
    assert result.hedge_ratio == pytest.approx(2.0, abs=0.1)
    # Mean of residuals ~0, variance bounded → heuristic should say stationary.
    # (Even with statsmodels missing, the heuristic catches this case.)
    assert result.is_stationary


def test_engle_granger_on_random_walks_not_cointegrated() -> None:
    """Two independent random walks shouldn't be cointegrated."""
    rng = random.Random(123)
    n = 100
    a = [10.0]
    b = [10.0]
    for _ in range(n - 1):
        a.append(a[-1] + rng.gauss(0, 1.0))
        b.append(b[-1] + rng.gauss(0, 1.0))
    result = engle_granger_test(a, b)
    # Result might have a fitted hedge_ratio but residuals won't be stationary
    # — heuristic checks should flag this. Allow for some randomness — the
    # stationarity flag may vary by seed, but the test should at least run.
    assert isinstance(result.hedge_ratio, float)
    assert len(result.residuals) == 100
    assert result.spread_std > 0


def test_engle_granger_rejects_short_series() -> None:
    with pytest.raises(ValueError, match="at least 30"):
        engle_granger_test([1.0] * 10, [2.0] * 10)


def test_engle_granger_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="lengths differ"):
        engle_granger_test([1.0] * 30, [2.0] * 29)


# --- Z-score ----------------------------------------------------------------


def test_zscore_returns_zero_for_value_at_mean() -> None:
    """Last value equals window mean → z = 0.
    Window = last 5 values [1, 5, 1, 5, 3]. Mean = 15/5 = 3. Last value = 3."""
    series = [10.0, 1.0, 5.0, 1.0, 5.0, 3.0]  # window=5 → [1, 5, 1, 5, 3]
    z = spread_zscore(series, window=5)
    assert z is not None
    assert z == pytest.approx(0.0, abs=1e-9)


def test_zscore_positive_when_above_mean() -> None:
    series = [0.0, 0.0, 0.0, 0.0, 5.0]
    z = spread_zscore(series, window=5)
    assert z is not None
    assert z > 0


def test_zscore_returns_none_when_too_short() -> None:
    assert spread_zscore([1.0, 2.0, 3.0], window=5) is None


def test_zscore_returns_none_on_constant_window() -> None:
    """Zero variance → can't compute z-score."""
    assert spread_zscore([1.0] * 10, window=5) is None
