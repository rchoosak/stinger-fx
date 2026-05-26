"""Cointegration helpers for pairs trading and statistical arbitrage.

Two series are *cointegrated* when their individual price levels are
non-stationary (random walks) but a specific linear combination of them
is stationary. That stationary spread is what pairs-trading strategies
exploit: when the spread deviates far from its mean, it's expected to
revert.

This module provides:

  * :func:`ols_regression`        — bare OLS slope + intercept (stdlib only)
  * :func:`rolling_hedge_ratio`   — recent-window slope, for dynamic pairs
  * :func:`engle_granger_test`    — full two-step test that returns the
                                    hedge ratio, residuals, and an ADF
                                    p-value when ``statsmodels`` is
                                    installed; falls back to a heuristic
                                    stationarity check otherwise

Why optional statsmodels? The full ADF test (with lag selection by AIC)
is heavy. For users who just want pairs-trading basics, the heuristic
is good enough. Install ``stinger-fx[pairs]`` extra to get the proper
p-value.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from typing import NamedTuple


class CointegrationResult(NamedTuple):
    """Outcome of an Engle-Granger two-step test."""

    hedge_ratio: float          # slope from regress(a, b)
    intercept: float
    residuals: list[float]      # a - (intercept + hedge_ratio * b)
    spread_mean: float
    spread_std: float
    adf_pvalue: float | None    # None if statsmodels not installed
    is_stationary: bool         # True if cointegration is plausible


# --- OLS --------------------------------------------------------------------


def ols_regression(
    y: Sequence[float], x: Sequence[float]
) -> tuple[float, float]:
    """Ordinary least squares: ``y = slope * x + intercept``.

    Returns ``(slope, intercept)``.  Raises ``ValueError`` when the
    series have zero variance in x (can't solve) or mismatched lengths.
    """
    if len(y) != len(x):
        raise ValueError(f"length mismatch: y={len(y)} x={len(x)}")
    if len(x) < 2:
        raise ValueError("need at least 2 samples")
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=True))
    den = sum((xi - mean_x) ** 2 for xi in x)
    if den == 0:
        raise ValueError("zero variance in x — can't solve OLS")
    slope = num / den
    intercept = mean_y - slope * mean_x
    return slope, intercept


def rolling_hedge_ratio(
    series_a: Sequence[float], series_b: Sequence[float], window: int
) -> float | None:
    """Hedge ratio over the trailing ``window`` samples (None if too short)."""
    if window <= 1:
        raise ValueError(f"window must be > 1, got {window}")
    if len(series_a) != len(series_b):
        raise ValueError(
            f"series lengths differ: a={len(series_a)} b={len(series_b)}"
        )
    if len(series_a) < window:
        return None
    a_w = list(series_a[-window:])
    b_w = list(series_b[-window:])
    try:
        slope, _ = ols_regression(a_w, b_w)
    except ValueError:
        return None
    return slope


# --- Engle-Granger two-step -------------------------------------------------


def engle_granger_test(
    series_a: Sequence[float],
    series_b: Sequence[float],
    *,
    min_samples: int = 30,
) -> CointegrationResult:
    """Two-step Engle-Granger test for cointegration.

    Step 1: OLS regression ``a ~ b`` → residuals
    Step 2: Test residuals for stationarity — proper ADF when statsmodels
            is installed, heuristic mean+variance check otherwise.

    The ``is_stationary`` flag is the headline result: True suggests
    cointegration; False suggests the pair shouldn't be traded as a
    mean-reverting spread.
    """
    if len(series_a) != len(series_b):
        raise ValueError(
            f"series lengths differ: a={len(series_a)} b={len(series_b)}"
        )
    if len(series_a) < min_samples:
        raise ValueError(
            f"need at least {min_samples} samples for cointegration test, "
            f"got {len(series_a)}"
        )

    a_list = list(series_a)
    b_list = list(series_b)
    slope, intercept = ols_regression(a_list, b_list)
    residuals = [a - (intercept + slope * b) for a, b in zip(a_list, b_list, strict=True)]

    n = len(residuals)
    mean = sum(residuals) / n
    var = sum((r - mean) ** 2 for r in residuals) / n
    std = sqrt(var) if var > 0 else 0.0

    # Try statsmodels for a proper ADF p-value
    adf_p: float | None = None
    try:
        from statsmodels.tsa.stattools import adfuller  # type: ignore[import-untyped]

        adf_result = adfuller(residuals, autolag="AIC")
        adf_p = float(adf_result[1])
        is_stationary = adf_p < 0.05
    except ImportError:
        is_stationary = _heuristic_stationarity(residuals)

    return CointegrationResult(
        hedge_ratio=slope,
        intercept=intercept,
        residuals=residuals,
        spread_mean=mean,
        spread_std=std,
        adf_pvalue=adf_p,
        is_stationary=is_stationary,
    )


# --- Heuristic stationarity (no statsmodels) --------------------------------


def _heuristic_stationarity(residuals: list[float]) -> bool:
    """Rough cointegration check used when statsmodels isn't installed.

    Two cheap checks:
      1. The series mean is small relative to its standard deviation
         (mean-reverting around zero, not drifting)
      2. First-half variance and second-half variance are roughly equal
         (no obvious heteroskedasticity / trend in variance)
    """
    n = len(residuals)
    if n < 10:
        return False
    mean = sum(residuals) / n
    var = sum((r - mean) ** 2 for r in residuals) / n
    std = sqrt(var) if var > 0 else 0.0
    if std == 0:
        return False
    # Check 1: mean should be small relative to std
    if abs(mean) > 0.5 * std:
        return False
    # Check 2: first-half vs second-half variance ratio
    half = n // 2
    first = residuals[:half]
    second = residuals[half:]
    m1 = sum(first) / len(first)
    m2 = sum(second) / len(second)
    v1 = sum((r - m1) ** 2 for r in first) / len(first)
    v2 = sum((r - m2) ** 2 for r in second) / len(second)
    if v1 == 0 or v2 == 0:
        return False
    ratio = max(v1, v2) / min(v1, v2)
    return ratio < 3.0


# --- Z-score helper ---------------------------------------------------------


def spread_zscore(
    spread_series: Sequence[float], window: int = 20
) -> float | None:
    """Z-score of the most recent spread value over a rolling window.

    A z-score above +2 (or below -2) is the conventional entry threshold
    for mean-reversion strategies: the spread is two standard deviations
    away from its recent mean, expected to revert.

    Returns ``None`` when the series has fewer than ``window`` samples
    or the rolling standard deviation is zero.
    """
    if window <= 1:
        raise ValueError(f"window must be > 1, got {window}")
    if len(spread_series) < window:
        return None
    w = spread_series[-window:]
    mean = sum(w) / window
    var = sum((s - mean) ** 2 for s in w) / window
    std = sqrt(var) if var > 0 else 0.0
    if std == 0:
        return None
    return (spread_series[-1] - mean) / std
