"""Tests for the expanded indicator library (Bollinger, MACD, Stochastic, Donchian)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from stinger_fx.domain import Bar, Timeframe
from stinger_fx.strategies.indicators import (
    bollinger,
    donchian,
    macd,
    stochastic,
)


def _bar(i: int, *, h: float, l: float, c: float) -> Bar:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.M15,
        time=base + timedelta(minutes=15 * i),
        open=(h + l) / 2,
        high=h,
        low=l,
        close=c,
        is_closed=True,
    )


# --- Bollinger ---------------------------------------------------------------


def test_bollinger_none_when_short() -> None:
    assert bollinger([1.0] * 5, period=20) is None


def test_bollinger_flat_series_collapses_bands() -> None:
    bb = bollinger([10.0] * 30, period=20, stddev_mult=2.0)
    assert bb is not None
    assert bb.upper == pytest.approx(10.0)
    assert bb.middle == pytest.approx(10.0)
    assert bb.lower == pytest.approx(10.0)


def test_bollinger_widens_with_volatility() -> None:
    volatile = [10.0, 12.0] * 15
    bb = bollinger(volatile, period=20, stddev_mult=2.0)
    assert bb is not None
    assert bb.upper > bb.middle > bb.lower
    # mean of [10, 12, 10, 12, ...] is 11
    assert bb.middle == pytest.approx(11.0)


def test_bollinger_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        bollinger([1.0] * 5, period=1)
    with pytest.raises(ValueError):
        bollinger([1.0] * 5, period=3, stddev_mult=0)


# --- MACD --------------------------------------------------------------------


def test_macd_none_when_short() -> None:
    assert macd([1.0] * 10) is None  # default needs slow + signal = 35


def test_macd_zero_on_flat_series() -> None:
    result = macd([10.0] * 100, fast=12, slow=26, signal=9)
    assert result is not None
    assert result.macd == pytest.approx(0.0)
    assert result.signal == pytest.approx(0.0)
    assert result.histogram == pytest.approx(0.0)


def test_macd_positive_in_strong_uptrend() -> None:
    # Linear ramp — MACD stays positive (fast EMA > slow EMA), and the signal
    # converges to MACD, so the histogram trends toward 0. We only assert the
    # sign of MACD itself.
    rising = [float(i) for i in range(1, 101)]
    result = macd(rising, fast=12, slow=26, signal=9)
    assert result is not None
    assert result.macd > 0


def test_macd_histogram_positive_when_trend_accelerates() -> None:
    # Quadratic accelerating growth — MACD keeps increasing, signal lags →
    # histogram remains positive.
    accelerating = [float(i * i) for i in range(1, 101)]
    result = macd(accelerating, fast=12, slow=26, signal=9)
    assert result is not None
    assert result.macd > 0
    assert result.histogram > 0


def test_macd_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        macd([1.0] * 50, fast=26, slow=12)
    with pytest.raises(ValueError):
        macd([1.0] * 50, signal=0)


# --- Stochastic --------------------------------------------------------------


def test_stochastic_none_when_short() -> None:
    bars = [_bar(i, h=1.1, l=1.0, c=1.05) for i in range(5)]
    assert stochastic(bars, k_period=14, d_period=3) is None


def test_stochastic_at_top_of_range() -> None:
    # Close pinned at the high — %K should be 100
    bars = [_bar(i, h=1.1, l=1.0, c=1.1) for i in range(20)]
    result = stochastic(bars, k_period=14, d_period=3)
    assert result is not None
    assert result.k == pytest.approx(100.0)
    assert result.d == pytest.approx(100.0)


def test_stochastic_at_bottom_of_range() -> None:
    bars = [_bar(i, h=1.1, l=1.0, c=1.0) for i in range(20)]
    result = stochastic(bars, k_period=14, d_period=3)
    assert result is not None
    assert result.k == pytest.approx(0.0)


def test_stochastic_flat_range_returns_neutral() -> None:
    # All bars at the same level — degenerate denominator; should not crash
    bars = [_bar(i, h=1.05, l=1.05, c=1.05) for i in range(20)]
    result = stochastic(bars, k_period=14, d_period=3)
    assert result is not None
    assert result.k == pytest.approx(50.0)


# --- Donchian ----------------------------------------------------------------


def test_donchian_none_when_short() -> None:
    bars = [_bar(i, h=1.1, l=1.0, c=1.05) for i in range(5)]
    assert donchian(bars, period=20) is None


def test_donchian_picks_window_extremes() -> None:
    bars = [_bar(i, h=1.10 + 0.01 * i, l=1.00 - 0.01 * i, c=1.05) for i in range(20)]
    result = donchian(bars, period=20)
    assert result is not None
    # last bar h=1.29, l=0.81; first bar h=1.10, l=1.00 → max h=1.29, min l=0.81
    assert result.upper == pytest.approx(1.29)
    assert result.lower == pytest.approx(0.81)
    assert result.middle == pytest.approx((1.29 + 0.81) / 2)


def test_donchian_rejects_bad_period() -> None:
    with pytest.raises(ValueError):
        donchian([_bar(0, h=1, l=1, c=1)], period=0)
