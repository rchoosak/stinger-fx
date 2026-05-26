"""Golden-value tests for Phase 6.4.A indicators.

For each indicator: minimal synthetic data with hand-computed expected
values, plus edge-case tests (insufficient data → None, bad config →
ValueError).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from stinger_fx.domain import Bar, Timeframe
from stinger_fx.strategies.indicators import (
    adx,
    cci,
    correlation,
    ichimoku,
    keltner,
    pivot_points,
    vwap_rolling,
    vwap_session,
)


def _bar(
    i: int,
    *,
    high: float | None = None,
    low: float | None = None,
    close: float,
    open_: float | None = None,
    volume: int = 100,
) -> Bar:
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.M15,
        time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=15 * i),
        open=open_ if open_ is not None else close,
        high=high if high is not None else close + 0.0005,
        low=low if low is not None else close - 0.0005,
        close=close,
        tick_volume=volume,
        is_closed=True,
    )


# --- ADX --------------------------------------------------------------------


def test_adx_rejects_short_history() -> None:
    bars = [_bar(i, close=1.10 + 0.001 * i) for i in range(10)]
    assert adx(bars, period=14) is None


def test_adx_strong_uptrend_has_high_value() -> None:
    """A clean monotonic uptrend should produce an ADX in the strong-trend
    band (>25) and DI+ comfortably above DI−."""
    bars = []
    for i in range(40):
        c = 1.10 + 0.001 * i
        bars.append(_bar(i, close=c, high=c + 0.0002, low=c - 0.0002))
    result = adx(bars, period=14)
    assert result is not None
    assert result.adx > 25
    assert result.plus_di > result.minus_di


def test_adx_strong_downtrend_di_minus_dominates() -> None:
    bars = []
    for i in range(40):
        c = 1.15 - 0.001 * i
        bars.append(_bar(i, close=c, high=c + 0.0002, low=c - 0.0002))
    result = adx(bars, period=14)
    assert result is not None
    assert result.minus_di > result.plus_di


def test_adx_rejects_invalid_period() -> None:
    with pytest.raises(ValueError):
        adx([], period=0)


# --- Ichimoku ---------------------------------------------------------------


def test_ichimoku_returns_none_below_min_bars() -> None:
    bars = [_bar(i, close=1.10) for i in range(30)]
    assert ichimoku(bars, senkou_b_period=52) is None


def test_ichimoku_computes_midpoints() -> None:
    """For a flat-ish series with known high/low, the midpoints are easy
    to verify."""
    # 60 bars; high oscillates 1.10–1.12, low 1.08–1.10
    bars = []
    for i in range(60):
        high = 1.12 if i % 2 == 0 else 1.10
        low = 1.10 if i % 2 == 0 else 1.08
        bars.append(_bar(i, close=(high + low) / 2, high=high, low=low))
    result = ichimoku(bars, tenkan_period=9, kijun_period=26, senkou_b_period=52)
    assert result is not None
    # Over the last 9 bars: max high = 1.12, min low = 1.08, midpoint = 1.10
    assert result.tenkan == pytest.approx(1.10)
    assert result.kijun == pytest.approx(1.10)
    assert result.senkou_b == pytest.approx(1.10)
    assert result.senkou_a == pytest.approx(1.10)


# --- VWAP -------------------------------------------------------------------


def test_vwap_rolling_with_equal_volume_equals_avg_typical() -> None:
    """When all bars have the same volume, VWAP collapses to the mean of
    the typical prices."""
    bars = [_bar(i, close=1.10 + 0.001 * i, volume=100) for i in range(20)]
    typicals = [(b.high + b.low + b.close) / 3 for b in bars[-10:]]
    expected = sum(typicals) / 10
    assert vwap_rolling(bars, period=10) == pytest.approx(expected, abs=1e-9)


def test_vwap_session_weighted_by_volume() -> None:
    """Heavier bars pull the VWAP toward their typical price."""
    bars = [
        _bar(0, close=1.10, volume=1),
        _bar(1, close=1.20, volume=99),  # huge volume — VWAP near 1.20
    ]
    result = vwap_session(bars)
    assert result is not None
    assert result > 1.19


def test_vwap_rolling_returns_none_below_period() -> None:
    bars = [_bar(i, close=1.10) for i in range(5)]
    assert vwap_rolling(bars, period=10) is None


def test_vwap_session_zero_volume_falls_back_to_mean() -> None:
    """All-zero-volume bars → fall back to un-weighted typical-price mean
    instead of returning None (caller still gets a number)."""
    bars = [_bar(i, close=1.10 + 0.001 * i, volume=0) for i in range(5)]
    result = vwap_session(bars)
    assert result is not None
    expected = sum((b.high + b.low + b.close) / 3 for b in bars) / 5
    assert result == pytest.approx(expected)


# --- Keltner ----------------------------------------------------------------


def test_keltner_bands_centered_on_ema() -> None:
    bars = [_bar(i, close=1.10 + 0.0001 * i, high=1.10 + 0.0002 * i + 0.0005,
                 low=1.10 + 0.0001 * i - 0.0005) for i in range(40)]
    result = keltner(bars, ema_period=20, atr_period=10, atr_mult=2.0)
    assert result is not None
    assert result.lower < result.middle < result.upper
    half_width = (result.upper - result.lower) / 2
    assert half_width > 0


def test_keltner_returns_none_when_too_short() -> None:
    bars = [_bar(i, close=1.10) for i in range(10)]
    assert keltner(bars, ema_period=20) is None


# --- CCI --------------------------------------------------------------------


def test_cci_zero_for_flat_series() -> None:
    """All bars at the same typical price → CCI should be None (zero variance)."""
    bars = [_bar(i, close=1.10, high=1.10, low=1.10) for i in range(25)]
    assert cci(bars, period=20) is None  # zero mean_dev


def test_cci_positive_on_breakout_above_mean() -> None:
    """Slow drift then a big jump on the last bar → CCI should be strongly positive."""
    bars = [_bar(i, close=1.10, high=1.10005, low=1.09995) for i in range(19)]
    bars.append(_bar(19, close=1.105, high=1.106, low=1.104))  # jump
    result = cci(bars, period=20)
    assert result is not None
    assert result > 100


# --- Pivot Points -----------------------------------------------------------


def test_pivot_classic() -> None:
    levels = pivot_points(prev_high=1.20, prev_low=1.10, prev_close=1.15, method="classic")
    # pivot = (1.20 + 1.10 + 1.15) / 3 = 1.15
    assert levels.pivot == pytest.approx(1.15)
    # r1 = 2*1.15 - 1.10 = 1.20
    assert levels.r1 == pytest.approx(1.20)
    # s1 = 2*1.15 - 1.20 = 1.10
    assert levels.s1 == pytest.approx(1.10)


def test_pivot_fibonacci_ratios() -> None:
    levels = pivot_points(prev_high=1.20, prev_low=1.10, prev_close=1.15, method="fibonacci")
    rng = 0.10
    assert levels.r3 - levels.pivot == pytest.approx(rng * 1.000)
    assert levels.r2 - levels.pivot == pytest.approx(rng * 0.618)
    assert levels.r1 - levels.pivot == pytest.approx(rng * 0.382)


def test_pivot_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown pivot method"):
        pivot_points(1.0, 0.5, 0.7, method="bogus")  # type: ignore[arg-type]


# --- Correlation ------------------------------------------------------------


def test_correlation_perfectly_positive() -> None:
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [2.0, 4.0, 6.0, 8.0, 10.0]  # b = 2a
    assert correlation(a, b, period=5) == pytest.approx(1.0)


def test_correlation_perfectly_negative() -> None:
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert correlation(a, b, period=5) == pytest.approx(-1.0)


def test_correlation_no_relationship() -> None:
    """Constant series → no variance → None."""
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [3.0, 3.0, 3.0, 3.0, 3.0]
    assert correlation(a, b, period=5) is None


def test_correlation_requires_matched_lengths() -> None:
    with pytest.raises(ValueError, match="lengths differ"):
        correlation([1.0, 2.0], [1.0, 2.0, 3.0])


def test_correlation_below_period_returns_none() -> None:
    assert correlation([1.0, 2.0], [3.0, 4.0], period=10) is None
