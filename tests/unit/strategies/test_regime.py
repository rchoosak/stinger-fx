"""Regime detection filters — Trending / Ranging / HighVol / LowVol / Composite."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from stinger_fx.domain import Bar, Timeframe
from stinger_fx.strategies.regime import (
    CompositeFilter,
    HighVolatilityFilter,
    LowVolatilityFilter,
    RangingFilter,
    RegimeFilter,
    TrendingFilter,
)


def _trending_bars(n: int = 50, slope: float = 0.001) -> list[Bar]:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    out = []
    for i in range(n):
        c = 1.10 + slope * i
        out.append(Bar(
            symbol="EURUSD", timeframe=Timeframe.M15,
            time=base + timedelta(minutes=15 * i),
            open=c, high=c + 0.0002, low=c - 0.0002, close=c,
            tick_volume=100, is_closed=True,
        ))
    return out


def _ranging_bars(n: int = 100) -> list[Bar]:
    """Sideways chop oscillating ±2 pips around 1.10."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    out = []
    for i in range(n):
        c = 1.10 + (0.0002 if i % 2 == 0 else -0.0002)
        out.append(Bar(
            symbol="EURUSD", timeframe=Timeframe.M15,
            time=base + timedelta(minutes=15 * i),
            open=c, high=c + 0.0001, low=c - 0.0001, close=c,
            tick_volume=100, is_closed=True,
        ))
    return out


def _expanding_volatility_bars(n: int, start_range: float = 0.0001) -> list[Bar]:
    """Bars whose range grows over time — late bars have higher ATR."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    out = []
    for i in range(n):
        rng = start_range * (1 + i * 0.05)
        c = 1.10
        out.append(Bar(
            symbol="EURUSD", timeframe=Timeframe.M15,
            time=base + timedelta(minutes=15 * i),
            open=c, high=c + rng, low=c - rng, close=c,
            tick_volume=100, is_closed=True,
        ))
    return out


def _compressing_volatility_bars(n: int, start_range: float = 0.005) -> list[Bar]:
    """Bars whose range shrinks over time — late bars have lower ATR."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    out = []
    for i in range(n):
        rng = max(start_range * (1 - i * 0.01), 0.0001)
        c = 1.10
        out.append(Bar(
            symbol="EURUSD", timeframe=Timeframe.M15,
            time=base + timedelta(minutes=15 * i),
            open=c, high=c + rng, low=c - rng, close=c,
            tick_volume=100, is_closed=True,
        ))
    return out


# --- Trending / Ranging filters ---------------------------------------------


def test_trending_filter_allows_in_strong_trend() -> None:
    bars = _trending_bars(n=50)
    f = TrendingFilter(adx_period=14, threshold=20.0)
    assert f.allows(bars) is True


def test_trending_filter_blocks_in_chop() -> None:
    bars = _ranging_bars(n=100)
    f = TrendingFilter(adx_period=14, threshold=25.0)
    assert f.allows(bars) is False


def test_ranging_filter_is_mirror_of_trending() -> None:
    """In trend → ranging filter blocks; in chop → ranging filter allows."""
    trend = _trending_bars(n=50)
    chop = _ranging_bars(n=100)
    ranging = RangingFilter(threshold=25.0)
    assert ranging.allows(trend) is False
    assert ranging.allows(chop) is True


def test_trending_filter_rejects_short_history() -> None:
    """Not enough bars → always False (never trades on incomplete data)."""
    bars = _trending_bars(n=10)
    f = TrendingFilter()
    assert f.allows(bars) is False


def test_trending_filter_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        TrendingFilter(adx_period=1)
    with pytest.raises(ValueError):
        TrendingFilter(threshold=0)


def test_ranging_filter_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        RangingFilter(adx_period=0)
    with pytest.raises(ValueError):
        RangingFilter(threshold=-1)


# --- Volatility filters -----------------------------------------------------


def test_high_volatility_filter_allows_when_range_expanding() -> None:
    """Bars with growing range → current ATR in top percentile."""
    bars = _expanding_volatility_bars(80)
    f = HighVolatilityFilter(atr_period=14, lookback=50, percentile=75)
    assert f.allows(bars) is True


def test_low_volatility_filter_allows_when_range_compressing() -> None:
    bars = _compressing_volatility_bars(80)
    f = LowVolatilityFilter(atr_period=14, lookback=50, percentile=25)
    assert f.allows(bars) is True


def test_volatility_filters_reject_short_history() -> None:
    bars = _expanding_volatility_bars(20)
    assert HighVolatilityFilter().allows(bars) is False
    assert LowVolatilityFilter().allows(bars) is False


def test_high_volatility_blocks_in_compressing_range() -> None:
    bars = _compressing_volatility_bars(80)
    assert HighVolatilityFilter(percentile=75).allows(bars) is False


def test_volatility_filter_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        HighVolatilityFilter(atr_period=0)
    with pytest.raises(ValueError):
        HighVolatilityFilter(lookback=5)
    with pytest.raises(ValueError):
        HighVolatilityFilter(percentile=100)
    with pytest.raises(ValueError):
        LowVolatilityFilter(percentile=0)


# --- Composite --------------------------------------------------------------


def test_composite_filter_logical_and() -> None:
    """A composite of two filters allows only when BOTH allow."""
    bars = _trending_bars(n=80)
    # Trending = True; LowVol should be False (trend = directional moves)
    composite = CompositeFilter(
        TrendingFilter(threshold=20.0),
        LowVolatilityFilter(percentile=25),
    )
    # Trending bars typically aren't in the bottom quartile of ATR — likely False
    result = composite.allows(bars)
    assert isinstance(result, bool)
    # Verify it's strictly AND: one True + one False → False
    trending_only = TrendingFilter(threshold=20.0).allows(bars)
    lowvol_only = LowVolatilityFilter(percentile=25).allows(bars)
    assert result == (trending_only and lowvol_only)


def test_composite_filter_requires_one_subfilter() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CompositeFilter()


def test_protocol_compliance() -> None:
    """All filter classes satisfy the RegimeFilter Protocol."""
    assert isinstance(TrendingFilter(), RegimeFilter)
    assert isinstance(RangingFilter(), RegimeFilter)
    assert isinstance(HighVolatilityFilter(), RegimeFilter)
    assert isinstance(LowVolatilityFilter(), RegimeFilter)
    assert isinstance(CompositeFilter(TrendingFilter()), RegimeFilter)
