"""Market regime detection — filters for "should we be trading right now?"

Most strategies that work in one regime fail in another. A momentum
breakout system bleeds money in chop; a mean-reversion oscillator gets
run over in a strong trend. Regime detection lets you gate signal
emission on whether the current market state matches what your strategy
was designed for.

Two regime dimensions are supported:

  * **Trend** — measured via ADX. ``adx > threshold`` means a trend is
    in progress (default threshold 25, the textbook value).
  * **Volatility** — measured via ATR percentile. Compute ATR over a
    rolling window and compare the current ATR against the distribution
    of past ATRs over a longer lookback.

Each dimension exposes two filters: ``Trending`` / ``Ranging`` for trend,
``HighVolatility`` / ``LowVolatility`` for vol. Combine via
``CompositeFilter`` (logical AND):

    f = CompositeFilter(TrendingFilter(threshold=20), LowVolatilityFilter())
    if f.allows(ctx.history.bars()):
        await ctx.buy(0.1)

All filters return ``False`` (never allows) when there isn't enough
history yet — strategies don't get tricked into trading on incomplete data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from stinger_fx.domain import Bar
from stinger_fx.strategies.indicators import adx, atr


@runtime_checkable
class RegimeFilter(Protocol):
    """Anything that can decide whether the current bars allow trading."""

    def allows(self, bars: Sequence[Bar]) -> bool: ...


# --- Trend filters ----------------------------------------------------------


class TrendingFilter:
    """Allow trading only when ADX indicates a trend is in progress.

    Default threshold of 25 is the classic Wilder cutoff: below 20 = no
    trend, above 25 = confirmed trend, above 40 = strong / possibly
    exhausted.
    """

    def __init__(self, adx_period: int = 14, threshold: float = 25.0) -> None:
        if adx_period <= 1:
            raise ValueError(f"adx_period must be > 1, got {adx_period}")
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        self._period = adx_period
        self._threshold = threshold

    def allows(self, bars: Sequence[Bar]) -> bool:
        result = adx(bars, self._period)
        if result is None:
            return False
        return result.adx >= self._threshold


class RangingFilter:
    """Allow trading only when ADX indicates no clear trend.

    Mirror of TrendingFilter — useful for mean-reversion strategies that
    work in chop and lose in trends.
    """

    def __init__(self, adx_period: int = 14, threshold: float = 25.0) -> None:
        if adx_period <= 1:
            raise ValueError(f"adx_period must be > 1, got {adx_period}")
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        self._period = adx_period
        self._threshold = threshold

    def allows(self, bars: Sequence[Bar]) -> bool:
        result = adx(bars, self._period)
        if result is None:
            return False
        return result.adx < self._threshold


# --- Volatility filters -----------------------------------------------------


class HighVolatilityFilter:
    """Allow trading when current ATR is above its rolling percentile.

    Method: compute ATR at the current bar, then compute the distribution
    of bar-by-bar ATR over the last ``lookback`` bars (recomputing ATR at
    each historical offset). If the current ATR is at or above the
    ``percentile``-th value in that distribution, the regime is "high vol".

    Default lookback=50 bars + atr_period=14 → needs 64 bars to make a
    decision.
    """

    def __init__(
        self,
        atr_period: int = 14,
        lookback: int = 50,
        percentile: int = 75,
    ) -> None:
        if atr_period <= 1:
            raise ValueError(f"atr_period must be > 1, got {atr_period}")
        if lookback < 10:
            raise ValueError(f"lookback must be >= 10, got {lookback}")
        if not (0 < percentile < 100):
            raise ValueError(f"percentile must be in (0, 100), got {percentile}")
        self._atr_period = atr_period
        self._lookback = lookback
        self._percentile = percentile

    def allows(self, bars: Sequence[Bar]) -> bool:
        return _vol_regime(bars, self._atr_period, self._lookback, self._percentile, "high")


class LowVolatilityFilter:
    """Allow trading when current ATR is BELOW its rolling percentile.

    Mirror of HighVolatilityFilter. Useful for strategies (e.g. grid /
    scalping) that thrive in compressed volatility.
    """

    def __init__(
        self,
        atr_period: int = 14,
        lookback: int = 50,
        percentile: int = 25,
    ) -> None:
        if atr_period <= 1:
            raise ValueError(f"atr_period must be > 1, got {atr_period}")
        if lookback < 10:
            raise ValueError(f"lookback must be >= 10, got {lookback}")
        if not (0 < percentile < 100):
            raise ValueError(f"percentile must be in (0, 100), got {percentile}")
        self._atr_period = atr_period
        self._lookback = lookback
        self._percentile = percentile

    def allows(self, bars: Sequence[Bar]) -> bool:
        return _vol_regime(bars, self._atr_period, self._lookback, self._percentile, "low")


# --- Composite --------------------------------------------------------------


class CompositeFilter:
    """Combine multiple regime filters with logical AND.

    Returns True only when *every* sub-filter allows. Use to express
    "trade only in trending markets with low volatility":

        f = CompositeFilter(
            TrendingFilter(threshold=20),
            LowVolatilityFilter(percentile=40),
        )
    """

    def __init__(self, *filters: RegimeFilter) -> None:
        if not filters:
            raise ValueError("CompositeFilter requires at least one sub-filter")
        self._filters = list(filters)

    def allows(self, bars: Sequence[Bar]) -> bool:
        return all(f.allows(bars) for f in self._filters)


# --- Internals --------------------------------------------------------------


def _vol_regime(
    bars: Sequence[Bar],
    atr_period: int,
    lookback: int,
    percentile: int,
    side: str,
) -> bool:
    """Compute current vs historical ATR distribution and return the
    requested side ("high" or "low") of the percentile band."""
    needed = atr_period + lookback
    if len(bars) < needed:
        return False
    bars_list = list(bars)
    # ATR at each historical offset — most recent at index 0
    atrs: list[float] = []
    for offset in range(lookback):
        end = len(bars_list) - offset
        if end < atr_period + 1:
            break
        a = atr(bars_list[:end], atr_period)
        if a is not None:
            atrs.append(a)
    if not atrs:
        return False
    current = atrs[0]
    historical_sorted = sorted(atrs)
    idx = int(len(historical_sorted) * percentile / 100)
    idx = max(0, min(idx, len(historical_sorted) - 1))
    threshold = historical_sorted[idx]
    if side == "high":
        return current >= threshold
    # side == "low"
    return current <= threshold
