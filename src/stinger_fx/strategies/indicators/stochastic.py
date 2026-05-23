"""Stochastic oscillator — %K and %D over recent highs/lows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from stinger_fx.domain import Bar


class StochasticResult(NamedTuple):
    k: float
    d: float


def stochastic(
    bars: Sequence[Bar],
    k_period: int = 14,
    d_period: int = 3,
) -> StochasticResult | None:
    if k_period <= 0 or d_period <= 0:
        raise ValueError("periods must be > 0")
    if len(bars) < k_period + d_period - 1:
        return None
    # Compute %K for the last d_period bars so we can average them into %D.
    k_values: list[float] = []
    for i in range(len(bars) - d_period + 1, len(bars) + 1):
        window = bars[i - k_period:i]
        if len(window) < k_period:
            return None
        high = max(b.high for b in window)
        low = min(b.low for b in window)
        if high == low:
            k = 50.0
        else:
            k = (window[-1].close - low) / (high - low) * 100
        k_values.append(k)
    return StochasticResult(k=k_values[-1], d=sum(k_values) / len(k_values))
