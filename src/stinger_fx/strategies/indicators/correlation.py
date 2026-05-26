"""Rolling Pearson correlation between two price series.

Used by pairs-trading and cross-asset strategies — high correlation
between two symbols means they tend to move together, low correlation
means they don't.

The two series must be aligned (one observation per timestamp). The
caller is responsible for synchronising bars before calling — typically
by taking close prices from the same bar timestamps on both symbols.

Returns a value in [-1, +1], or ``None`` if either series has fewer
than ``period`` samples or the rolling window has zero variance on
either side.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt


def correlation(
    series_a: Sequence[float],
    series_b: Sequence[float],
    period: int = 20,
) -> float | None:
    if period <= 1:
        raise ValueError("period must be > 1")
    if len(series_a) != len(series_b):
        raise ValueError(
            f"series lengths differ: a={len(series_a)} b={len(series_b)}"
        )
    if len(series_a) < period:
        return None

    a = list(series_a[-period:])
    b = list(series_b[-period:])
    mean_a = sum(a) / period
    mean_b = sum(b) / period
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    denom = sqrt(var_a * var_b)
    if denom == 0:
        return None
    return cov / denom
