"""Pivot Points — Classic, Fibonacci, and Camarilla.

All three methods derive S1/S2/S3 and R1/R2/R3 from the *previous* period's
high, low, and close. Typically computed once per session (yesterday's
H/L/C → today's pivots) but the math is the same for any timeframe.

Returns a structured result with the central pivot plus the three
support and three resistance levels.
"""

from __future__ import annotations

from typing import Literal, NamedTuple


class PivotLevels(NamedTuple):
    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float


def pivot_points(
    prev_high: float,
    prev_low: float,
    prev_close: float,
    method: Literal["classic", "fibonacci", "camarilla"] = "classic",
) -> PivotLevels:
    """Return support/resistance levels derived from the previous period.

    ``method``:
      * ``"classic"``    — standard pivot (most common)
      * ``"fibonacci"``  — uses 0.382 / 0.618 / 1.000 fib ratios on range
      * ``"camarilla"``  — tighter S/R levels via 1.1/12 ratio family
    """
    rng = prev_high - prev_low
    pivot = (prev_high + prev_low + prev_close) / 3

    if method == "classic":
        r1 = 2 * pivot - prev_low
        s1 = 2 * pivot - prev_high
        r2 = pivot + rng
        s2 = pivot - rng
        r3 = prev_high + 2 * (pivot - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)
    elif method == "fibonacci":
        r1 = pivot + 0.382 * rng
        r2 = pivot + 0.618 * rng
        r3 = pivot + 1.000 * rng
        s1 = pivot - 0.382 * rng
        s2 = pivot - 0.618 * rng
        s3 = pivot - 1.000 * rng
    elif method == "camarilla":
        r1 = prev_close + rng * 1.1 / 12
        r2 = prev_close + rng * 1.1 / 6
        r3 = prev_close + rng * 1.1 / 4
        s1 = prev_close - rng * 1.1 / 12
        s2 = prev_close - rng * 1.1 / 6
        s3 = prev_close - rng * 1.1 / 4
    else:
        raise ValueError(f"unknown pivot method: {method!r}")

    return PivotLevels(pivot=pivot, r1=r1, r2=r2, r3=r3, s1=s1, s2=s2, s3=s3)
