"""Bollinger Bands — SMA ± N standard deviations."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from typing import NamedTuple


class BollingerBands(NamedTuple):
    upper: float
    middle: float
    lower: float


def bollinger(
    values: Sequence[float],
    period: int = 20,
    stddev_mult: float = 2.0,
) -> BollingerBands | None:
    if period <= 1:
        raise ValueError("period must be > 1")
    if stddev_mult <= 0:
        raise ValueError("stddev_mult must be > 0")
    if len(values) < period:
        return None
    window = values[-period:]
    mean = sum(window) / period
    variance = sum((v - mean) ** 2 for v in window) / period
    sd = sqrt(variance)
    return BollingerBands(
        upper=mean + stddev_mult * sd,
        middle=mean,
        lower=mean - stddev_mult * sd,
    )
