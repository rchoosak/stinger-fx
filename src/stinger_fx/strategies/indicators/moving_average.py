"""Simple and exponential moving averages over a closing-price series."""

from __future__ import annotations

from collections.abc import Sequence


def sma(values: Sequence[float], period: int) -> float | None:
    """Latest simple moving average. Returns None if not enough data."""
    if period <= 0:
        raise ValueError("period must be > 0")
    if len(values) < period:
        return None
    window = values[-period:]
    return sum(window) / period


def ema(values: Sequence[float], period: int) -> float | None:
    """Latest exponential moving average (seeded with SMA)."""
    if period <= 0:
        raise ValueError("period must be > 0")
    if len(values) < period:
        return None
    alpha = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    out = seed
    for v in values[period:]:
        out = alpha * v + (1 - alpha) * out
    return out
