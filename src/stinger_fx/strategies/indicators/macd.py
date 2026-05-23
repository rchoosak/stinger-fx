"""MACD — fast EMA − slow EMA, with EMA-smoothed signal line."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple


class MACDResult(NamedTuple):
    macd: float
    signal: float
    histogram: float


def _ema_series(values: Sequence[float], period: int) -> list[float]:
    """Return the full EMA series (one value per input index >= period-1).

    Seeded with the SMA over the first `period` values so the first output
    aligns with the canonical MACD implementations.
    """
    if len(values) < period:
        return []
    alpha = 2.0 / (period + 1)
    out = [sum(values[:period]) / period]
    for v in values[period:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def macd(
    values: Sequence[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> MACDResult | None:
    if fast >= slow:
        raise ValueError("fast must be < slow")
    if signal <= 0:
        raise ValueError("signal must be > 0")
    if len(values) < slow + signal:
        return None
    fast_series = _ema_series(values, fast)
    slow_series = _ema_series(values, slow)
    # Align tails — `fast_series` has more entries than `slow_series` because
    # it has more bars to work with. Trim from the left.
    diff = len(fast_series) - len(slow_series)
    fast_aligned = fast_series[diff:] if diff > 0 else fast_series
    macd_series = [f - s for f, s in zip(fast_aligned, slow_series, strict=True)]
    if len(macd_series) < signal:
        return None
    signal_series = _ema_series(macd_series, signal)
    if not signal_series:
        return None
    macd_now = macd_series[-1]
    signal_now = signal_series[-1]
    return MACDResult(
        macd=macd_now,
        signal=signal_now,
        histogram=macd_now - signal_now,
    )
