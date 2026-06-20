"""RSI — Wilder's smoothing."""

from __future__ import annotations

from collections.abc import Sequence


def rsi(values: Sequence[float], period: int = 14) -> float | None:
    if period <= 0:
        raise ValueError("period must be > 0")
    if len(values) <= period:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        delta = values[i] - values[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    for i in range(period + 1, len(values)):
        delta = values[i] - values[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def rsi_series(values: Sequence[float], period: int = 14) -> list[float]:
    """Wilder's RSI at *every* index from ``period`` onward, in one O(n) pass.

    Returns ``[rsi_at_period, rsi_at_period+1, …, rsi_at_last]`` — the last
    element equals ``rsi(values, period)``. Empty when there isn't enough
    data (``len(values) <= period``).

    This exists because ``stoch_rsi`` needs the RSI value at many adjacent
    tail offsets. Calling ``rsi(values[:end])`` in a loop is O(n) per call
    → O(n·k) overall; computing the whole series incrementally is O(n),
    ~18× faster for the default Stoch-RSI parameters.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    if len(values) <= period:
        return []
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        delta = values[i] - values[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period

    def _val(g: float, loss: float) -> float:
        if loss == 0:
            return 100.0
        rs = g / loss
        return 100.0 - 100.0 / (1.0 + rs)

    out = [_val(avg_gain, avg_loss)]  # RSI at index `period`
    for i in range(period + 1, len(values)):
        delta = values[i] - values[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out.append(_val(avg_gain, avg_loss))
    return out
