"""Stochastic RSI — applies the Stochastic formula to the RSI series.

Two-step indicator:
  1. Compute RSI on the close series (default 14-period)
  2. Apply Stochastic formula on the resulting RSI values:
       %K_raw = (current_rsi - min_rsi_window) /
                (max_rsi_window - min_rsi_window) * 100
  3. Smooth %K_raw with a SMA over ``k_smooth`` periods
  4. %D = SMA of smoothed %K over ``d_smooth`` periods

Range: 0–100. Common reading:
  * %K > 80 = overbought (RSI is near its recent peak)
  * %K < 20 = oversold (RSI is near its recent trough)

Stoch RSI reacts faster than plain Stochastic or plain RSI because it
amplifies turning points in the already-smoothed RSI series.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from stinger_fx.strategies.indicators.rsi import rsi


class StochRSIResult(NamedTuple):
    k: float    # 0–100, smoothed
    d: float    # 0–100, SMA of k


def stoch_rsi(
    closes: Sequence[float],
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> StochRSIResult | None:
    """Compute the Stochastic RSI %K (smoothed) + %D for the latest bar.

    Needs at least ``rsi_period + stoch_period + max(k_smooth, d_smooth)``
    closes to produce a stable output. Returns None until then.
    """
    if min(rsi_period, stoch_period, k_smooth, d_smooth) < 1:
        raise ValueError("all periods must be >= 1")
    # We need enough RSI history to compute raw %K over a `stoch_period`
    # window at each of the last `(k_smooth + d_smooth - 1)` positions —
    # plus one more value for the current bar. Total RSI samples needed:
    #   stoch_period + (k_smooth + d_smooth - 1).
    extra = stoch_period + k_smooth + d_smooth - 2
    needed = rsi_period + extra + 1
    if len(closes) < needed:
        return None

    # Compute RSI at each tail offset
    rsi_series: list[float] = []
    for offset in range(extra + 1):
        end = len(closes) - offset
        val = rsi(closes[:end], rsi_period)
        if val is None:
            return None
        rsi_series.insert(0, val)  # most recent at end

    # Compute raw Stoch %K at each of the last (k_smooth + d_smooth - 1) positions
    raw_k_series: list[float] = []
    for i in range(k_smooth + d_smooth - 1, -1, -1):
        # Stoch RSI window ends at position len - i, length stoch_period
        end = len(rsi_series) - i
        if end < stoch_period:
            return None
        window = rsi_series[end - stoch_period:end]
        cur = window[-1]
        lo, hi = min(window), max(window)
        if hi == lo:
            raw_k_series.append(50.0)   # neutral when no movement
        else:
            raw_k_series.append((cur - lo) / (hi - lo) * 100)

    # Smooth %K with SMA over k_smooth
    if len(raw_k_series) < k_smooth:
        return None
    smoothed_k_series: list[float] = []
    for i in range(d_smooth):
        end = len(raw_k_series) - (d_smooth - 1 - i)
        if end < k_smooth:
            return None
        smoothed_k_series.append(sum(raw_k_series[end - k_smooth:end]) / k_smooth)

    k = smoothed_k_series[-1]
    d = sum(smoothed_k_series) / d_smooth
    return StochRSIResult(k=k, d=d)
