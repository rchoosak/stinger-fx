"""Ehlers / Hilbert-Transform-family indicators.

This module ships two of John Ehlers' best-known smoothers from
*Cybernetic Analysis for Stocks and Futures*:

  * :func:`ehlers_super_smoother` — 2-pole Butterworth low-pass filter
    that smooths price with significantly less lag than EMA at
    equivalent rejection of high-frequency noise. Drop-in replacement
    for SMA/EMA when you want crisper turning points.

  * :func:`ehlers_decycler` — high-pass + low-pass cascade that
    removes the dominant cycle, leaving only the trend. Useful as a
    pure-trend filter for regime detection.

Both indicators converge after a short warmup (~ period bars). Pure
stdlib (math only), no numpy dependency.

For the full Hilbert-Transform Dominant Cycle / Sine Wave / Instantaneous
Trendline indicators, see Ehlers' books — they require the homodyne
discriminator (~80 lines of state-tracking math). The Super Smoother
gives 80% of the value for 10% of the code.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def ehlers_super_smoother(
    closes: Sequence[float],
    period: int = 10,
) -> float | None:
    """2-pole Super Smoother (Butterworth-style) low-pass filter.

    Returns the latest filtered value. The filter:

        c1 = 1 - c2 - c3
        c2 = 2 * a1 * cos(1.414 * pi / period)
        c3 = -a1**2
        a1 = exp(-1.414 * pi / period)

        filt[i] = c1 * (close[i] + close[i-1]) / 2
                + c2 * filt[i-1]
                + c3 * filt[i-2]

    Needs at least ``period`` closes to seed; returns None until then.
    """
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    n = len(closes)
    if n < period:
        return None

    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2 * a1 * math.cos(1.414 * math.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    # Seed: first two values copy the input
    filt = [float(closes[0]), float(closes[1])]
    for i in range(2, n):
        new = (
            c1 * (float(closes[i]) + float(closes[i - 1])) / 2
            + c2 * filt[i - 1]
            + c3 * filt[i - 2]
        )
        filt.append(new)
    return filt[-1]


def ehlers_decycler(
    closes: Sequence[float],
    period: int = 60,
) -> float | None:
    """High-pass decycler — strips the dominant cycle from the close series.

    What's left is (approximately) the underlying trend. Useful as a
    regime input: positive when the trend is up, negative when down,
    near zero in chop.

    Formula (Ehlers 2013):

        alpha = (cos(0.707 * 2*pi/period) + sin(0.707 * 2*pi/period) - 1)
              / cos(0.707 * 2*pi/period)
        hp[i] = (1 - alpha/2)**2 * (close[i] - 2*close[i-1] + close[i-2])
              + 2*(1 - alpha) * hp[i-1]
              - (1 - alpha)**2 * hp[i-2]

        decycler[i] = close[i] - hp[i]      ← what you'd read

    Needs at least ``period`` closes to warm up.
    """
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    n = len(closes)
    if n < period:
        return None

    rad = 0.707 * 2 * math.pi / period
    alpha = (math.cos(rad) + math.sin(rad) - 1) / math.cos(rad)
    one_minus_a = 1 - alpha

    hp = [0.0, 0.0]
    for i in range(2, n):
        new_hp = (
            (1 - alpha / 2) ** 2
            * (float(closes[i]) - 2 * float(closes[i - 1]) + float(closes[i - 2]))
            + 2 * one_minus_a * hp[i - 1]
            - one_minus_a ** 2 * hp[i - 2]
        )
        hp.append(new_hp)
    return float(closes[-1]) - hp[-1]
