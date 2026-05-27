"""Parabolic SAR (Stop And Reverse) — Welles Wilder's trailing-stop dots.

The SAR accelerates toward price as the trend continues. When price
crosses the SAR, the trend is presumed reversed: SAR flips to the
opposite side, the acceleration factor resets, and a new extreme point
seeds the next trend.

Parameters:
  * ``af_start`` — initial acceleration factor (Wilder's default 0.02)
  * ``af_step``  — increment each time price makes a new extreme (0.02)
  * ``af_max``   — cap on the acceleration factor (0.20)

Returns the current SAR value plus a hint at the prevailing trend.
Needs at least 3 bars to seed the initial direction.

This is a from-scratch computation over all supplied bars — fine for
typical history lengths (<= a few thousand). For tight loops, cache
the latest SAR + state externally and update incrementally.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NamedTuple

from stinger_fx.domain import Bar


class PSARResult(NamedTuple):
    value: float                            # current SAR price
    trend: Literal["up", "down"]            # which side of price


def psar(
    bars: Sequence[Bar],
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.20,
) -> PSARResult | None:
    if af_start <= 0 or af_step <= 0 or af_max <= 0:
        raise ValueError("all AF parameters must be > 0")
    if af_start > af_max:
        raise ValueError("af_start must be <= af_max")
    if len(bars) < 3:
        return None

    # Seed direction from the first two bars
    if bars[1].close >= bars[0].close:
        is_up = True
        sar = bars[0].low
        ep = bars[0].high
    else:
        is_up = False
        sar = bars[0].high
        ep = bars[0].low
    af = af_start

    for i in range(1, len(bars)):
        prev = bars[i - 1]
        curr = bars[i]
        # Tentative new SAR
        new_sar = sar + af * (ep - sar)

        if is_up:
            # In an up-trend, SAR can't be above the lows of the last two bars
            two_prev = bars[i - 2] if i >= 2 else prev
            new_sar = min(new_sar, prev.low, two_prev.low)
            if curr.low < new_sar:
                # Trend flipped
                is_up = False
                sar = ep            # SAR jumps to the old extreme
                ep = curr.low       # seed new extreme
                af = af_start
            else:
                sar = new_sar
                if curr.high > ep:
                    ep = curr.high
                    af = min(af + af_step, af_max)
        else:
            # Down-trend: SAR can't be below the highs of the last two bars
            two_prev = bars[i - 2] if i >= 2 else prev
            new_sar = max(new_sar, prev.high, two_prev.high)
            if curr.high > new_sar:
                is_up = True
                sar = ep
                ep = curr.high
                af = af_start
            else:
                sar = new_sar
                if curr.low < ep:
                    ep = curr.low
                    af = min(af + af_step, af_max)

    return PSARResult(value=sar, trend="up" if is_up else "down")
