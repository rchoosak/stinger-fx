"""Williams %R — momentum oscillator scaled to [-100, 0].

  %R = (highest_high - close) / (highest_high - lowest_low) * -100

Reading:
  * %R > -20  → overbought (close is near the recent high)
  * %R < -80  → oversold  (close is near the recent low)

Williams %R is the same idea as Stochastic %K, just scaled differently:
%K = 100 - (-%R). Use whichever scale matches your habit.
"""

from __future__ import annotations

from collections.abc import Sequence

from stinger_fx.domain import Bar


def williams_r(bars: Sequence[Bar], period: int = 14) -> float | None:
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")
    if len(bars) < period:
        return None
    window = bars[-period:]
    hh = max(b.high for b in window)
    ll = min(b.low for b in window)
    close = window[-1].close
    if hh == ll:
        return -50.0    # neutral when there's no range
    return (hh - close) / (hh - ll) * -100
