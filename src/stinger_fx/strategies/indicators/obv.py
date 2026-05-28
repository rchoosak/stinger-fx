"""On-Balance Volume — cumulative volume signed by price direction.

  +volume on up bars (close > prev close)
  −volume on down bars
   0 on doji (close == prev close)

The absolute value is meaningless (depends on history start); only the
*direction* and *divergence vs price* matter. A rising OBV with falling
price suggests accumulation; the reverse suggests distribution.

Volume source is ``bar.tick_volume`` (broker-universal). Strategies on
real-volume markets can swap in ``bar.real_volume`` by computing OBV
themselves — the math is trivial.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

from stinger_fx.domain import Bar


def obv(bars: Sequence[Bar]) -> float | None:
    """Cumulative OBV from the start of the supplied window.

    Returns ``None`` for fewer than 2 bars (can't compare close-to-prev).
    """
    if len(bars) < 2:
        return None
    total = 0.0
    for prev, curr in itertools.pairwise(bars):
        if curr.close > prev.close:
            total += float(curr.tick_volume)
        elif curr.close < prev.close:
            total -= float(curr.tick_volume)
        # equal close → no change
    return total
