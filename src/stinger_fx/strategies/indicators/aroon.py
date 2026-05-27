"""Aroon — measures how recently the highest high / lowest low occurred.

  Aroon Up   = ((period - bars_since_highest_high) / period) * 100
  Aroon Down = ((period - bars_since_lowest_low)   / period) * 100
  Oscillator = Aroon Up - Aroon Down  (range -100..+100)

Reading:
  * Aroon Up = 100  → highest high is THIS bar (strong uptrend)
  * Aroon Up = 0    → highest high was `period` bars ago
  * Aroon Up > Aroon Down → up-trend; reverse → down-trend
  * Both < 50 → consolidation

Returns ``None`` when fewer than ``period + 1`` bars (need period+1 to
compare 0..period bars ago).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from stinger_fx.domain import Bar


class AroonResult(NamedTuple):
    up: float           # 0–100
    down: float         # 0–100
    oscillator: float   # up - down, range -100..+100


def aroon(bars: Sequence[Bar], period: int = 25) -> AroonResult | None:
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")
    if len(bars) < period + 1:
        return None
    window = list(bars[-(period + 1):])
    highs = [b.high for b in window]
    lows = [b.low for b in window]

    # Index of the most recent maximum (tie-breaks toward the latest)
    hh_idx = max(range(len(highs)), key=lambda i: (highs[i], i))
    ll_idx = min(range(len(lows)), key=lambda i: (lows[i], -i))

    bars_since_high = len(window) - 1 - hh_idx
    bars_since_low = len(window) - 1 - ll_idx

    up = (period - bars_since_high) / period * 100
    down = (period - bars_since_low) / period * 100
    return AroonResult(up=up, down=down, oscillator=up - down)
