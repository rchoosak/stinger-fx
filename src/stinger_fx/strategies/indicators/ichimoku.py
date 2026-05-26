"""Ichimoku Kinko Hyo — five-component trend system.

Components (standard 9/26/52 periods on daily charts; Japanese FX
convention uses 9/26/52 too):

  * **Tenkan-sen**  (Conversion Line) — midpoint of last 9 highs/lows
  * **Kijun-sen**   (Base Line)        — midpoint of last 26 highs/lows
  * **Senkou A**    (Leading Span A)   — (Tenkan + Kijun) / 2,
                                         plotted 26 bars ahead
  * **Senkou B**    (Leading Span B)   — midpoint of last 52 highs/lows,
                                         plotted 26 bars ahead
  * **Chikou**      (Lagging Span)     — current close,
                                         plotted 26 bars back

The Kumo (cloud) is the space between Senkou A and Senkou B.  Bullish
when price > Kumo; bearish when price < Kumo; consolidation inside.

The returned spans are the *current* values (not the projected future
or past displacement) — the caller chooses how to display them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from stinger_fx.domain import Bar


class IchimokuResult(NamedTuple):
    tenkan: float
    kijun: float
    senkou_a: float
    senkou_b: float
    chikou: float


def ichimoku(
    bars: Sequence[Bar],
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
) -> IchimokuResult | None:
    if min(tenkan_period, kijun_period, senkou_b_period) <= 0:
        raise ValueError("all periods must be > 0")
    if len(bars) < senkou_b_period:
        return None

    def _midpoint(window: Sequence[Bar]) -> float:
        return (max(b.high for b in window) + min(b.low for b in window)) / 2

    tenkan = _midpoint(bars[-tenkan_period:])
    kijun = _midpoint(bars[-kijun_period:])
    senkou_a = (tenkan + kijun) / 2
    senkou_b = _midpoint(bars[-senkou_b_period:])
    # Chikou is just current close (caller offsets when plotting)
    chikou = bars[-1].close
    return IchimokuResult(
        tenkan=tenkan,
        kijun=kijun,
        senkou_a=senkou_a,
        senkou_b=senkou_b,
        chikou=chikou,
    )
