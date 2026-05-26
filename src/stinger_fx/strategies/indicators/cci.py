"""Commodity Channel Index — Donald Lambert's mean-reversion oscillator.

  typical_price = (high + low + close) / 3
  sma_tp        = SMA(typical_price, period)
  mean_dev      = (1/period) * sum(|TP_i - sma_tp|) over the window
  CCI           = (TP_now - sma_tp) / (0.015 * mean_dev)

Reading: ±100 are common thresholds. CCI > +100 = strong up-move (often
overbought); CCI < -100 = strong down-move (often oversold). Lambert
designed the 0.015 constant so ~70-80% of values fall in [-100, +100].

Returns ``None`` when the input has fewer than ``period`` bars or when
the mean absolute deviation is zero (flat price stream).
"""

from __future__ import annotations

from collections.abc import Sequence

from stinger_fx.domain import Bar


def cci(bars: Sequence[Bar], period: int = 20) -> float | None:
    if period <= 0:
        raise ValueError("period must be > 0")
    if len(bars) < period:
        return None
    window = bars[-period:]
    typicals = [(b.high + b.low + b.close) / 3 for b in window]
    sma_tp = sum(typicals) / period
    mean_dev = sum(abs(t - sma_tp) for t in typicals) / period
    if mean_dev == 0:
        return None
    return (typicals[-1] - sma_tp) / (0.015 * mean_dev)
