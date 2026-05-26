"""Keltner Channels — EMA mid-band, ATR-scaled outer bands.

  upper  = ema(close, ema_period) + atr_mult * atr(period)
  middle = ema(close, ema_period)
  lower  = ema(close, ema_period) - atr_mult * atr(period)

Like Bollinger Bands but using ATR instead of standard deviation, so
they expand and contract with directional volatility rather than
two-sided variance. Common settings: ema_period=20, atr_period=10,
atr_mult=2.0.

Reuses :func:`stinger_fx.strategies.indicators.ema` for the mid-band
and :func:`stinger_fx.strategies.indicators.atr` for the width.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from stinger_fx.domain import Bar
from stinger_fx.strategies.indicators.atr import atr
from stinger_fx.strategies.indicators.moving_average import ema


class KeltnerChannels(NamedTuple):
    upper: float
    middle: float
    lower: float


def keltner(
    bars: Sequence[Bar],
    ema_period: int = 20,
    atr_period: int = 10,
    atr_mult: float = 2.0,
) -> KeltnerChannels | None:
    if ema_period <= 0 or atr_period <= 0:
        raise ValueError("periods must be > 0")
    if atr_mult <= 0:
        raise ValueError("atr_mult must be > 0")
    if len(bars) < max(ema_period, atr_period + 1):
        return None

    middle = ema([b.close for b in bars], ema_period)
    width = atr(bars, atr_period)
    if middle is None or width is None:
        return None
    return KeltnerChannels(
        upper=middle + atr_mult * width,
        middle=middle,
        lower=middle - atr_mult * width,
    )
