"""Heikin-Ashi candles — smoothed candlesticks for cleaner trend reading.

Each HA candle is computed from a regular OHLC candle plus the previous
HA candle:

  ha_close = (open + high + low + close) / 4
  ha_open  = (prev_ha_open + prev_ha_close) / 2
  ha_high  = max(high, ha_open, ha_close)
  ha_low   = min(low,  ha_open, ha_close)

Consecutive HA candles share state, so the series is path-dependent —
seeding from a different starting bar gives slightly different values
until the chain stabilises.

Use HA when you want to filter out single-bar noise from a candle-based
strategy: an HA candle with no upper shadow + green body = strong up
momentum (no buyer hesitation).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from stinger_fx.domain import Bar


class HeikinAshiCandle(NamedTuple):
    open: float
    high: float
    low: float
    close: float

    @property
    def body(self) -> float:
        """Signed body size — positive when close > open."""
        return self.close - self.open

    @property
    def is_green(self) -> bool:
        return self.close >= self.open

    @property
    def upper_shadow(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_shadow(self) -> float:
        return min(self.open, self.close) - self.low


def heikin_ashi(bars: Sequence[Bar]) -> HeikinAshiCandle | None:
    """Return the LATEST Heikin-Ashi candle for the supplied bar history.

    Iterates the chain from bars[0] to bars[-1]; returns the last value.
    Returns None for empty input.
    """
    series = heikin_ashi_series(bars)
    return series[-1] if series else None


def heikin_ashi_series(bars: Sequence[Bar]) -> list[HeikinAshiCandle]:
    """Convert an OHLC bar series into the corresponding HA series.

    Returns a list the same length as ``bars``. Empty input → empty list.
    """
    if not bars:
        return []
    out: list[HeikinAshiCandle] = []
    # Seed the first HA candle (no previous HA — use the bar's own open/close)
    first = bars[0]
    ha_close = (first.open + first.high + first.low + first.close) / 4
    ha_open = (first.open + first.close) / 2
    ha_high = max(first.high, ha_open, ha_close)
    ha_low = min(first.low, ha_open, ha_close)
    out.append(HeikinAshiCandle(open=ha_open, high=ha_high, low=ha_low, close=ha_close))

    for b in bars[1:]:
        prev = out[-1]
        new_close = (b.open + b.high + b.low + b.close) / 4
        new_open = (prev.open + prev.close) / 2
        new_high = max(b.high, new_open, new_close)
        new_low = min(b.low, new_open, new_close)
        out.append(HeikinAshiCandle(
            open=new_open, high=new_high, low=new_low, close=new_close
        ))
    return out
