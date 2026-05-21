"""Average True Range — Wilder's smoothing on (high, low, close) bars."""

from __future__ import annotations

from collections.abc import Sequence

from stinger_fx.domain import Bar


def atr(bars: Sequence[Bar], period: int = 14) -> float | None:
    if period <= 0:
        raise ValueError("period must be > 0")
    if len(bars) <= period:
        return None
    trs: list[float] = []
    prev_close = bars[0].close
    for b in bars[1:]:
        tr = max(b.high - b.low, abs(b.high - prev_close), abs(b.low - prev_close))
        trs.append(tr)
        prev_close = b.close
    if len(trs) < period:
        return None
    out = sum(trs[:period]) / period
    for tr in trs[period:]:
        out = (out * (period - 1) + tr) / period
    return out
