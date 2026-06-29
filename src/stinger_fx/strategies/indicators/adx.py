"""Average Directional Index + DI+ / DI− — Wilder's trend-strength indicator.

ADX answers "how strong is the trend?" without caring about direction.
DI+ and DI− answer "which way is it moving?". Both use Wilder's smoothing
(equivalent to an exponential moving average with α = 1/period).

Conventional reading:
  * ADX < 20: weak / non-trending — range strategies preferred
  * ADX 20–40: confirmed trend
  * ADX > 40: strong trend, often near exhaustion
  * DI+ > DI−: up-trend; DI+ < DI−: down-trend

Returns ``None`` when the input has fewer than ``2 * period`` bars (one
period to seed the running average, another to compute ADX from DX).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from stinger_fx.domain import Bar
from stinger_fx.strategies.indicators._smoothing import ADX_TAIL_FACTOR, wilder_tail


class ADXResult(NamedTuple):
    adx: float
    plus_di: float
    minus_di: float


def adx(bars: Sequence[Bar], period: int = 14) -> ADXResult | None:
    if period <= 1:
        raise ValueError("period must be > 1")
    if len(bars) < 2 * period:
        return None

    # Two cascaded Wilder stages, so the seed decays slower than a single one —
    # cap the input to a wider trailing window that's still bit-identical (pinned
    # by property tests) instead of recomputing over a 2000-bar history each bar.
    bars = wilder_tail(bars, period, ADX_TAIL_FACTOR)

    # Build per-bar TR, +DM, -DM
    trs: list[float] = []
    plus_dms: list[float] = []
    minus_dms: list[float] = []
    for i in range(1, len(bars)):
        prev, curr = bars[i - 1], bars[i]
        up = curr.high - prev.high
        down = prev.low - curr.low
        plus_dm = up if (up > down and up > 0) else 0.0
        minus_dm = down if (down > up and down > 0) else 0.0
        tr = max(
            curr.high - curr.low,
            abs(curr.high - prev.close),
            abs(curr.low - prev.close),
        )
        trs.append(tr)
        plus_dms.append(plus_dm)
        minus_dms.append(minus_dm)

    # Wilder seed averages over the first `period` values
    tr_smooth = sum(trs[:period])
    plus_smooth = sum(plus_dms[:period])
    minus_smooth = sum(minus_dms[:period])

    dxs: list[float] = []
    # Compute DX for the initial period
    if tr_smooth == 0:
        return None
    plus_di_val = 100 * plus_smooth / tr_smooth
    minus_di_val = 100 * minus_smooth / tr_smooth
    di_sum = plus_di_val + minus_di_val
    dx = 100 * abs(plus_di_val - minus_di_val) / di_sum if di_sum > 0 else 0.0
    dxs.append(dx)

    # Wilder smoothing for subsequent bars
    for i in range(period, len(trs)):
        tr_smooth = tr_smooth - tr_smooth / period + trs[i]
        plus_smooth = plus_smooth - plus_smooth / period + plus_dms[i]
        minus_smooth = minus_smooth - minus_smooth / period + minus_dms[i]
        if tr_smooth == 0:
            continue
        plus_di_val = 100 * plus_smooth / tr_smooth
        minus_di_val = 100 * minus_smooth / tr_smooth
        di_sum = plus_di_val + minus_di_val
        dx = 100 * abs(plus_di_val - minus_di_val) / di_sum if di_sum > 0 else 0.0
        dxs.append(dx)

    if len(dxs) < period:
        return None

    # ADX is Wilder-smoothed DX
    adx_val = sum(dxs[:period]) / period
    for dx_val in dxs[period:]:
        adx_val = (adx_val * (period - 1) + dx_val) / period

    return ADXResult(adx=adx_val, plus_di=plus_di_val, minus_di=minus_di_val)
