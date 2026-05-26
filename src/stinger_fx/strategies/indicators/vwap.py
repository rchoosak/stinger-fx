"""Volume-Weighted Average Price — rolling and session variants.

Two flavours:

  * :func:`vwap_rolling` — VWAP over the last ``period`` bars.  Useful
    when you want an indicator that "follows" recent activity but
    doesn't anchor to a calendar boundary.

  * :func:`vwap_session` — VWAP from the start of the supplied bar list,
    typically called with bars filtered to "today's session". The caller
    is responsible for picking session boundaries (UTC midnight, broker
    rollover, etc.) because those vary by market.

Both use the typical price (H+L+C)/3 weighted by ``bar.tick_volume`` —
real volume (real_volume) is available too but tick_volume is universal
across symbols.
"""

from __future__ import annotations

from collections.abc import Sequence

from stinger_fx.domain import Bar


def vwap_rolling(bars: Sequence[Bar], period: int = 20) -> float | None:
    """VWAP over the trailing ``period`` bars (None until enough data)."""
    if period <= 0:
        raise ValueError("period must be > 0")
    if len(bars) < period:
        return None
    window = bars[-period:]
    return _vwap(window)


def vwap_session(bars: Sequence[Bar]) -> float | None:
    """VWAP from the start of the supplied window.  Pass session-filtered
    bars (e.g. today's bars only)."""
    if not bars:
        return None
    return _vwap(bars)


def _vwap(window: Sequence[Bar]) -> float | None:
    total_pv = 0.0
    total_v = 0.0
    for b in window:
        typical = (b.high + b.low + b.close) / 3
        vol = float(b.tick_volume)
        total_pv += typical * vol
        total_v += vol
    if total_v == 0:
        # All bars have zero volume — no meaningful VWAP. Fall back to
        # un-weighted mean of typical price so the caller still gets a
        # number instead of None (caller can decide what to do).
        if not window:
            return None
        return sum((b.high + b.low + b.close) / 3 for b in window) / len(window)
    return total_pv / total_v
