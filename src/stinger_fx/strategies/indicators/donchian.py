"""Donchian channels — highest-high / lowest-low over N bars."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from stinger_fx.domain import Bar


class DonchianChannels(NamedTuple):
    upper: float
    middle: float
    lower: float


def donchian(bars: Sequence[Bar], period: int = 20) -> DonchianChannels | None:
    if period <= 0:
        raise ValueError("period must be > 0")
    if len(bars) < period:
        return None
    window = bars[-period:]
    upper = max(b.high for b in window)
    lower = min(b.low for b in window)
    return DonchianChannels(upper=upper, middle=(upper + lower) / 2, lower=lower)
