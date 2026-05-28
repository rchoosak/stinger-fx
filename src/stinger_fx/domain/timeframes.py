"""Trading timeframes supported by Stinger-Fx.

Some sub-minute timeframes (M2, M3, M10, M45) are NOT native to MT5 — they
are synthesized by the bar aggregator from M1 bars or ticks.
"""

from __future__ import annotations

from enum import StrEnum


class Timeframe(StrEnum):
    """Canonical timeframe identifiers used throughout the platform."""

    TICK = "TICK"
    M1 = "M1"
    M2 = "M2"
    M3 = "M3"
    M5 = "M5"
    M10 = "M10"
    M15 = "M15"
    M30 = "M30"
    M45 = "M45"
    H1 = "H1"
    H2 = "H2"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN1 = "MN1"

    @property
    def seconds(self) -> int:
        """Bar duration in seconds. Raises for TICK (which has no fixed duration)."""
        match self:
            case Timeframe.TICK:
                raise ValueError("TICK has no fixed duration")
            case Timeframe.M1:
                return 60
            case Timeframe.M2:
                return 120
            case Timeframe.M3:
                return 180
            case Timeframe.M5:
                return 300
            case Timeframe.M10:
                return 600
            case Timeframe.M15:
                return 900
            case Timeframe.M30:
                return 1800
            case Timeframe.M45:
                return 2700
            case Timeframe.H1:
                return 3600
            case Timeframe.H2:
                return 7200
            case Timeframe.H4:
                return 14400
            case Timeframe.D1:
                return 86400
            case Timeframe.W1:
                return 604800
            case Timeframe.MN1:
                # Calendar month — variable; use 30d as a stable approximation
                # for non-calendar math (cache keys, sizing). Calendar-aware
                # callers must compute the real boundary themselves.
                return 30 * 86400

    @property
    def is_native_mt5(self) -> bool:
        """True if MT5 exposes this timeframe directly. False = needs aggregation."""
        return self not in {Timeframe.M2, Timeframe.M3, Timeframe.M10, Timeframe.M45}

    @property
    def is_tick(self) -> bool:
        return self is Timeframe.TICK
