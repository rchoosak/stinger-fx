"""Wall-clock throttle for backtest replay.

The default backtest path replays historical bars/ticks as fast as Python
can iterate them — perfect for getting a metric out, but useless when you
want to *watch* a strategy work (debug UI, demo, live coding, dashboard
playback). `PlaybackThrottle` lets the caller pin replay to wall-clock
time at any rate from "max speed" (the default) through real-time
(`speed=1`) up to arbitrary multipliers (`speed=60` → 1 sim-minute per
wall-second) or slow-motion (`speed=0.5` → half real-time).

The throttle is wall-clock pacing only; it does NOT touch ``SimClock``
(strategies still see the simulated broker timestamp via ``clock.now()``).
That separation matters because metrics, equity samples, and the
backtest report are all keyed on sim time — a throttled replay produces
identical numbers, just delivered to subscribers at a different pace.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime


class PlaybackThrottle:
    """Sleep `wait_for()` calls so events land at the right wall-clock pace.

    Semantics:
        speed == 0.0  → no-op (max speed, behavior unchanged)
        speed == 1.0  → real-time: 1 sim-second per wall-second
        speed >  1.0  → faster than real-time (e.g. ``60`` = 1 min/sec)
        speed <  1.0  → slow-motion (e.g. ``0.5`` = half real-time)

    The first call to ``wait_for`` is the baseline — it returns immediately
    and anchors both the sim clock and the wall clock. Subsequent calls
    sleep just enough to keep ``(sim - sim_start) / speed`` aligned with
    ``(wall - wall_start)``. If the caller is already late (handler took
    longer than the budget), no sleep happens — we just keep up.

    Raises:
        ValueError: if ``speed`` is negative.
    """

    def __init__(self, speed: float) -> None:
        if speed < 0:
            raise ValueError(f"speed must be >= 0; got {speed}")
        self._speed = float(speed)
        self._sim_start: datetime | None = None
        self._wall_start: float | None = None

    @property
    def enabled(self) -> bool:
        """True when the throttle will actually delay; False = no-op mode."""
        return self._speed > 0.0

    @property
    def speed(self) -> float:
        return self._speed

    async def wait_for(self, sim_time: datetime) -> None:
        """Block until wall-clock catches up to where ``sim_time`` should land.

        Cheap when ``speed == 0`` (just an inlined `if` + return).
        Cheap when the caller is already behind schedule (no `asyncio.sleep`).
        """
        if not self.enabled:
            return
        if self._sim_start is None:
            # Baseline — anchor both clocks here, don't sleep.
            self._sim_start = sim_time
            self._wall_start = time.monotonic()
            return
        assert self._wall_start is not None  # narrowed by sim_start check
        sim_elapsed = (sim_time - self._sim_start).total_seconds() / self._speed
        wall_elapsed = time.monotonic() - self._wall_start
        lag = sim_elapsed - wall_elapsed
        if lag > 0:
            await asyncio.sleep(lag)
