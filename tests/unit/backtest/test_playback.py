"""Unit tests for ``PlaybackThrottle`` — wall-clock pacing for backtest replay.

Pins:
  * ``speed == 0`` is a no-op (returns immediately, never sleeps).
  * ``speed == 1`` paces events to wall-clock real-time (1 sim-sec = 1 wall-sec).
  * ``speed > 1`` compresses time proportionally; ``speed < 1`` expands it.
  * The first ``wait_for`` call anchors both clocks and returns immediately —
    consumers can call it on the first event without paying a sleep penalty.
  * If the caller is already behind schedule, ``wait_for`` returns immediately
    (no negative sleep, no exception) — slow handlers naturally fall behind
    rather than tripping the throttle.
  * Negative speed is rejected at construction.
"""
from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest

from stinger_fx.backtest.playback import PlaybackThrottle


BASE = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_no_throttle_when_speed_is_zero() -> None:
    """speed=0 → no sleep, even across far-apart sim timestamps."""
    throttle = PlaybackThrottle(0.0)
    assert throttle.enabled is False

    start = time.monotonic()
    await throttle.wait_for(BASE)
    await throttle.wait_for(BASE + timedelta(hours=1))  # huge sim jump
    elapsed = time.monotonic() - start
    # Should be ~0 — well under 50ms.
    assert elapsed < 0.05, f"speed=0 must not sleep; took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_first_call_establishes_baseline() -> None:
    """The very first wait_for never sleeps — it just anchors both clocks."""
    throttle = PlaybackThrottle(1.0)
    start = time.monotonic()
    await throttle.wait_for(BASE)
    elapsed = time.monotonic() - start
    assert elapsed < 0.05, f"first call must not sleep; took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_real_time_playback_matches_sim_delta() -> None:
    """speed=1 → second wait_for should sleep ~the sim-time delta."""
    throttle = PlaybackThrottle(1.0)
    await throttle.wait_for(BASE)
    start = time.monotonic()
    await throttle.wait_for(BASE + timedelta(milliseconds=100))
    elapsed = time.monotonic() - start
    # Real-time: 100ms sim → ~100ms wall. Allow ±30ms scheduler jitter.
    assert 0.07 < elapsed < 0.20, (
        f"real-time playback should sleep ~0.10s; got {elapsed:.3f}s"
    )


@pytest.mark.asyncio
async def test_speed_multiplier_compresses_time() -> None:
    """speed=10 → 1.0s sim delta should land in ~0.10s wall."""
    throttle = PlaybackThrottle(10.0)
    await throttle.wait_for(BASE)
    start = time.monotonic()
    await throttle.wait_for(BASE + timedelta(seconds=1.0))
    elapsed = time.monotonic() - start
    assert 0.07 < elapsed < 0.20, (
        f"speed=10 must compress 1.0s sim to ~0.10s wall; got {elapsed:.3f}s"
    )


@pytest.mark.asyncio
async def test_slow_motion_expands_time() -> None:
    """speed=0.5 → 0.05s sim delta should land in ~0.10s wall (slow-mo)."""
    throttle = PlaybackThrottle(0.5)
    await throttle.wait_for(BASE)
    start = time.monotonic()
    await throttle.wait_for(BASE + timedelta(milliseconds=50))
    elapsed = time.monotonic() - start
    assert 0.07 < elapsed < 0.20, (
        f"speed=0.5 should expand 0.05s sim to ~0.10s wall; got {elapsed:.3f}s"
    )


@pytest.mark.asyncio
async def test_no_sleep_if_already_behind_wall_clock() -> None:
    """If the caller is slow (real wall > target wall), wait_for must return
    immediately rather than sleep a negative duration."""
    throttle = PlaybackThrottle(100.0)  # very fast — target windows are tiny
    await throttle.wait_for(BASE)
    # Force the caller to be late: do a real sleep BEFORE asking for the
    # next target. By the time we ask, wall has already advanced past where
    # the throttle wanted us.
    time.sleep(0.05)  # 50ms wall passed
    start = time.monotonic()
    # At speed=100, the throttle's target for sim+1ms is wall + 0.01ms — way
    # in the past. Must return immediately.
    await throttle.wait_for(BASE + timedelta(milliseconds=1))
    elapsed = time.monotonic() - start
    assert elapsed < 0.02, (
        f"behind-schedule caller must not sleep; took {elapsed:.3f}s"
    )


def test_negative_speed_rejected() -> None:
    with pytest.raises(ValueError, match="speed must be >= 0"):
        PlaybackThrottle(-1.0)


def test_enabled_flag_reflects_speed() -> None:
    assert PlaybackThrottle(0.0).enabled is False
    assert PlaybackThrottle(0.001).enabled is True
    assert PlaybackThrottle(1.0).enabled is True
    assert PlaybackThrottle(1000.0).enabled is True


def test_speed_property_returns_constructor_value() -> None:
    assert PlaybackThrottle(0.0).speed == 0.0
    assert PlaybackThrottle(2.5).speed == 2.5
