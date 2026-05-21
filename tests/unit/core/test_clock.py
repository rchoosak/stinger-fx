from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from stinger_fx.core import LiveClock, SimClock


def test_live_clock_returns_utc() -> None:
    now = LiveClock().now()
    assert now.tzinfo is not None
    assert now.tzinfo.utcoffset(now) == timedelta(0)


def test_sim_clock_requires_tz() -> None:
    with pytest.raises(ValueError):
        SimClock(datetime(2024, 1, 1))


def test_sim_clock_cannot_run_backwards() -> None:
    clock = SimClock(datetime(2024, 1, 1, tzinfo=UTC))
    clock.advance(datetime(2024, 1, 2, tzinfo=UTC))
    with pytest.raises(ValueError):
        clock.advance(datetime(2024, 1, 1, 12, tzinfo=UTC))


def test_sim_clock_advances_monotonically() -> None:
    clock = SimClock(datetime(2024, 1, 1, tzinfo=UTC))
    target = datetime(2024, 1, 5, 12, 30, tzinfo=UTC)
    clock.advance(target)
    assert clock.now() == target
