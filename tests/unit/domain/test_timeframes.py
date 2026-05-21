from __future__ import annotations

import pytest

from stinger_fx.domain import Timeframe


def test_seconds_for_native_timeframes() -> None:
    assert Timeframe.M1.seconds == 60
    assert Timeframe.M5.seconds == 300
    assert Timeframe.H1.seconds == 3600
    assert Timeframe.D1.seconds == 86_400


def test_tick_has_no_duration() -> None:
    with pytest.raises(ValueError):
        _ = Timeframe.TICK.seconds


def test_synthetic_timeframes_not_native_to_mt5() -> None:
    for tf in (Timeframe.M2, Timeframe.M3, Timeframe.M10, Timeframe.M45):
        assert tf.is_native_mt5 is False
    for tf in (Timeframe.M1, Timeframe.M15, Timeframe.H1):
        assert tf.is_native_mt5 is True
