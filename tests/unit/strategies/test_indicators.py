from __future__ import annotations

import pytest

from stinger_fx.strategies.indicators import ema, rsi, rsi_series, sma


def test_sma_returns_none_when_not_enough_data() -> None:
    assert sma([1.0, 2.0, 3.0], 5) is None


def test_rsi_series_matches_pointwise_rsi() -> None:
    """rsi_series is the O(n) backbone of stoch_rsi — it must produce the
    exact same value at every index that calling rsi(values[:end]) would,
    otherwise the optimisation silently changes Stoch RSI output."""
    closes = [100.0 + (i % 7) - (i % 3) * 0.5 + i * 0.1 for i in range(60)]
    series = rsi_series(closes, 14)
    # Length: one value per index from `period` (14) to the last.
    assert len(series) == len(closes) - 14
    # Last element == plain rsi over the full series.
    assert series[-1] == pytest.approx(rsi(closes, 14))
    # Each tail element matches rsi over the truncated prefix.
    for offset in range(1, 6):
        end = len(closes) - offset + 1
        assert series[-offset] == pytest.approx(rsi(closes[:end], 14))


def test_rsi_series_empty_when_insufficient_data() -> None:
    assert rsi_series([1.0, 2.0, 3.0], 14) == []
    assert rsi_series([], 14) == []


def test_sma_simple() -> None:
    assert sma([1.0, 2.0, 3.0, 4.0, 5.0], 5) == 3.0
    assert sma([1.0, 2.0, 3.0, 4.0, 5.0], 3) == 4.0


def test_ema_returns_none_when_not_enough_data() -> None:
    assert ema([1.0, 2.0], 5) is None


def test_ema_weighting_recent_values() -> None:
    flat = [10.0] * 10
    assert ema(flat, 5) == pytest.approx(10.0)
    # Step jump on the tail — EMA should lean toward the recent values more
    # than SMA does over a series that's flat for most of the window.
    series = [10.0] * 10 + [20.0] * 3
    e = ema(series, 5)
    s = sma(series, 5)
    assert e is not None and s is not None
    assert e > s


def test_rsi_extremes() -> None:
    # Continuously rising → RSI close to 100
    rising = [float(i) for i in range(1, 20)]
    assert rsi(rising, 14) == pytest.approx(100.0)
    # Continuously falling → RSI close to 0
    falling = [float(i) for i in range(20, 1, -1)]
    assert rsi(falling, 14) == pytest.approx(0.0, abs=1.0)


def test_rsi_requires_more_than_period() -> None:
    assert rsi([1.0] * 14, 14) is None
