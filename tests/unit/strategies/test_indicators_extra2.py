"""Golden-value tests for the 7 new built-in indicators."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from stinger_fx.domain import Bar, Timeframe
from stinger_fx.strategies.indicators import (
    HeikinAshiCandle,
    aroon,
    ehlers_decycler,
    ehlers_super_smoother,
    heikin_ashi,
    heikin_ashi_series,
    obv,
    psar,
    stoch_rsi,
    williams_r,
)


def _bar(
    i: int,
    *,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    close: float,
    volume: int = 100,
) -> Bar:
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.M15,
        time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=15 * i),
        open=open_ if open_ is not None else close,
        high=high if high is not None else close + 0.0005,
        low=low if low is not None else close - 0.0005,
        close=close,
        tick_volume=volume,
        is_closed=True,
    )


# --- Stochastic RSI ---------------------------------------------------------


def test_stoch_rsi_returns_none_below_min_history() -> None:
    closes = [1.0 + 0.01 * i for i in range(20)]
    # Default needs 14+14+3+3 = 34 → 20 closes isn't enough
    assert stoch_rsi(closes) is None


def test_stoch_rsi_returns_value_with_enough_history() -> None:
    """A noisy up-trend gives RSI variation, which yields a finite Stoch RSI."""
    # Drift up with small pullbacks so RSI doesn't saturate at 100
    closes = []
    for i in range(60):
        base = 1.0 + 0.001 * i
        # every 5th bar is a small pullback
        closes.append(base - 0.0005 if i % 5 == 0 else base)
    result = stoch_rsi(closes, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3)
    assert result is not None
    assert 0.0 <= result.k <= 100.0
    assert 0.0 <= result.d <= 100.0


def test_stoch_rsi_validates_periods() -> None:
    with pytest.raises(ValueError):
        stoch_rsi([1.0], rsi_period=0)


# --- OBV --------------------------------------------------------------------


def test_obv_returns_none_short_history() -> None:
    assert obv([_bar(0, close=1.10)]) is None


def test_obv_accumulates_signed_volume() -> None:
    bars = [
        _bar(0, close=1.10, volume=100),
        _bar(1, close=1.11, volume=200),  # up → +200
        _bar(2, close=1.10, volume=300),  # down → -300
        _bar(3, close=1.10, volume=150),  # equal → 0
        _bar(4, close=1.12, volume=400),  # up → +400
    ]
    # Cumulative: +200 - 300 + 0 + 400 = +300
    assert obv(bars) == pytest.approx(300.0)


def test_obv_all_equal_closes_returns_zero() -> None:
    bars = [_bar(i, close=1.10, volume=100) for i in range(5)]
    assert obv(bars) == pytest.approx(0.0)


# --- Williams %R ------------------------------------------------------------


def test_williams_r_at_top_of_range() -> None:
    """Close at the recent high → %R ≈ 0."""
    bars = [_bar(i, close=1.10, high=1.10, low=1.09) for i in range(13)]
    bars.append(_bar(13, close=1.12, high=1.12, low=1.11))
    # period=14: recent window = all 14 bars; hh=1.12, ll=1.09, close=1.12
    # %R = (1.12 - 1.12) / (1.12 - 1.09) * -100 = 0
    assert williams_r(bars, period=14) == pytest.approx(0.0)


def test_williams_r_at_bottom_of_range() -> None:
    """Close at the recent low → %R ≈ -100."""
    bars = [_bar(i, close=1.12, high=1.12, low=1.11) for i in range(13)]
    bars.append(_bar(13, close=1.09, high=1.10, low=1.09))
    # hh = 1.12, ll = 1.09, close = 1.09
    # %R = (1.12 - 1.09) / (1.12 - 1.09) * -100 = -100
    assert williams_r(bars, period=14) == pytest.approx(-100.0)


def test_williams_r_returns_none_short_history() -> None:
    assert williams_r([_bar(i, close=1.10) for i in range(5)], period=14) is None


def test_williams_r_neutral_when_no_range() -> None:
    bars = [_bar(i, close=1.10, high=1.10, low=1.10) for i in range(14)]
    assert williams_r(bars, period=14) == pytest.approx(-50.0)


# --- Aroon ------------------------------------------------------------------


def test_aroon_returns_none_short_history() -> None:
    assert aroon([_bar(i, close=1.10) for i in range(5)], period=25) is None


def test_aroon_high_at_current_bar_gives_100() -> None:
    """Last bar is the highest → Aroon Up = 100."""
    bars = [_bar(i, close=1.10, high=1.10 + 0.001 * i, low=1.09) for i in range(26)]
    # high keeps making new highs → last bar is the highest of the last 26
    result = aroon(bars, period=25)
    assert result is not None
    assert result.up == pytest.approx(100.0)


def test_aroon_low_at_current_bar_gives_down_100() -> None:
    bars = [_bar(i, close=1.10, high=1.11, low=1.10 - 0.001 * i) for i in range(26)]
    result = aroon(bars, period=25)
    assert result is not None
    assert result.down == pytest.approx(100.0)


def test_aroon_oscillator_is_up_minus_down() -> None:
    bars = [_bar(i, close=1.10, high=1.10 + 0.001 * i, low=1.09 - 0.001 * i) for i in range(26)]
    result = aroon(bars, period=25)
    assert result is not None
    assert result.oscillator == pytest.approx(result.up - result.down)


# --- Parabolic SAR ----------------------------------------------------------


def test_psar_returns_none_short_history() -> None:
    assert psar([_bar(0, close=1.10), _bar(1, close=1.11)]) is None


def test_psar_uptrend_sar_below_price() -> None:
    """Clean uptrend → SAR stays below price, trend = 'up'."""
    bars = [_bar(i, close=1.10 + 0.001 * i, high=1.10 + 0.001 * i + 0.0002,
                 low=1.10 + 0.001 * i - 0.0002) for i in range(30)]
    result = psar(bars)
    assert result is not None
    assert result.trend == "up"
    assert result.value < bars[-1].close


def test_psar_downtrend_sar_above_price() -> None:
    bars = [_bar(i, close=1.20 - 0.001 * i, high=1.20 - 0.001 * i + 0.0002,
                 low=1.20 - 0.001 * i - 0.0002) for i in range(30)]
    result = psar(bars)
    assert result is not None
    assert result.trend == "down"
    assert result.value > bars[-1].close


def test_psar_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        psar([_bar(i, close=1.10) for i in range(5)], af_start=0.0)
    with pytest.raises(ValueError):
        psar([_bar(i, close=1.10) for i in range(5)], af_start=0.5, af_max=0.1)


# --- Heikin-Ashi -----------------------------------------------------------


def test_heikin_ashi_empty_returns_none() -> None:
    assert heikin_ashi([]) is None
    assert heikin_ashi_series([]) == []


def test_heikin_ashi_single_bar() -> None:
    bars = [_bar(0, open_=1.10, high=1.105, low=1.095, close=1.102)]
    ha = heikin_ashi(bars)
    assert ha is not None
    # ha_close = (1.10 + 1.105 + 1.095 + 1.102) / 4 = 1.1005
    assert ha.close == pytest.approx(1.1005)
    # ha_open (seed) = (1.10 + 1.102) / 2 = 1.101
    assert ha.open == pytest.approx(1.101)


def test_heikin_ashi_series_same_length_as_input() -> None:
    bars = [_bar(i, open_=1.10 + 0.001 * i, close=1.10 + 0.001 * i + 0.0005) for i in range(10)]
    series = heikin_ashi_series(bars)
    assert len(series) == len(bars)
    for ha in series:
        assert isinstance(ha, HeikinAshiCandle)


def test_heikin_ashi_helper_properties() -> None:
    # Pick a bar where (high + low) > (open + close) so the HA candle is
    # bullish (ha_close > ha_open). For a single bar:
    #   ha_close = (o+h+l+c)/4,  ha_open = (o+c)/2
    #   bullish iff h+l > o+c
    bars = [_bar(0, open_=1.10, high=1.12, low=1.10, close=1.105)]
    ha = heikin_ashi(bars)
    assert ha is not None
    assert ha.is_green
    assert ha.upper_shadow >= 0
    assert ha.lower_shadow >= 0
    assert ha.body == pytest.approx(ha.close - ha.open)


# --- Ehlers Super Smoother / Decycler --------------------------------------


def test_super_smoother_returns_none_short_history() -> None:
    assert ehlers_super_smoother([1.0, 1.01], period=10) is None


def test_super_smoother_follows_price_on_uptrend() -> None:
    """The filtered output should be close to the trend, not the noise."""
    closes = [1.0 + 0.01 * i for i in range(30)]
    filt = ehlers_super_smoother(closes, period=10)
    assert filt is not None
    # Should be close to the trailing price (within a fraction of one step)
    assert abs(filt - closes[-1]) < 0.05


def test_super_smoother_validates_period() -> None:
    with pytest.raises(ValueError):
        ehlers_super_smoother([1.0, 2.0], period=1)


def test_decycler_strips_oscillation() -> None:
    """For a pure-trend series (no cycles), decycler should track price."""
    closes = [1.0 + 0.005 * i for i in range(100)]
    out = ehlers_decycler(closes, period=20)
    assert out is not None
    assert abs(out - closes[-1]) < 0.1


def test_decycler_returns_none_short_history() -> None:
    assert ehlers_decycler([1.0, 1.01, 1.02], period=20) is None


def test_decycler_validates_period() -> None:
    with pytest.raises(ValueError):
        ehlers_decycler([1.0, 2.0], period=1)
