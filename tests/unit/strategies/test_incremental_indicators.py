"""Incremental indicators must reproduce the pure-function output bit-for-bit.

Fed a series one value at a time, each streaming class must equal the
corresponding batch indicator computed over the same prefix at *every* step —
including returning ``None`` on exactly the same (warmup) bars.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from stinger_fx.domain import Bar, Timeframe
from stinger_fx.strategies.indicators import adx, rsi, stoch_rsi
from stinger_fx.strategies.indicators.incremental import (
    IncrementalADX,
    IncrementalATR,
    IncrementalRSI,
    IncrementalStochRSI,
)


def _rsi_ref(values: list[float], period: int) -> float | None:
    """Un-capped full-history Wilder RSI (the production rsi() caps its input)."""
    if len(values) <= period:
        return None
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = values[i] - values[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    ag, al = gains / period, losses / period
    for i in range(period + 1, len(values)):
        d = values[i] - values[i - 1]
        ag = (ag * (period - 1) + max(d, 0.0)) / period
        al = (al * (period - 1) + max(-d, 0.0)) / period
    if al == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + ag / al)


def _atr_ref(bars: list[Bar], period: int) -> float | None:
    if len(bars) <= period:
        return None
    trs: list[float] = []
    pc = bars[0].close
    for b in bars[1:]:
        trs.append(max(b.high - b.low, abs(b.high - pc), abs(b.low - pc)))
        pc = b.close
    out = sum(trs[:period]) / period
    for tr in trs[period:]:
        out = (out * (period - 1) + tr) / period
    return out


def _series(n: int, seed: int) -> list[float]:
    import random

    rng = random.Random(seed)
    px = 2000.0
    out = []
    for _ in range(n):
        px += rng.uniform(-3.0, 3.0)
        out.append(px)
    return out


def _bars(closes: list[float]) -> list[Bar]:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Bar(symbol="X", timeframe=Timeframe.M1, time=t0 + timedelta(minutes=i),
            open=c, high=c + 0.7, low=c - 0.7, close=c, tick_volume=1, is_closed=True)
        for i, c in enumerate(closes)
    ]


@pytest.mark.parametrize("period", [2, 14, 50])
def test_incremental_rsi_matches_full_wilder_every_step(period: int) -> None:
    series = _series(400, seed=period)
    inc = IncrementalRSI(period)
    for i, x in enumerate(series):
        got = inc.update(x)
        assert got == _rsi_ref(series[: i + 1], period)
        assert got == inc.value


@pytest.mark.parametrize("period", [2, 14, 50])
def test_incremental_atr_matches_full_wilder_every_step(period: int) -> None:
    bars = _bars(_series(400, seed=period + 1))
    inc = IncrementalATR(period)
    for i, b in enumerate(bars):
        got = inc.update(b)
        assert got == _atr_ref(bars[: i + 1], period)


def test_incremental_stoch_rsi_matches_batch_every_step() -> None:
    # Series short enough that the production stoch_rsi's internal tail-cap is a
    # no-op, so it equals the full Wilder — then incremental must match it exactly.
    series = _series(300, seed=7)
    inc = IncrementalStochRSI(14, 14, 3, 3)
    for i, x in enumerate(series):
        got = inc.update(x)
        ref = stoch_rsi(series[: i + 1], 14, 14, 3, 3)
        if ref is None:
            assert got is None
        else:
            assert got is not None
            assert got[0] == ref.k
            assert got[1] == ref.d


def test_incremental_rsi_within_ulp_of_capped_production_on_long_series() -> None:
    # On a history far past the cap the production rsi() reseeds on a moving
    # window while the incremental carries one seed → they differ, but only at
    # the ~1-ULP level (no economic meaning).
    series = _series(3000, seed=123)
    inc = IncrementalRSI(14)
    for x in series:
        inc.update(x)
    assert inc.value == pytest.approx(rsi(series, 14), abs=1e-9)


def test_incremental_rsi_value_is_none_before_warm() -> None:
    inc = IncrementalRSI(14)
    for x in _series(14, seed=1):  # 14 values → 13 deltas < period
        assert inc.update(x) is None
    assert inc.value is None


@pytest.mark.parametrize("period", [5, 14])
def test_incremental_adx_matches_batch_every_step(period: int) -> None:
    # adx() is NOT tail-capped, so adx(bars[:i+1]) is the full from-start
    # computation — the streaming class anchors on the same seed → bit-identical
    # (all three fields), at every step including the warmup Nones.
    bars = _bars(_series(400, seed=period * 7))
    inc = IncrementalADX(period)
    for i, b in enumerate(bars):
        got = inc.update(b)
        ref = adx(bars[: i + 1], period)
        if ref is None:
            assert got is None
        else:
            assert got is not None
            assert got.adx == ref.adx
            assert got.plus_di == ref.plus_di
            assert got.minus_di == ref.minus_di


def test_incremental_adx_flat_seed_stays_none_like_batch() -> None:
    # A flat (zero-range) seed window makes the batch return None and keep
    # returning None (it re-seeds the same flat window); the streaming class
    # latches dead to match, even once a live tail arrives.
    period = 14
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    flat = [
        Bar(symbol="X", timeframe=Timeframe.M1, time=t0 + timedelta(minutes=i),
            open=2000.0, high=2000.0, low=2000.0, close=2000.0,
            tick_volume=1, is_closed=True)
        for i in range(period + 2)
    ]
    tail = _bars(_series(200, seed=3))
    for i in range(len(tail)):
        tail[i] = tail[i].model_copy(
            update={"time": t0 + timedelta(minutes=len(flat) + i)}
        )
    inc = IncrementalADX(period)
    for b in flat + tail:
        inc.update(b)
    assert inc.value is None
    assert adx(flat + tail, period) is None
