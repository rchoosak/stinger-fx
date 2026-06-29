"""Tail-capping the Wilder-smoothed indicators (rsi / atr / stoch_rsi) must be
**bit-identical** to the full-window computation.

Each of these is a recursive Wilder average; the influence of a value k steps
back decays as (1-1/period)^k, so beyond ~37*period steps it's below a float64
ULP and the leading inputs can't change the result. The production functions
truncate long inputs to the last ``WILDER_TAIL_FACTOR * period`` (=50*period)
elements for speed — these tests assert that truncation doesn't move a single
bit versus a reference implementation that consumes the whole series.
"""
from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from stinger_fx.domain import Bar, Timeframe
from stinger_fx.strategies.indicators import adx, atr, ema, rsi, stoch_rsi
from stinger_fx.strategies.indicators._smoothing import (
    ADX_TAIL_FACTOR,
    EMA_TAIL_FACTOR,
    WILDER_TAIL_FACTOR,
)
from stinger_fx.strategies.indicators.rsi import rsi_series

# --- reference (uncapped) implementations -------------------------------- #

def _rsi_ref(values: list[float], period: int) -> float | None:
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


def _bars(closes: list[float]) -> list[Bar]:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    out = []
    for i, c in enumerate(closes):
        out.append(Bar(symbol="X", timeframe=Timeframe.M1, time=t0 + timedelta(minutes=i),
                       open=c, high=c + 0.6, low=c - 0.6, close=c,
                       tick_volume=1, is_closed=True))
    return out


def _series(n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    px = 2000.0
    out = []
    for _ in range(n):
        px += rng.uniform(-3.0, 3.0)
        out.append(px)
    return out


@pytest.mark.parametrize("period", [2, 14, 50])
def test_rsi_tail_cap_bit_identical(period: int) -> None:
    # Series comfortably longer than the cap so truncation actually engages.
    series = _series(period * WILDER_TAIL_FACTOR + 800, seed=period)
    assert rsi(series, period) == _rsi_ref(series, period)


@pytest.mark.parametrize("period", [2, 14, 50])
def test_atr_tail_cap_bit_identical(period: int) -> None:
    bars = _bars(_series(period * WILDER_TAIL_FACTOR + 800, seed=period + 1))
    assert atr(bars, period) == _atr_ref(bars, period)


def test_stoch_rsi_tail_cap_bit_identical() -> None:
    # stoch_rsi caps its `closes` input, then reads only the last `extra+1`
    # values of rsi_series(closes). The cap is correct iff that consumed tail is
    # bit-identical to the one from the full (un-capped) RSI series — prove that
    # directly (rsi_series itself is never capped), no stoch math duplicated.
    rp = 14
    extra = 14 + 3 + 3 - 2
    series = _series(rp * WILDER_TAIL_FACTOR + 800, seed=7)
    cap = max(rp * WILDER_TAIL_FACTOR, rp + extra + 1)
    full_tail = rsi_series(series, rp)[-(extra + 1):]
    capped_tail = rsi_series(series[-cap:], rp)[-(extra + 1):]
    assert full_tail == capped_tail
    # And the public function (which caps internally) is self-consistent.
    assert stoch_rsi(series) == stoch_rsi(series[-cap:])


def test_short_input_is_unaffected_by_cap() -> None:
    # Below the cap → no truncation, identical to reference (regression guard).
    series = _series(120, seed=99)
    assert rsi(series, 14) == _rsi_ref(series, 14)
    bars = _bars(series)
    assert atr(bars, 14) == _atr_ref(bars, 14)


# --- EMA + ADX (added with their own factors) ---------------------------- #

def _ema_ref(values: list[float], period: int) -> float | None:
    """Un-capped EMA (the production ema() caps its input)."""
    if len(values) < period:
        return None
    alpha = 2.0 / (period + 1)
    out = sum(values[:period]) / period
    for v in values[period:]:
        out = alpha * v + (1 - alpha) * out
    return out


def _adx_ref(bars: list[Bar], period: int) -> float | None:
    """Un-capped ADX — mirrors adx() without the trailing-window cap."""
    if len(bars) < 2 * period:
        return None
    trs: list[float] = []
    plus_dms: list[float] = []
    minus_dms: list[float] = []
    for i in range(1, len(bars)):
        prev, curr = bars[i - 1], bars[i]
        up = curr.high - prev.high
        down = prev.low - curr.low
        plus_dms.append(up if (up > down and up > 0) else 0.0)
        minus_dms.append(down if (down > up and down > 0) else 0.0)
        trs.append(max(curr.high - curr.low, abs(curr.high - prev.close),
                       abs(curr.low - prev.close)))
    tr_s = sum(trs[:period])
    plus_s = sum(plus_dms[:period])
    minus_s = sum(minus_dms[:period])
    dxs: list[float] = []
    def _dx(ps: float, ms: float, ts: float) -> float:
        if ts == 0:
            return 0.0
        pd, md = 100 * ps / ts, 100 * ms / ts
        s = pd + md
        return 100 * abs(pd - md) / s if s > 0 else 0.0
    dxs.append(_dx(plus_s, minus_s, tr_s))
    for i in range(period, len(trs)):
        tr_s = tr_s - tr_s / period + trs[i]
        plus_s = plus_s - plus_s / period + plus_dms[i]
        minus_s = minus_s - minus_s / period + minus_dms[i]
        if tr_s == 0:
            continue
        dxs.append(_dx(plus_s, minus_s, tr_s))
    if len(dxs) < period:
        return None
    adx_val = sum(dxs[:period]) / period
    for dx_val in dxs[period:]:
        adx_val = (adx_val * (period - 1) + dx_val) / period
    return adx_val


@pytest.mark.parametrize("period", [14, 20, 50])
def test_ema_tail_cap_bit_identical(period: int) -> None:
    series = _series(period * EMA_TAIL_FACTOR + 800, seed=period + 3)
    assert ema(series, period) == _ema_ref(series, period)


@pytest.mark.parametrize("period", [14, 20])
def test_adx_tail_cap_bit_identical(period: int) -> None:
    # Several seeds so the (data-dependent) convergence margin is exercised.
    for seed in (1, 2, 3):
        bars = _bars(_series(period * ADX_TAIL_FACTOR + 1200, seed=seed))
        got = adx(bars, period)
        ref = _adx_ref(bars, period)
        assert got is not None and ref is not None
        assert got.adx == ref
