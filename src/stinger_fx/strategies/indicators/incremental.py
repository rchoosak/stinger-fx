"""Incremental (streaming) Wilder indicators — O(1) per bar.

The pure functions ``rsi`` / ``atr`` / ``stoch_rsi`` recompute from scratch over
the whole history each call, which is O(window) per bar. These stateful classes
carry the Wilder running state forward and update in O(1) per closed bar — the
same arithmetic, in the same order, so fed a series one value at a time each
class reproduces the corresponding pure function **bit-for-bit** at every step
(see ``test_incremental_indicators.py``).

Usage (one instance per (symbol, timeframe) feed; feed only *closed* bars, in
order, so there's no lookahead):

    rsi = IncrementalRSI(14)
    for bar in closed_bars:
        v = rsi.update(bar.close)   # None until warm, then the RSI

``value`` re-reads the last result without advancing state.

Note vs the production pure functions: those cap their input to the last
``50*period`` bars (``_smoothing.wilder_tail``), so on a *windowed* history far
past the cap they reseed on a moving window while these carry one continuous
seed — the two then differ at the ~1-ULP level (economically irrelevant, but not
bit-identical). These classes match the *un-capped* full-history Wilder exactly.
"""

from __future__ import annotations

from collections import deque

from stinger_fx.domain import Bar


class IncrementalRSI:
    """Wilder's RSI, streamed. Matches ``rsi(values[: i + 1], period)``."""

    __slots__ = ("_avg_gain", "_avg_loss", "_n", "_period", "_prev", "_value")

    def __init__(self, period: int = 14) -> None:
        if period <= 0:
            raise ValueError("period must be > 0")
        self._period = period
        self._prev: float | None = None
        self._n = 0  # number of deltas seen
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._value: float | None = None

    def update(self, x: float) -> float | None:
        p = self._period
        if self._prev is None:
            self._prev = x
            return None
        delta = x - self._prev
        self._prev = x
        gain = delta if delta > 0.0 else 0.0
        loss = -delta if delta < 0.0 else 0.0
        self._n += 1
        if self._n < p:
            # Accumulate the seed sum in _avg_* (divided once the seed is full).
            self._avg_gain += gain
            self._avg_loss += loss
            return None
        if self._n == p:
            self._avg_gain = (self._avg_gain + gain) / p
            self._avg_loss = (self._avg_loss + loss) / p
        else:
            self._avg_gain = (self._avg_gain * (p - 1) + gain) / p
            self._avg_loss = (self._avg_loss * (p - 1) + loss) / p
        self._value = (
            100.0
            if self._avg_loss == 0.0
            else 100.0 - 100.0 / (1.0 + self._avg_gain / self._avg_loss)
        )
        return self._value

    @property
    def value(self) -> float | None:
        return self._value


class IncrementalATR:
    """Wilder's ATR, streamed. Matches ``atr(bars[: i + 1], period)``."""

    __slots__ = ("_atr", "_n", "_period", "_prev_close", "_value")

    def __init__(self, period: int = 14) -> None:
        if period <= 0:
            raise ValueError("period must be > 0")
        self._period = period
        self._prev_close: float | None = None
        self._n = 0  # number of TRs seen
        self._atr = 0.0
        self._value: float | None = None

    def update(self, bar: Bar) -> float | None:
        p = self._period
        if self._prev_close is None:
            self._prev_close = bar.close
            return None
        tr = max(
            bar.high - bar.low,
            abs(bar.high - self._prev_close),
            abs(bar.low - self._prev_close),
        )
        self._prev_close = bar.close
        self._n += 1
        if self._n < p:
            self._atr += tr  # seed sum
            return None
        if self._n == p:
            self._atr = (self._atr + tr) / p
        else:
            self._atr = (self._atr * (p - 1) + tr) / p
        self._value = self._atr
        return self._value

    @property
    def value(self) -> float | None:
        return self._value


class IncrementalStochRSI:
    """Stochastic RSI (%K smoothed, %D), streamed. Matches ``stoch_rsi``.

    Returns ``(k, d)`` once warm, else ``None``.
    """

    __slots__ = ("_d", "_k", "_rawk", "_rsi", "_rsi_win", "_sp", "_value")

    def __init__(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_smooth: int = 3,
        d_smooth: int = 3,
    ) -> None:
        if min(rsi_period, stoch_period, k_smooth, d_smooth) < 1:
            raise ValueError("all periods must be >= 1")
        self._sp = stoch_period
        self._k = k_smooth
        self._d = d_smooth
        self._rsi = IncrementalRSI(rsi_period)
        self._rsi_win: deque[float] = deque(maxlen=stoch_period)
        # Batch stoch_rsi computes the last (k_smooth + d_smooth) raw %K but only
        # smooths over the last (k_smooth + d_smooth - 1) of them (the oldest is
        # discarded). Keep the same k+d window so warmup fires on the identical
        # bar — the oldest entry is intentionally never used.
        self._rawk: deque[float] = deque(maxlen=k_smooth + d_smooth)
        self._value: tuple[float, float] | None = None

    def update(self, close: float) -> tuple[float, float] | None:
        r = self._rsi.update(close)
        if r is None:
            return None
        self._rsi_win.append(r)
        if len(self._rsi_win) < self._sp:
            return None
        lo = min(self._rsi_win)
        hi = max(self._rsi_win)
        raw_k = 50.0 if hi == lo else (r - lo) / (hi - lo) * 100.0
        self._rawk.append(raw_k)
        if len(self._rawk) < self._k + self._d:
            return None
        rawk = list(self._rawk)
        # Mirror the batch: smooth over rawk[1:], i.e. windows rawk[1+j : 1+j+k]
        # for j in range(d_smooth) — the oldest reading (rawk[0]) is dropped.
        smoothed = [
            sum(rawk[1 + j : 1 + j + self._k]) / self._k for j in range(self._d)
        ]
        k = smoothed[-1]
        d = sum(smoothed) / self._d
        self._value = (k, d)
        return self._value

    @property
    def value(self) -> tuple[float, float] | None:
        return self._value
