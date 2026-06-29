"""HistoryView's streaming indicator accessors must equal the pure functions
computed over the same buffer at every bar (within the tail-cap window, where
both reduce to the full Wilder)."""
from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

from stinger_fx.domain import Bar, Timeframe
from stinger_fx.strategies.context import HistoryView
from stinger_fx.strategies.indicators import atr, rsi, stoch_rsi


def _bars(n: int, seed: int) -> list[Bar]:
    rng = random.Random(seed)
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    px = 2000.0
    out = []
    for i in range(n):
        px += rng.uniform(-3.0, 3.0)
        out.append(Bar(symbol="XAUUSD", timeframe=Timeframe.M1,
                       time=t0 + timedelta(minutes=i), open=px, high=px + 0.7,
                       low=px - 0.7, close=px, tick_volume=1, is_closed=True))
    return out


def test_view_rsi_matches_batch_every_bar() -> None:
    view = HistoryView("XAUUSD", Timeframe.M1)
    for b in _bars(400, seed=1):
        view.append_bar(b)
        assert view.rsi(14) == rsi(view.closes(), 14)


def test_view_atr_matches_batch_every_bar() -> None:
    view = HistoryView("XAUUSD", Timeframe.M1)
    for b in _bars(400, seed=2):
        view.append_bar(b)
        assert view.atr(14) == atr(list(view.bars()), 14)


def test_view_stoch_rsi_matches_batch_every_bar() -> None:
    view = HistoryView("XAUUSD", Timeframe.M1)
    for b in _bars(300, seed=3):
        view.append_bar(b)
        got = view.stoch_rsi(14, 14, 3, 3)
        ref = stoch_rsi(view.closes(), 14, 14, 3, 3)
        if ref is None:
            assert got is None
        else:
            assert got == (ref.k, ref.d)


def test_view_indicator_lazy_created_after_warmup_still_matches() -> None:
    # First request happens mid-stream (bar 250). Back-fill from the buffer must
    # still produce the same value as the batch over the whole buffer.
    view = HistoryView("XAUUSD", Timeframe.M1)
    bars = _bars(300, seed=4)
    for b in bars[:250]:
        view.append_bar(b)
    assert view.rsi(14) == rsi(view.closes(), 14)  # first call → back-fill
    for b in bars[250:]:
        view.append_bar(b)
        assert view.rsi(14) == rsi(view.closes(), 14)  # streamed thereafter
