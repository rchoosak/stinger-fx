"""iter_ticks() — Parquet → Tick iterator for tick-level backtest replay."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stinger_fx.data import iter_ticks
from stinger_fx.data.parquet_store import ParquetStore
from stinger_fx.domain import Tick


def _seed_ticks(root: Path, symbol: str, base: datetime, count: int) -> None:
    store = ParquetStore(root)
    ticks = [
        Tick(
            symbol=symbol,
            time=base + timedelta(milliseconds=100 * i),
            bid=1.10 + i * 1e-5,
            ask=1.10 + i * 1e-5 + 2e-5,
        )
        for i in range(count)
    ]
    store.append_ticks(symbol, ticks)


def test_iter_ticks_yields_in_time_order(tmp_path: Path) -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    _seed_ticks(tmp_path, "EURUSD", base, count=10)
    out = list(iter_ticks(tmp_path, "EURUSD", base, base + timedelta(seconds=10)))
    assert len(out) == 10
    times = [t.time for t in out]
    assert times == sorted(times)


def test_iter_ticks_respects_date_range(tmp_path: Path) -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    _seed_ticks(tmp_path, "EURUSD", base, count=20)
    # Pick only the middle slice — 100ms apart, ticks 5..14 fall in [500ms..1500ms)
    out = list(
        iter_ticks(
            tmp_path,
            "EURUSD",
            base + timedelta(milliseconds=500),
            base + timedelta(milliseconds=1500),
        )
    )
    assert 0 < len(out) <= 20
    for t in out:
        assert base + timedelta(milliseconds=500) <= t.time < base + timedelta(milliseconds=1500)


def test_iter_ticks_empty_when_no_data(tmp_path: Path) -> None:
    out = list(iter_ticks(tmp_path, "EURUSD", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC)))
    assert out == []


def test_iter_ticks_carries_bid_ask(tmp_path: Path) -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    _seed_ticks(tmp_path, "EURUSD", base, count=3)
    out = list(iter_ticks(tmp_path, "EURUSD", base, base + timedelta(seconds=1)))
    assert all(t.symbol == "EURUSD" for t in out)
    assert all(t.ask > t.bid for t in out)
