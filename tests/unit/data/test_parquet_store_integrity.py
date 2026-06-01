from __future__ import annotations

from datetime import UTC, datetime

import pytest

from stinger_fx.data import parquet_store
from stinger_fx.data.parquet_store import ParquetStore
from stinger_fx.domain import Bar, Tick, Timeframe


def _tick(second: int, *, bid: float = 1.1, ask: float = 1.1002) -> Tick:
    return Tick(
        symbol="EURUSD",
        time=datetime(2024, 1, 1, 0, 0, second, tzinfo=UTC),
        bid=bid,
        ask=ask,
    )


def _bar(minute: int, *, close: float = 1.1) -> Bar:
    high = max(1.0, close)
    low = min(1.0, close)
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        time=datetime(2024, 1, 1, 0, minute, tzinfo=UTC),
        open=1.0,
        high=high,
        low=low,
        close=close,
        tick_volume=1,
    )


def test_append_ticks_dedupes_and_reads_sorted(tmp_path) -> None:
    store = ParquetStore(tmp_path)
    later = _tick(2, bid=1.2, ask=1.2002)
    earlier = _tick(1)

    store.append_ticks("EURUSD", [later, earlier, earlier])
    store.append_ticks("EURUSD", [earlier])

    table = store.read_ticks(
        "EURUSD",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
    )
    rows = table.to_pylist()
    assert [r["time_ns"].second for r in rows] == [1, 2]
    assert store.duplicate_count("EURUSD", Timeframe.TICK, datetime(2024, 1, 1, tzinfo=UTC).date()) == 0


def test_append_bars_dedupes_by_time_and_keeps_last(tmp_path) -> None:
    store = ParquetStore(tmp_path)

    store.append_bars("EURUSD", Timeframe.M1, [_bar(1, close=1.1)])
    store.append_bars("EURUSD", Timeframe.M1, [_bar(1, close=1.2), _bar(0, close=1.0)])

    table = store.read_bars(
        "EURUSD",
        Timeframe.M1,
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
    )
    rows = table.to_pylist()
    assert [r["time"].minute for r in rows] == [0, 1]
    assert rows[1]["close"] == pytest.approx(1.2)
    assert store.duplicate_count("EURUSD", Timeframe.M1, datetime(2024, 1, 1, tzinfo=UTC).date()) == 0


def test_failed_atomic_write_leaves_existing_partition_intact(monkeypatch, tmp_path) -> None:
    store = ParquetStore(tmp_path)
    store.append_bars("EURUSD", Timeframe.M1, [_bar(0, close=1.1)])

    path = tmp_path / "EURUSD" / "M1" / "2024-01-01.parquet"
    before = path.read_bytes()

    def boom(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(parquet_store.pq, "write_table", boom)

    with pytest.raises(RuntimeError, match="disk full"):
        store.append_bars("EURUSD", Timeframe.M1, [_bar(1, close=1.2)])

    assert path.read_bytes() == before
