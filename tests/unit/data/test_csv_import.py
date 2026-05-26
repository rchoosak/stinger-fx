"""CSV → Parquet tick importer — round-trip + edge cases."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from stinger_fx.data import iter_ticks
from stinger_fx.data.csv_import import CsvImportError, import_ticks_csv


def _write_csv(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_csv_round_trip_iso_timestamps(tmp_path: Path) -> None:
    """An ISO-8601 tick CSV imports into Parquet and round-trips via iter_ticks."""
    csv = tmp_path / "eurusd.csv"
    _write_csv(
        csv,
        [
            "time,bid,ask",
            "2024-01-15T00:00:00Z,1.10000,1.10002",
            "2024-01-15T00:00:01Z,1.10001,1.10003",
            "2024-01-15T00:00:02Z,1.10002,1.10004",
        ],
    )
    parquet_root = tmp_path / "parquet"
    n = import_ticks_csv(
        csv,
        symbol="EURUSD",
        parquet_root=parquet_root,
    )
    assert n == 3

    out = list(
        iter_ticks(
            parquet_root,
            "EURUSD",
            datetime(2024, 1, 15, tzinfo=UTC),
            datetime(2024, 1, 16, tzinfo=UTC),
        )
    )
    assert len(out) == 3
    assert [t.bid for t in out] == [1.10000, 1.10001, 1.10002]
    assert [t.ask for t in out] == [1.10002, 1.10003, 1.10004]
    # All ticks must be tz-aware UTC after import
    assert all(t.time.tzinfo is not None for t in out)


def test_csv_naive_timestamps_attach_tz(tmp_path: Path) -> None:
    """Naive 'YYYY-MM-DD HH:MM:SS' timestamps get interpreted in --tz."""
    csv = tmp_path / "ticks.csv"
    _write_csv(
        csv,
        [
            "ts,bid,ask",
            "2024-01-15 09:00:00,1.10,1.10002",
            "2024-01-15 09:00:01,1.11,1.11002",
        ],
    )
    parquet_root = tmp_path / "parquet"
    n = import_ticks_csv(
        csv,
        symbol="EURUSD",
        parquet_root=parquet_root,
        time_col="ts",
        tz="America/New_York",  # 09:00 NY = 14:00 UTC in winter
    )
    assert n == 2

    out = list(
        iter_ticks(
            parquet_root,
            "EURUSD",
            datetime(2024, 1, 15, tzinfo=UTC),
            datetime(2024, 1, 16, tzinfo=UTC),
        )
    )
    assert len(out) == 2
    # Jan 15 09:00 in America/New_York = 14:00 UTC (EST, UTC-5)
    assert out[0].time == datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)


def test_csv_missing_required_column_errors(tmp_path: Path) -> None:
    """Missing time/bid/ask columns raise CsvImportError with a useful message."""
    csv = tmp_path / "broken.csv"
    _write_csv(csv, ["time,bid", "2024-01-15T00:00:00Z,1.10"])  # ask missing
    parquet_root = tmp_path / "parquet"
    with pytest.raises(CsvImportError, match="ask"):
        import_ticks_csv(csv, symbol="EURUSD", parquet_root=parquet_root)


def test_csv_optional_columns_pass_through(tmp_path: Path) -> None:
    """When the CSV has last/volume/flags and they're mapped, they survive the
    round-trip; when they're missing, the importer fills sane defaults."""
    csv = tmp_path / "rich.csv"
    _write_csv(
        csv,
        [
            "time,bid,ask,last,volume,flags",
            "2024-01-15T00:00:00Z,1.10000,1.10002,1.10001,5,2",
        ],
    )
    parquet_root = tmp_path / "parquet"
    n = import_ticks_csv(
        csv,
        symbol="EURUSD",
        parquet_root=parquet_root,
        last_col="last",
        volume_col="volume",
        flags_col="flags",
    )
    assert n == 1
    out = list(
        iter_ticks(
            parquet_root,
            "EURUSD",
            datetime(2024, 1, 15, tzinfo=UTC),
            datetime(2024, 1, 16, tzinfo=UTC),
        )
    )
    assert len(out) == 1
    t = out[0]
    assert t.bid == 1.10000
    assert t.ask == 1.10002
    assert t.last == 1.10001
    assert t.volume == 5
    assert t.flags == 2


def test_csv_dukascopy_shape(tmp_path: Path) -> None:
    """Dukascopy-style column names: Local time, Ask, Bid."""
    csv = tmp_path / "dukascopy.csv"
    _write_csv(
        csv,
        [
            "Local time,Ask,Bid,AskVolume,BidVolume",
            "2024-01-15T00:00:00Z,1.10002,1.10000,1.5,1.2",
            "2024-01-15T00:00:01Z,1.10003,1.10001,1.5,1.2",
        ],
    )
    parquet_root = tmp_path / "parquet"
    n = import_ticks_csv(
        csv,
        symbol="EURUSD",
        parquet_root=parquet_root,
        time_col="Local time",
        bid_col="Bid",
        ask_col="Ask",
    )
    assert n == 2
    out = list(
        iter_ticks(
            parquet_root,
            "EURUSD",
            datetime(2024, 1, 15, tzinfo=UTC),
            datetime(2024, 1, 16, tzinfo=UTC),
        )
    )
    assert [t.bid for t in out] == [1.10000, 1.10001]
    assert [t.ask for t in out] == [1.10002, 1.10003]
