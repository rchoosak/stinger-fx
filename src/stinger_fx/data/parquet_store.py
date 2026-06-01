"""Parquet store for time-series data.

Layout: `<root>/<symbol>/<timeframe>/<yyyy-mm-dd>.parquet`

Daily partitioning gives cheap range scans and trivial compaction/deletion.
We write through `pyarrow` directly to avoid the pandas overhead on the
hot path; readers return either an arrow Table or a pandas DataFrame depending
on caller.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from stinger_fx.domain import Bar, Tick
from stinger_fx.domain.timeframes import Timeframe

TICK_SCHEMA = pa.schema(
    [
        ("time_ns", pa.timestamp("ns", tz="UTC")),
        ("bid", pa.float64()),
        ("ask", pa.float64()),
        ("last", pa.float64()),
        ("volume", pa.int64()),
        ("flags", pa.int64()),
    ]
)

BAR_SCHEMA = pa.schema(
    [
        ("time", pa.timestamp("s", tz="UTC")),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("tick_volume", pa.int64()),
        ("real_volume", pa.int64()),
        ("spread", pa.int64()),
    ]
)

TICK_DEDUPE_KEYS = ("time_ns", "bid", "ask", "last", "volume", "flags")
BAR_DEDUPE_KEYS = ("time",)


def _utc_date(ts: datetime) -> date:
    if ts.tzinfo is None:
        raise ValueError("timestamp must be tz-aware")
    return ts.astimezone(UTC).date()


class ParquetStore:
    """Append-only writer + ranged reader over the daily-partitioned layout."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    # --- paths ----------------------------------------------------------------

    def _ticks_dir(self, symbol: str) -> Path:
        return self.root / symbol / "TICK"

    def _bars_dir(self, symbol: str, tf: Timeframe) -> Path:
        return self.root / symbol / tf.value

    @staticmethod
    def _day_file(folder: Path, d: date) -> Path:
        return folder / f"{d.isoformat()}.parquet"

    # --- write ---------------------------------------------------------------

    def append_ticks(self, symbol: str, ticks: Iterable[Tick]) -> int:
        rows = [(t.time, t.bid, t.ask, t.last, t.volume, t.flags) for t in ticks]
        if not rows:
            return 0
        # Group by UTC date so each Parquet file holds one calendar day.
        by_day: dict[date, list[tuple]] = {}
        for r in rows:
            by_day.setdefault(_utc_date(r[0]), []).append(r)
        folder = self._ticks_dir(symbol)
        folder.mkdir(parents=True, exist_ok=True)
        n = 0
        for d, day_rows in by_day.items():
            tbl = pa.Table.from_pylist(
                [
                    {
                        "time_ns": r[0],
                        "bid": r[1],
                        "ask": r[2],
                        "last": r[3],
                        "volume": r[4],
                        "flags": r[5],
                    }
                    for r in day_rows
                ],
                schema=TICK_SCHEMA,
            )
            self._append_day(
                self._day_file(folder, d),
                tbl,
                TICK_SCHEMA,
                sort_by="time_ns",
                dedupe_keys=TICK_DEDUPE_KEYS,
            )
            n += len(day_rows)
        return n

    def append_bars(self, symbol: str, tf: Timeframe, bars: Iterable[Bar]) -> int:
        if tf is Timeframe.TICK:
            raise ValueError("use append_ticks for tick data")
        rows = [
            (b.time, b.open, b.high, b.low, b.close, b.tick_volume, b.real_volume, b.spread)
            for b in bars
        ]
        if not rows:
            return 0
        by_day: dict[date, list[tuple]] = {}
        for r in rows:
            by_day.setdefault(_utc_date(r[0]), []).append(r)
        folder = self._bars_dir(symbol, tf)
        folder.mkdir(parents=True, exist_ok=True)
        n = 0
        for d, day_rows in by_day.items():
            tbl = pa.Table.from_pylist(
                [
                    {
                        "time": r[0],
                        "open": r[1],
                        "high": r[2],
                        "low": r[3],
                        "close": r[4],
                        "tick_volume": r[5],
                        "real_volume": r[6],
                        "spread": r[7],
                    }
                    for r in day_rows
                ],
                schema=BAR_SCHEMA,
            )
            self._append_day(
                self._day_file(folder, d),
                tbl,
                BAR_SCHEMA,
                sort_by="time",
                dedupe_keys=BAR_DEDUPE_KEYS,
            )
            n += len(day_rows)
        return n

    @staticmethod
    def _append_day(
        path: Path,
        new_table: pa.Table,
        schema: pa.Schema,
        *,
        sort_by: str,
        dedupe_keys: tuple[str, ...],
    ) -> None:
        if path.exists():
            existing = pq.read_table(path, schema=schema)
            combined = pa.concat_tables([existing, new_table])
        else:
            combined = new_table
        combined = _sort_and_dedupe(combined, schema, sort_by, dedupe_keys)
        _write_table_atomic(combined, path)

    def duplicate_count(self, symbol: str, tf: Timeframe, d: date) -> int:
        """Return duplicate rows beyond the first for one daily partition."""
        path = (
            self._day_file(self._ticks_dir(symbol), d)
            if tf is Timeframe.TICK
            else self._day_file(self._bars_dir(symbol, tf), d)
        )
        if not path.exists():
            return 0
        schema = TICK_SCHEMA if tf is Timeframe.TICK else BAR_SCHEMA
        keys = TICK_DEDUPE_KEYS if tf is Timeframe.TICK else BAR_DEDUPE_KEYS
        table = pq.read_table(path, schema=schema)
        return _duplicate_count(table, keys)

    # --- read ----------------------------------------------------------------

    def read_bars(
        self,
        symbol: str,
        tf: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pa.Table:
        folder = self._bars_dir(symbol, tf)
        if not folder.exists():
            return BAR_SCHEMA.empty_table()
        ds = pads.dataset(folder, format="parquet", schema=BAR_SCHEMA)
        filt = (pads.field("time") >= pa.scalar(start, type=BAR_SCHEMA.field("time").type)) & (
            pads.field("time") < pa.scalar(end, type=BAR_SCHEMA.field("time").type)
        )
        return ds.to_table(filter=filt).sort_by("time")

    def read_ticks(self, symbol: str, start: datetime, end: datetime) -> pa.Table:
        folder = self._ticks_dir(symbol)
        if not folder.exists():
            return TICK_SCHEMA.empty_table()
        ds = pads.dataset(folder, format="parquet", schema=TICK_SCHEMA)
        filt = (pads.field("time_ns") >= pa.scalar(start, type=TICK_SCHEMA.field("time_ns").type)) & (
            pads.field("time_ns") < pa.scalar(end, type=TICK_SCHEMA.field("time_ns").type)
        )
        return ds.to_table(filter=filt).sort_by("time_ns")


def _sort_and_dedupe(
    table: pa.Table,
    schema: pa.Schema,
    sort_by: str,
    dedupe_keys: tuple[str, ...],
) -> pa.Table:
    """Sort a partition and drop duplicate keys, preserving the last row."""
    if table.num_rows <= 1:
        return table.cast(schema)
    df = table.to_pandas()
    df = df.drop_duplicates(subset=list(dedupe_keys), keep="last")
    df = df.sort_values(sort_by, kind="mergesort")
    return pa.Table.from_pandas(df, schema=schema, preserve_index=False)


def _duplicate_count(table: pa.Table, keys: tuple[str, ...]) -> int:
    if table.num_rows <= 1:
        return 0
    df = table.to_pandas()
    return int(df.duplicated(subset=list(keys)).sum())


def _write_table_atomic(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        pq.write_table(table, tmp, compression="zstd")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
