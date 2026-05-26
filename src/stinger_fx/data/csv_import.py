"""CSV → Parquet tick importer.

`import_ticks_csv(...)` reads a tick CSV with `pyarrow.csv.read_csv` (fast,
streaming-friendly), normalises the columns into `TICK_SCHEMA`, and routes
through `ParquetStore.append_ticks` so the resulting daily partitions match
the layout produced by the live broker pump.

Designed for two common shapes:

  • Dukascopy-style:  Local time, Ask, Bid, AskVolume, BidVolume
  • MT5 export:       time, bid, ask, last, volume, flags

Mapping is explicit (caller passes `time_col`, `bid_col`, `ask_col`) so we
don't have to guess. Timestamps may be naive — the caller supplies a `tz`
(default UTC).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv

from stinger_fx.data.parquet_store import TICK_SCHEMA, ParquetStore
from stinger_fx.domain import Tick

logger = logging.getLogger("stinger.data.csv_import")


class CsvImportError(ValueError):
    """Raised on a malformed CSV (missing column, unparseable timestamps, …)."""


def _coerce_time_column(arr: pa.Array, tz: str) -> pa.Array:
    """Return a `timestamp[ns, tz=UTC]` array regardless of the input dtype.

    Accepts:
      • Already-typed timestamp arrays (with or without tz)
      • String arrays in any format pyarrow's strptime / cast can parse,
        falling back to pandas for "2024.01.15 00:00:00" style dates.
      • Int arrays interpreted as Unix seconds / milliseconds / nanoseconds
        (auto-detected by magnitude).
    """
    target = pa.timestamp("ns", tz="UTC")

    if pa.types.is_timestamp(arr.type):
        # Already a timestamp — attach tz if naive (interpret values as local
        # time in `tz`), then convert to UTC. `cast` doesn't shift values for
        # naive→aware; `assume_timezone` does the right thing.
        if arr.type.tz is None:
            attached = pc.assume_timezone(arr, tz)
        else:
            attached = arr
        return attached.cast(target)

    if pa.types.is_integer(arr.type):
        # Auto-detect epoch unit: ns >= 1e18, μs >= 1e15, ms >= 1e12, else s.
        sample = arr.to_pylist()[:1]
        if not sample:
            raise CsvImportError("time column is empty")
        s = sample[0]
        if s is None:
            raise CsvImportError("time column has a null value in the first row")
        if s >= 10**18:
            unit = "ns"
        elif s >= 10**15:
            unit = "us"
        elif s >= 10**12:
            unit = "ms"
        else:
            unit = "s"
        # Build a tz-aware UTC timestamp directly from the epoch ints —
        # epoch values have no concept of "naive", so `tz` is ignored here.
        ts = arr.cast(pa.timestamp(unit, tz="UTC"))
        return ts.cast(target)

    if pa.types.is_string(arr.type):
        # Try ISO 8601 first via pyarrow's cast (which understands a wide
        # range), then fall back to pandas / dateutil for dialects like
        # "2024.01.15 00:00:00" that pyarrow rejects.
        try:
            ts = arr.cast(pa.timestamp("ns"))
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            try:
                import pandas as pd
            except ImportError as e:
                raise CsvImportError(
                    "string timestamps require pandas for non-ISO formats"
                ) from e
            series = pd.to_datetime(arr.to_pylist(), utc=False, errors="raise")
            # pandas always returns ns-resolution
            ts = pa.array(series.to_numpy(), type=pa.timestamp("ns"))

        # ts is now naive ns — interpret values as local time in `tz`, then
        # convert to UTC. Use `assume_timezone` (which performs the shift)
        # rather than `cast` (which only attaches a label).
        try:
            ZoneInfo(tz)
        except Exception as e:
            raise CsvImportError(f"unknown timezone {tz!r}") from e
        attached = pc.assume_timezone(ts, tz)
        return attached.cast(target)

    raise CsvImportError(
        f"time column has unsupported dtype {arr.type}; expected timestamp, "
        f"integer epoch, or parseable string"
    )


def _coerce_float_column(arr: pa.Array, name: str) -> pa.Array:
    if pa.types.is_floating(arr.type):
        return arr.cast(pa.float64())
    if pa.types.is_integer(arr.type):
        return arr.cast(pa.float64())
    if pa.types.is_string(arr.type):
        try:
            return pc.cast(arr, pa.float64())
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
            raise CsvImportError(f"column {name!r} has non-numeric values") from e
    raise CsvImportError(
        f"column {name!r} has unsupported dtype {arr.type}; expected numeric"
    )


def import_ticks_csv(
    csv_path: Path,
    *,
    symbol: str,
    parquet_root: Path,
    time_col: str = "time",
    bid_col: str = "bid",
    ask_col: str = "ask",
    last_col: str | None = None,
    volume_col: str | None = None,
    flags_col: str | None = None,
    tz: str = "UTC",
    batch_size: int = 100_000,
) -> int:
    """Import ticks from a CSV file into the project's Parquet store.

    Returns the number of ticks written.

    The CSV is read in blocks via `pyarrow.csv.read_csv` (zero-copy where
    possible). Each block is normalised into `TICK_SCHEMA`, materialised as
    `Tick` domain objects, and appended through `ParquetStore.append_ticks`
    which handles daily partitioning.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise CsvImportError(f"CSV not found: {csv_path}")

    read_options = pa_csv.ReadOptions(block_size=batch_size * 64)
    parse_options = pa_csv.ParseOptions()
    convert_options = pa_csv.ConvertOptions(strings_can_be_null=True)

    try:
        table = pa_csv.read_csv(
            str(csv_path),
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        )
    except pa.ArrowInvalid as e:
        raise CsvImportError(f"failed to parse CSV: {e}") from e

    # Validate required columns first — fail loudly with the column we expected
    missing = [c for c in (time_col, bid_col, ask_col) if c not in table.column_names]
    if missing:
        raise CsvImportError(
            f"CSV missing required column(s) {missing!r}; "
            f"available columns: {table.column_names!r}"
        )

    # Normalise columns into TICK_SCHEMA order
    time_arr = _coerce_time_column(table.column(time_col), tz=tz)
    bid_arr = _coerce_float_column(table.column(bid_col), bid_col)
    ask_arr = _coerce_float_column(table.column(ask_col), ask_col)

    n = table.num_rows
    if last_col and last_col in table.column_names:
        last_arr = _coerce_float_column(table.column(last_col), last_col)
    else:
        last_arr = pa.array([0.0] * n, type=pa.float64())

    if volume_col and volume_col in table.column_names:
        v = table.column(volume_col)
        if pa.types.is_integer(v.type):
            volume_arr = v.cast(pa.int64())
        elif pa.types.is_floating(v.type):
            volume_arr = pc.cast(v, pa.int64())
        else:
            volume_arr = pc.cast(v, pa.int64())
    else:
        volume_arr = pa.array([0] * n, type=pa.int64())

    if flags_col and flags_col in table.column_names:
        flags_arr = table.column(flags_col).cast(pa.int64())
    else:
        flags_arr = pa.array([0] * n, type=pa.int64())

    normalised = pa.Table.from_arrays(
        [time_arr, bid_arr, ask_arr, last_arr, volume_arr, flags_arr],
        schema=TICK_SCHEMA,
    )

    # Iterate to_pylist in chunks so we don't materialise N×6 Python floats
    # all at once on a giant CSV.
    store = ParquetStore(parquet_root)
    written = 0
    for batch in normalised.to_batches(max_chunksize=batch_size):
        ticks = _batch_to_ticks(symbol, batch)
        written += store.append_ticks(symbol, ticks)

    logger.info(
        "csv_import.done file=%s symbol=%s rows=%s", csv_path, symbol, written
    )
    return written


def _batch_to_ticks(symbol: str, batch: pa.RecordBatch) -> Iterable[Tick]:
    rows = batch.to_pylist()
    for r in rows:
        yield Tick(
            symbol=symbol,
            time=r["time_ns"],
            bid=r["bid"],
            ask=r["ask"],
            last=r.get("last") or 0.0,
            volume=r.get("volume") or 0,
            flags=r.get("flags") or 0,
        )
