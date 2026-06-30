"""Read historical bars/ticks from Parquet → domain objects for backtest replay."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.compute as pc

from stinger_fx.data.parquet_store import ParquetStore
from stinger_fx.domain import Bar, Tick, Timeframe

_INT64 = pa.int64()
_NS_PER_SECOND = 1_000_000_000


def _timestamp_values(batch: pa.RecordBatch, name: str) -> list[int]:
    """Return Arrow timestamp values as raw epoch integers.

    Arrow's timezone-aware ``to_pylist()`` path is very expensive in tick
    backtests because it imports timezone machinery for each row. Casting to
    int64 keeps the hot loop simple, then we build Python datetimes ourselves.
    """
    return cast(list[int], pc.cast(batch.column(name), _INT64).to_pylist())


def iter_bars_from_table(
    symbol: str,
    tf: Timeframe,
    table: pa.Table,
) -> Iterator[Bar]:
    """Yield bars from an Arrow table without per-row dict allocation."""
    for batch in table.to_batches():
        times = _timestamp_values(batch, "time")
        opens = batch.column("open").to_pylist()
        highs = batch.column("high").to_pylist()
        lows = batch.column("low").to_pylist()
        closes = batch.column("close").to_pylist()
        tick_volumes = batch.column("tick_volume").to_pylist()
        real_volumes = batch.column("real_volume").to_pylist()
        spreads = batch.column("spread").to_pylist()
        for time_s, open_, high, low, close, tick_volume, real_volume, spread in zip(
            times,
            opens,
            highs,
            lows,
            closes,
            tick_volumes,
            real_volumes,
            spreads,
            strict=True,
        ):
            yield Bar(
                symbol=symbol,
                timeframe=tf,
                time=datetime.fromtimestamp(time_s, UTC),
                open=open_,
                high=high,
                low=low,
                close=close,
                tick_volume=tick_volume,
                real_volume=real_volume,
                spread=spread,
                is_closed=True,
            )


def iter_ticks_from_table(symbol: str, table: pa.Table) -> Iterator[Tick]:
    """Yield ticks from an Arrow table without timezone-heavy ``to_pylist``."""
    for batch in table.to_batches():
        time_ns_values = _timestamp_values(batch, "time_ns")
        bids = batch.column("bid").to_pylist()
        asks = batch.column("ask").to_pylist()
        lasts = batch.column("last").to_pylist()
        volumes = batch.column("volume").to_pylist()
        flags = batch.column("flags").to_pylist()
        for time_ns, bid, ask, last, volume, flags_ in zip(
            time_ns_values,
            bids,
            asks,
            lasts,
            volumes,
            flags,
            strict=True,
        ):
            yield Tick(
                symbol=symbol,
                time=datetime.fromtimestamp(time_ns / _NS_PER_SECOND, UTC),
                bid=bid,
                ask=ask,
                last=last if last is not None else 0.0,
                volume=volume if volume is not None else 0,
                flags=flags_ if flags_ is not None else 0,
            )


def iter_bars(
    parquet_root: Path,
    symbol: str,
    tf: Timeframe,
    start: datetime,
    end: datetime,
) -> Iterator[Bar]:
    """Yield bars in chronological order."""
    store = ParquetStore(parquet_root)
    table = store.read_bars(symbol, tf, start, end)
    yield from iter_bars_from_table(symbol, tf, table)


def iter_ticks(
    parquet_root: Path,
    symbol: str,
    start: datetime,
    end: datetime,
) -> Iterator[Tick]:
    """Yield ticks in chronological order from `data/parquet/<symbol>/TICK/`.

    Pages through arrow `to_batches()` instead of materialising the whole
    range — one day of EUR/USD ticks can be 100k+ rows.
    """
    store = ParquetStore(parquet_root)
    table = store.read_ticks(symbol, start, end)
    yield from iter_ticks_from_table(symbol, table)
