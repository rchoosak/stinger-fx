"""Read historical bars/ticks from Parquet → domain objects for backtest replay."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from stinger_fx.data.parquet_store import ParquetStore
from stinger_fx.domain import Bar, Tick, Timeframe


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
    for batch in table.to_batches():
        rows = batch.to_pylist()
        for row in rows:
            yield Bar(
                symbol=symbol,
                timeframe=tf,
                time=row["time"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                tick_volume=row["tick_volume"],
                real_volume=row["real_volume"],
                spread=row["spread"],
                is_closed=True,
            )


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
    for batch in table.to_batches():
        rows = batch.to_pylist()
        for row in rows:
            yield Tick(
                symbol=symbol,
                time=row["time_ns"],
                bid=row["bid"],
                ask=row["ask"],
                last=row.get("last") or 0.0,
                volume=row.get("volume") or 0,
                flags=row.get("flags") or 0,
            )
