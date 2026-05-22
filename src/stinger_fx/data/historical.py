"""Read historical bars from Parquet → Bar domain objects for backtest replay."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from stinger_fx.data.parquet_store import ParquetStore
from stinger_fx.domain import Bar, Timeframe


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
