"""Pytest fixtures shared across the suite."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stinger_fx.data.parquet_store import ParquetStore
from stinger_fx.domain import Bar, Timeframe


@pytest.fixture
def utc_now() -> datetime:
    return datetime.now(UTC)


@pytest.fixture
def seeded_parquet(tmp_path: Path) -> Path:
    """Return a Parquet root with 100 deterministic EURUSD M15 bars."""
    root = tmp_path / "parquet"
    store = ParquetStore(root)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    # 100 bars in a clear up-then-down pattern so MA crossover will fire.
    prices = (
        [1.10 + 0.001 * i for i in range(50)]
        + [1.15 - 0.001 * i for i in range(50)]
    )
    bars = [
        Bar(
            symbol="EURUSD",
            timeframe=Timeframe.M15,
            time=base + timedelta(minutes=15 * i),
            open=p,
            high=p + 0.0002,
            low=p - 0.0002,
            close=p,
            tick_volume=100,
            is_closed=True,
        )
        for i, p in enumerate(prices)
    ]
    store.append_bars("EURUSD", Timeframe.M15, bars)
    return root
