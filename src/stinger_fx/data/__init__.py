"""Data layer — SQLite + Parquet stores."""

from stinger_fx.data.historical import iter_bars
from stinger_fx.data.parquet_store import ParquetStore
from stinger_fx.data.repositories import (
    BacktestRepo,
    ConfigAuditRepo,
    DecisionRepo,
    OrderRepo,
    SignalRepo,
    TradeRepo,
)
from stinger_fx.data.sqlite_store import SqliteStore, in_memory_store

__all__ = [
    "BacktestRepo",
    "ConfigAuditRepo",
    "DecisionRepo",
    "OrderRepo",
    "ParquetStore",
    "SignalRepo",
    "SqliteStore",
    "TradeRepo",
    "in_memory_store",
    "iter_bars",
]
