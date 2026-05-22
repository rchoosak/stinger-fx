"""SQLModel tables — relational state for Stinger-Fx.

Time-series data (ticks, bars) lives in Parquet; SQLite holds everything that
benefits from indexed lookup, joins, or aggregation.
"""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel


class Account(SQLModel, table=True):
    __tablename__ = "accounts"

    account_id: str = Field(primary_key=True)
    broker: str
    login: int
    currency: str
    leverage: int
    server: str
    created_at: datetime


class AccountSnapshotRow(SQLModel, table=True):
    __tablename__ = "account_snapshots"

    id: int | None = Field(default=None, primary_key=True)
    account_id: str = Field(index=True)
    ts: datetime = Field(index=True)
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float = 0.0
    profit: float = 0.0


class StrategyRow(SQLModel, table=True):
    __tablename__ = "strategies"

    id: str = Field(primary_key=True)
    name: str
    version: str
    params_json: str
    enabled: bool = True
    created_at: datetime
    updated_at: datetime


class SignalRow(SQLModel, table=True):
    __tablename__ = "signals"

    id: int | None = Field(default=None, primary_key=True)
    strategy_id: str = Field(index=True)
    ts: datetime = Field(index=True)
    symbol: str
    side: str
    strength: str
    params_json: str = ""
    decision_id: int | None = None


class DecisionRow(SQLModel, table=True):
    __tablename__ = "decisions"

    id: int | None = Field(default=None, primary_key=True)
    signal_id: int | None = Field(default=None, index=True)
    ts: datetime = Field(index=True)
    action: str
    reason: str = ""
    risk_check_passed: bool = True


class OrderRow(SQLModel, table=True):
    __tablename__ = "orders"

    id: int | None = Field(default=None, primary_key=True)
    ticket: int = Field(index=True)
    strategy_id: str = Field(index=True)
    symbol: str
    side: str
    type: str
    volume: float
    price: float | None = None
    fill_price: float | None = None
    sl: float | None = None
    tp: float | None = None
    status: str
    requested_at: datetime | None = None
    filled_at: datetime | None = None
    comment: str = ""
    magic: int = 0
    client_order_id: str = Field(default="", index=True)


class TradeRow(SQLModel, table=True):
    """One row per closed position (matched open+close)."""

    __tablename__ = "trades"

    id: int | None = Field(default=None, primary_key=True)
    position_id: int = Field(index=True)
    strategy_id: str = Field(index=True)
    symbol: str
    side: str
    open_ts: datetime = Field(index=True)
    close_ts: datetime
    open_price: float
    close_price: float
    volume: float
    pnl: float
    fees: float = 0.0
    swap: float = 0.0


class BacktestRunRow(SQLModel, table=True):
    __tablename__ = "backtest_runs"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    strategy_id: str
    params_json: str
    started_at: datetime
    finished_at: datetime | None = None
    metrics_json: str = ""
    report_path: str = ""


class ConfigAuditRow(SQLModel, table=True):
    __tablename__ = "config_audit"

    id: int | None = Field(default=None, primary_key=True)
    ts: datetime = Field(index=True)
    file: str
    before_hash: str = ""
    after_hash: str = ""
    applied: bool = True
    error: str = ""
