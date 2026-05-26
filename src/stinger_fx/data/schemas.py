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


class SweepRow(SQLModel, table=True):
    """One row per `stinger-fx backtest sweep`."""

    __tablename__ = "sweep_runs"

    id: int | None = Field(default=None, primary_key=True)
    sweep_id: str = Field(index=True)
    strategy_id: str
    rank_by: str
    started_at: datetime
    finished_at: datetime | None = None
    total_combos: int = 0
    best_params_json: str = ""
    best_metric_value: float | None = None


class SweepResultRow(SQLModel, table=True):
    """One row per cell of the sweep's cartesian product."""

    __tablename__ = "sweep_results"

    id: int | None = Field(default=None, primary_key=True)
    sweep_id: str = Field(index=True)
    rank: int = Field(index=True)               # 1 = best, N = worst by rank_by
    params_json: str
    metrics_json: str


class PendingOrderRequestRow(SQLModel, table=True):
    """A persisted OrderRequest awaiting (or already past) broker submission.

    Lifecycle of `status`:
      * ``pending``  — row written, broker call not yet completed
      * ``sent``     — broker accepted, ticket recorded
      * ``failed``   — broker rejected or call raised

    `client_order_id` is UNIQUE so the OrderQueue can detect duplicate
    enqueue requests (e.g. a retry after a partial failure). The full
    request payload is preserved in `request_json` for crash-recovery
    replay across engine restarts.
    """

    __tablename__ = "pending_order_requests"

    id: int | None = Field(default=None, primary_key=True)
    client_order_id: str = Field(unique=True, index=True)
    strategy_id: str = Field(index=True)
    request_json: str                            # OrderRequest.model_dump_json()
    enqueued_at: datetime = Field(index=True)
    attempts: int = 0
    status: str = Field(default="pending", index=True)
    last_error: str = ""
    broker_ticket: int | None = None
    completed_at: datetime | None = None


class ReconciliationRow(SQLModel, table=True):
    """One row per detected mismatch between broker state and internal DB.

    Written by the ``Reconciler`` after every OrderFilledEvent. ``mismatch_type``
    is one of:

      * ``"position_missing"`` — the order filled but broker doesn't show the
        position (broker amnesia, partial-fill rounding, or our magic-number
        mismatch — needs investigation)
      * ``"volume_drift"``     — broker reports a different volume than what
        we filled (partial close we missed, broker rejected fill silently)
      * ``"price_drift"``      — open_price on broker side differs from our
        ``fill_price`` by more than the tolerance (rare; usually a sign of
        timezone or rounding mismatch)
      * ``"position_unexpected"`` — broker has a position we have no record
        of (cross-contamination from another EA on the same account)
    """

    __tablename__ = "reconciliations"

    id: int | None = Field(default=None, primary_key=True)
    ts: datetime = Field(index=True)
    ticket: int = Field(index=True)
    strategy_id: str = Field(index=True)
    mismatch_type: str = Field(index=True)
    expected_value: float | None = None
    actual_value: float | None = None
    details: str = ""


class OrderModificationRow(SQLModel, table=True):
    """One row per SL/TP modification or partial-close event.

    Populated by the engine's ModificationLogger when it receives
    ``OrderModifiedEvent`` or ``PartialClosedEvent`` from the bus.
    """

    __tablename__ = "order_modifications"

    id: int | None = Field(default=None, primary_key=True)
    ts: datetime = Field(index=True)
    ticket: int = Field(index=True)
    strategy_id: str = Field(index=True)
    modification_type: str = Field(index=True)  # "modify_sl_tp" | "partial_close"
    # SL/TP modification fields (NULL for partial_close rows)
    old_sl: float | None = None
    new_sl: float | None = None
    old_tp: float | None = None
    new_tp: float | None = None
    # Partial-close fields (NULL for modify_sl_tp rows)
    closed_volume: float | None = None
    realized_pnl: float | None = None
    reason: str = ""


class ConfigAuditRow(SQLModel, table=True):
    __tablename__ = "config_audit"

    id: int | None = Field(default=None, primary_key=True)
    ts: datetime = Field(index=True)
    file: str
    before_hash: str = ""
    after_hash: str = ""
    applied: bool = True
    error: str = ""
