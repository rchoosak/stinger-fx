"""Typed repositories — thin SQLModel session wrappers.

Engine, broker, and backtest code use these instead of touching `Session`
directly so all DB access stays auditable in one place.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlmodel import select

from stinger_fx.data.schemas import (
    BacktestRunRow,
    ConfigAuditRow,
    DecisionRow,
    OrderModificationRow,
    OrderRow,
    ReconciliationRow,
    SignalRow,
    SweepResultRow,
    SweepRow,
    TradeRow,
)
from stinger_fx.data.sqlite_store import SqliteStore
from stinger_fx.domain import Decision, Order, Signal


class SignalRepo:
    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def add(self, signal: Signal) -> int:
        with self._store.session() as s:
            row = SignalRow(
                strategy_id=signal.strategy_id,
                ts=signal.time,
                symbol=signal.symbol,
                side=signal.side.value,
                strength=signal.strength.value,
                params_json=json.dumps(signal.extra),
            )
            s.add(row)
            s.commit()
            s.refresh(row)
            assert row.id is not None
            return row.id


class DecisionRepo:
    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def add(self, decision: Decision, signal_id: int | None) -> int:
        with self._store.session() as s:
            row = DecisionRow(
                signal_id=signal_id,
                ts=decision.time,
                action=decision.action,
                reason=decision.reason,
                risk_check_passed=decision.risk_check_passed,
            )
            s.add(row)
            s.commit()
            s.refresh(row)
            assert row.id is not None
            return row.id


class OrderRepo:
    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def upsert(self, order: Order) -> int:
        with self._store.session() as s:
            existing = s.exec(select(OrderRow).where(OrderRow.ticket == order.ticket)).first()
            if existing is None:
                row = OrderRow(
                    ticket=order.ticket,
                    strategy_id=order.strategy_id,
                    symbol=order.symbol,
                    side=order.side.value,
                    type=order.type.value,
                    volume=order.volume,
                    price=order.price,
                    fill_price=order.fill_price,
                    sl=order.sl,
                    tp=order.tp,
                    status=order.status.value,
                    requested_at=order.requested_at,
                    filled_at=order.filled_at,
                    comment=order.comment,
                    magic=order.magic,
                    client_order_id=order.client_order_id,
                )
                s.add(row)
                s.commit()
                s.refresh(row)
                assert row.id is not None
                return row.id
            existing.status = order.status.value
            existing.fill_price = order.fill_price
            existing.filled_at = order.filled_at
            existing.sl = order.sl
            existing.tp = order.tp
            s.add(existing)
            s.commit()
            assert existing.id is not None
            return existing.id


class TradeRepo:
    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def add(
        self,
        *,
        position_id: int,
        strategy_id: str,
        symbol: str,
        side: str,
        open_ts: datetime,
        close_ts: datetime,
        open_price: float,
        close_price: float,
        volume: float,
        pnl: float,
        fees: float = 0.0,
        swap: float = 0.0,
    ) -> int:
        with self._store.session() as s:
            row = TradeRow(
                position_id=position_id,
                strategy_id=strategy_id,
                symbol=symbol,
                side=side,
                open_ts=open_ts,
                close_ts=close_ts,
                open_price=open_price,
                close_price=close_price,
                volume=volume,
                pnl=pnl,
                fees=fees,
                swap=swap,
            )
            s.add(row)
            s.commit()
            s.refresh(row)
            assert row.id is not None
            return row.id


class BacktestRepo:
    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def start_run(self, run_id: str, strategy_id: str, params: dict) -> int:
        with self._store.session() as s:
            row = BacktestRunRow(
                run_id=run_id,
                strategy_id=strategy_id,
                params_json=json.dumps(params),
                started_at=datetime.now(UTC),
            )
            s.add(row)
            s.commit()
            s.refresh(row)
            assert row.id is not None
            return row.id

    def finish_run(self, row_id: int, metrics: dict, report_path: str) -> None:
        with self._store.session() as s:
            row = s.get(BacktestRunRow, row_id)
            if row is None:
                return
            row.finished_at = datetime.now(UTC)
            row.metrics_json = json.dumps(metrics)
            row.report_path = report_path
            s.add(row)
            s.commit()

    def list_runs(self, limit: int = 50) -> list[BacktestRunRow]:
        with self._store.session() as s:
            stmt = select(BacktestRunRow).order_by(BacktestRunRow.started_at.desc()).limit(limit)  # type: ignore[union-attr]
            return list(s.exec(stmt))


class ConfigAuditRepo:
    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def log(
        self,
        *,
        file: str,
        before_hash: str,
        after_hash: str,
        applied: bool,
        error: str = "",
    ) -> None:
        with self._store.session() as s:
            row = ConfigAuditRow(
                ts=datetime.now(UTC),
                file=file,
                before_hash=before_hash,
                after_hash=after_hash,
                applied=applied,
                error=error,
            )
            s.add(row)
            s.commit()


class SweepRepo:
    """Persist parameter-sweep summary + per-cell ranked results."""

    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def record_sweep(self, sweep_id: str, strategy_id: str, report) -> int:
        """Write one SweepRow + len(results) SweepResultRow rows."""
        best = report.best()
        with self._store.session() as s:
            head = SweepRow(
                sweep_id=sweep_id,
                strategy_id=strategy_id,
                rank_by=report.rank_by,
                started_at=report.started_at,
                finished_at=report.finished_at,
                total_combos=report.total_combos,
                best_params_json=json.dumps(best.params) if best else "",
                best_metric_value=(best.metrics.get(report.rank_by) if best else None),
            )
            s.add(head)
            for rank, cell in enumerate(report.ranked, start=1):
                s.add(
                    SweepResultRow(
                        sweep_id=sweep_id,
                        rank=rank,
                        params_json=json.dumps(cell.params),
                        metrics_json=json.dumps(cell.metrics),
                    )
                )
            s.commit()
            s.refresh(head)
            assert head.id is not None
            return head.id

    def list_sweeps(self, limit: int = 50) -> list[SweepRow]:
        from sqlmodel import desc

        with self._store.session() as s:
            return list(
                s.exec(select(SweepRow).order_by(desc(SweepRow.started_at)).limit(limit))
            )

    def top_cells(self, sweep_id: str, n: int = 10) -> list[SweepResultRow]:
        with self._store.session() as s:
            return list(
                s.exec(
                    select(SweepResultRow)
                    .where(SweepResultRow.sweep_id == sweep_id)
                    .order_by(SweepResultRow.rank)
                    .limit(n)
                )
            )


class OrderModificationRepo:
    """Persist SL/TP modifications and partial-close events to SQLite.

    Call ``record_modify`` when the bus emits an ``OrderModifiedEvent`` and
    ``record_partial_close`` when it emits a ``PartialClosedEvent``.
    """

    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def record_modify(
        self,
        ts: datetime,
        ticket: int,
        strategy_id: str,
        *,
        old_sl: float | None,
        new_sl: float | None,
        old_tp: float | None,
        new_tp: float | None,
        reason: str = "",
    ) -> int:
        with self._store.session() as s:
            row = OrderModificationRow(
                ts=ts,
                ticket=ticket,
                strategy_id=strategy_id,
                modification_type="modify_sl_tp",
                old_sl=old_sl,
                new_sl=new_sl,
                old_tp=old_tp,
                new_tp=new_tp,
                reason=reason,
            )
            s.add(row)
            s.commit()
            s.refresh(row)
            assert row.id is not None
            return row.id

    def record_partial_close(
        self,
        ts: datetime,
        ticket: int,
        strategy_id: str,
        *,
        closed_volume: float,
        realized_pnl: float,
        reason: str = "",
    ) -> int:
        with self._store.session() as s:
            row = OrderModificationRow(
                ts=ts,
                ticket=ticket,
                strategy_id=strategy_id,
                modification_type="partial_close",
                closed_volume=closed_volume,
                realized_pnl=realized_pnl,
                reason=reason,
            )
            s.add(row)
            s.commit()
            s.refresh(row)
            assert row.id is not None
            return row.id

    def recent(self, limit: int = 100) -> list[OrderModificationRow]:
        from sqlmodel import desc

        with self._store.session() as s:
            return list(
                s.exec(
                    select(OrderModificationRow)
                    .order_by(desc(OrderModificationRow.ts))
                    .limit(limit)
                )
            )


class ReconciliationRepo:
    """Read access for the reconciliation log (writes go through Reconciler)."""

    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def recent(self, limit: int = 100) -> list[ReconciliationRow]:
        from sqlmodel import desc

        with self._store.session() as s:
            return list(
                s.exec(
                    select(ReconciliationRow)
                    .order_by(desc(ReconciliationRow.ts))
                    .limit(limit)
                )
            )

    def by_ticket(self, ticket: int) -> list[ReconciliationRow]:
        with self._store.session() as s:
            return list(
                s.exec(
                    select(ReconciliationRow).where(ReconciliationRow.ticket == ticket)
                )
            )
