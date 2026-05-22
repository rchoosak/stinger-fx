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
    OrderRow,
    SignalRow,
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
