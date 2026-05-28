"""OrderQueue — transactional outbox for broker order requests.

Every order goes through three states, persisted to SQLite at each
transition so the engine can survive a crash without losing requests:

  pending ──────► sent       (broker accepted, ticket recorded)
       │
       └──────► failed       (broker rejected or call raised)

The queue offers idempotency at the engine layer: a duplicate
`client_order_id` is recognised and refused. Broker-level idempotency
(MT5 doesn't natively support it) is out of scope — Phase 6.1.A's
retry loop plus this queue's persistence is the pragmatic best we can
do without broker-side cooperation.

Why no background worker? Because in this design `submit()` writes
the row + calls the broker + updates the row inline. The "queue"
metaphor is really a transactional outbox; we get crash recovery
through `replay_pending()` at engine startup.

Sim/file backtests don't use this queue (no value in SQLite latency on
the hot path). The OrderRouter accepts a queue as an optional dep.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from sqlmodel import select

from stinger_fx.brokers.base import BaseBroker
from stinger_fx.data.schemas import PendingOrderRequestRow
from stinger_fx.data.sqlite_store import SqliteStore
from stinger_fx.domain import OrderRequest, OrderResult, OrderStatus

logger = logging.getLogger("stinger.brokers.order_queue")


# `broker_lookup(strategy_id) -> BaseBroker` lets the queue route an order
# request to the right broker on replay (when the strategy_id is the only
# routing information available from the persisted row).
BrokerLookup = Callable[[str], BaseBroker]


class OrderQueue:
    """Persisted, idempotent submission layer between router and broker."""

    def __init__(
        self,
        store: SqliteStore,
        broker_lookup: BrokerLookup,
    ) -> None:
        self._store = store
        self._broker_lookup = broker_lookup

    # --- Submission ---------------------------------------------------------

    async def submit(self, req: OrderRequest, broker: BaseBroker) -> OrderResult:
        """Persist `req` and forward to `broker`.

        Idempotent on `req.client_order_id`. A duplicate id is rejected with
        a synthetic ``OrderResult(ok=False, status=REJECTED)`` so callers
        can distinguish "already submitted" from "broker said no".
        """
        # 1) Idempotency check + persist (pending)
        with self._store.session() as s:
            existing = s.exec(
                select(PendingOrderRequestRow).where(
                    PendingOrderRequestRow.client_order_id == req.client_order_id
                )
            ).first()
            if existing is not None:
                logger.info(
                    "order_queue_duplicate client_order_id=%s prev_status=%s",
                    req.client_order_id, existing.status,
                )
                return OrderResult(
                    ok=False,
                    status=OrderStatus.REJECTED,
                    message=f"duplicate client_order_id (prev_status={existing.status})",
                )
            row = PendingOrderRequestRow(
                client_order_id=req.client_order_id,
                strategy_id=req.strategy_id,
                request_json=req.model_dump_json(),
                enqueued_at=datetime.now(UTC),
                attempts=0,
                status="pending",
            )
            s.add(row)
            s.commit()
            s.refresh(row)
            row_id = row.id

        # 2) Forward to broker — outside the SQLite session so we don't hold
        # the connection open during a potentially long network call.
        try:
            result = await broker.place_order(req)
        except Exception as e:
            self._mark_failed(row_id, error=f"{type(e).__name__}: {e}")
            raise

        # 3) Update row with outcome
        self._update_status(row_id, result)
        return result

    # --- Replay -------------------------------------------------------------

    async def replay_pending(self) -> int:
        """Re-submit any rows still marked ``pending``.  Used at engine
        startup to recover requests written before a previous crash.

        Returns the number of rows replayed.

        Note: this may produce duplicate broker-side orders if the broker
        actually executed the request before the crash. The trade-off is
        acceptable because the alternative — silently dropping persisted
        requests — is worse. Brokers with native idempotency (or our
        reconciler in 6.1.D) catch the duplicate downstream.
        """
        with self._store.session() as s:
            rows = list(
                s.exec(
                    select(PendingOrderRequestRow)
                    .where(PendingOrderRequestRow.status == "pending")
                    .order_by(PendingOrderRequestRow.enqueued_at)  # type: ignore[arg-type]
                )
            )
        if not rows:
            return 0

        logger.warning("order_queue_replay rows=%s", len(rows))
        replayed = 0
        for row in rows:
            req = OrderRequest.model_validate_json(row.request_json)
            broker = self._broker_lookup(req.strategy_id)
            try:
                result = await broker.place_order(req)
            except Exception as e:
                self._mark_failed(row.id, error=f"replay {type(e).__name__}: {e}")
                continue
            self._update_status(row.id, result)
            replayed += 1
        return replayed

    # --- Introspection ------------------------------------------------------

    def pending_count(self) -> int:
        with self._store.session() as s:
            rows = list(
                s.exec(
                    select(PendingOrderRequestRow).where(
                        PendingOrderRequestRow.status == "pending"
                    )
                )
            )
        return len(rows)

    def row_for(self, client_order_id: str) -> PendingOrderRequestRow | None:
        with self._store.session() as s:
            return s.exec(
                select(PendingOrderRequestRow).where(
                    PendingOrderRequestRow.client_order_id == client_order_id
                )
            ).first()

    # --- Helpers ------------------------------------------------------------

    def _update_status(self, row_id: int | None, result: OrderResult) -> None:
        if row_id is None:
            return
        with self._store.session() as s:
            row = s.get(PendingOrderRequestRow, row_id)
            if row is None:
                return
            row.attempts += 1
            if result.ok:
                row.status = "sent"
                if result.ticket is not None:
                    row.broker_ticket = result.ticket
            else:
                row.status = "failed"
                row.last_error = (result.message or "")[:500]
            row.completed_at = datetime.now(UTC)
            s.add(row)
            s.commit()

    def _mark_failed(self, row_id: int | None, *, error: str) -> None:
        if row_id is None:
            return
        with self._store.session() as s:
            row = s.get(PendingOrderRequestRow, row_id)
            if row is None:
                return
            row.attempts += 1
            row.status = "failed"
            row.last_error = error[:500]
            row.completed_at = datetime.now(UTC)
            s.add(row)
            s.commit()
