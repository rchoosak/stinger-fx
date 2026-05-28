"""Reconciler — detects broker / DB state mismatches after every fill.

When the broker confirms a fill we have *our* version of the truth
(``Order.fill_price``, ``Order.volume``, ``ticket``). The Reconciler waits
a short grace period (default 5s — enough for broker books to settle),
then queries ``broker.get_positions()`` and compares what we expected with
what the broker actually shows.

Each discrepancy is persisted to ``ReconciliationRow`` and published as
``ReconciliationMismatchEvent`` so a notification sink can page the
operator. The component is a lifecycle dependency — engine wires it on
startup; ``stop()`` cancels any pending verification tasks.

Limitations / known noise sources we deliberately do NOT alarm on:
  * Partial fills that close instantly (the position would already be gone)
  * Trades closed by SL/TP between fill and verification (also fine)

To distinguish "closed too fast" from "broker lost it", a partial-close /
position-closed event arriving for the same ticket cancels the pending
verification task. That's left as a follow-up — the current implementation
errs on the side of reporting; operators can dismiss false positives in
the audit UI.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from stinger_fx.brokers.base import BaseBroker
from stinger_fx.core.event_bus import AsyncEventBus, Subscription
from stinger_fx.core.events import (
    OrderFilledEvent,
    PositionClosedEvent,
    ReconciliationMismatchEvent,
)
from stinger_fx.data.schemas import ReconciliationRow
from stinger_fx.data.sqlite_store import SqliteStore
from stinger_fx.domain import Order

logger = logging.getLogger("stinger.data.reconciliation")


class Reconciler:
    """Compare broker state to internal DB after every fill."""

    def __init__(
        self,
        bus: AsyncEventBus,
        broker: BaseBroker,
        store: SqliteStore,
        *,
        verify_delay_seconds: float = 5.0,
        price_tolerance_pips: float = 2.0,
        point: float = 0.0001,
    ) -> None:
        self._bus = bus
        self._broker = broker
        self._store = store
        self._verify_delay = verify_delay_seconds
        self._price_tolerance = price_tolerance_pips * point
        self._sub: Subscription | None = None
        self._close_sub: Subscription | None = None
        # Track per-ticket verification tasks so we can cancel when a
        # PositionClosedEvent arrives before the delay elapses.
        self._tasks: dict[int, asyncio.Task[None]] = {}

    async def start(self) -> None:
        self._sub = self._bus.subscribe(
            OrderFilledEvent, self._on_filled, name="reconciler.fill"
        )
        self._close_sub = self._bus.subscribe(
            PositionClosedEvent, self._on_closed, name="reconciler.close"
        )
        logger.info("reconciler_started verify_delay_s=%.1f", self._verify_delay)

    async def stop(self) -> None:
        if self._sub:
            await self._sub.unsubscribe()
            self._sub = None
        if self._close_sub:
            await self._close_sub.unsubscribe()
            self._close_sub = None
        for task in list(self._tasks.values()):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()

    # --- Event handlers -----------------------------------------------------

    async def _on_filled(self, evt: OrderFilledEvent) -> None:
        order = evt.order
        if order.ticket <= 0:
            return
        # If a verification is already in flight for this ticket (e.g.
        # because of a duplicate fill event), cancel it before scheduling
        # the new one — the latest data is most relevant.
        existing = self._tasks.get(order.ticket)
        if existing is not None and not existing.done():
            existing.cancel()
        task = asyncio.create_task(self._verify(order))
        self._tasks[order.ticket] = task
        task.add_done_callback(lambda _t: self._tasks.pop(order.ticket, None))

    async def _on_closed(self, evt: PositionClosedEvent) -> None:
        """Cancel a pending verify when the position closes before the
        delay elapses — closing too fast is a perfectly legal trajectory,
        not a reconciliation failure."""
        task = self._tasks.pop(evt.position.ticket, None)
        if task is not None and not task.done():
            task.cancel()

    # --- Verification logic -------------------------------------------------

    async def _verify(self, order: Order) -> None:
        try:
            await asyncio.sleep(self._verify_delay)
        except asyncio.CancelledError:
            return
        try:
            positions = await self._broker.get_positions()
        except Exception as e:
            logger.warning("reconciler_broker_unreachable err=%s", e)
            return
        match = next((p for p in positions if p.ticket == order.ticket), None)
        if match is None:
            await self._record(
                order,
                "position_missing",
                expected=order.volume,
                actual=0.0,
                details=f"order filled at {order.fill_price}, but broker shows no open position",
            )
            return
        # Volume comparison — broker rounds to lot step so 0.099999 vs 0.1
        # is fine; we use a 1e-6 tolerance (way smaller than any lot step).
        if abs(match.volume - order.volume) > 1e-6:
            await self._record(
                order,
                "volume_drift",
                expected=order.volume,
                actual=match.volume,
                details=f"expected {order.volume} lots, broker has {match.volume}",
            )
        if order.fill_price is not None:
            drift = abs(match.open_price - order.fill_price)
            if drift > self._price_tolerance:
                await self._record(
                    order,
                    "price_drift",
                    expected=order.fill_price,
                    actual=match.open_price,
                    details=(
                        f"open_price drift={drift:.5f} > tolerance={self._price_tolerance:.5f}"
                    ),
                )

    async def _record(
        self,
        order: Order,
        mismatch_type: str,
        *,
        expected: float | None,
        actual: float | None,
        details: str,
    ) -> None:
        try:
            with self._store.session() as s:
                row = ReconciliationRow(
                    ts=datetime.now(UTC),
                    ticket=order.ticket,
                    strategy_id=order.strategy_id,
                    mismatch_type=mismatch_type,
                    expected_value=expected,
                    actual_value=actual,
                    details=details,
                )
                s.add(row)
                s.commit()
        except Exception:
            logger.exception(
                "reconciler_db_write_failed ticket=%s type=%s", order.ticket, mismatch_type
            )
        await self._bus.publish(
            ReconciliationMismatchEvent(
                ticket=order.ticket,
                strategy_id=order.strategy_id,
                mismatch_type=mismatch_type,
                expected_value=expected,
                actual_value=actual,
                details=details,
            )
        )
        logger.warning(
            "reconciliation_mismatch type=%s ticket=%s expected=%s actual=%s details=%s",
            mismatch_type, order.ticket, expected, actual, details,
        )
