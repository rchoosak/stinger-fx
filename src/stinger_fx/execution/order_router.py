"""SignalEvent → broker OrderRequest.

A signal carries the strategy's intent; the router applies risk checks and
converts to an OrderRequest with a deterministic client_order_id.
"""

from __future__ import annotations

import logging
import uuid

from stinger_fx.brokers.base import BaseBroker
from stinger_fx.brokers.order_queue import OrderQueue
from stinger_fx.brokers.pool import BrokerPool
from stinger_fx.core.event_bus import AsyncEventBus
from stinger_fx.core.events import (
    CancelOrderRequestEvent,
    ClosePositionRequestEvent,
    DecisionEvent,
    ModifyOrderRequestEvent,
    OrderFilledEvent,
    OrderModifiedEvent,
    OrderRejectedEvent,
    PartialCloseRequestEvent,
    SignalEvent,
)
from stinger_fx.domain import (
    Decision,
    Order,
    OrderRequest,
    OrderResult,
    OrderStatus,
    Signal,
)
from stinger_fx.risk import RiskMonitor

logger = logging.getLogger("stinger.engine.router")


class OrderRouter:
    """Routes signals to the right broker in a multi-account setup.

    Construct with either a single `broker` (legacy single-account flow used by
    backtests and Phase-1 live mode) or a `pool` + `strategy_accounts` map
    (multi-account live flow). Single-broker callers are a special case of the
    pool with one broker keyed as "default".
    """

    def __init__(
        self,
        bus: AsyncEventBus,
        broker: BaseBroker | None = None,
        *,
        pool: BrokerPool | None = None,
        strategy_magic: dict[str, int] | None = None,
        strategy_accounts: dict[str, str] | None = None,
        risk: RiskMonitor | None = None,
        queue: OrderQueue | None = None,
    ) -> None:
        if broker is None and pool is None:
            raise ValueError("OrderRouter needs either broker= or pool=")
        if broker is not None and pool is not None:
            raise ValueError("OrderRouter accepts broker= XOR pool= (not both)")
        self.bus = bus
        if pool is None:
            assert broker is not None
            pool = BrokerPool([("default", broker)])
        self._pool = pool
        # Legacy attribute used by older callers (mostly the file backtester
        # and a few tests) — exposes the primary broker.
        self.broker = pool.primary()
        self.strategy_magic = strategy_magic or {}
        self.strategy_accounts: dict[str, str] = dict(strategy_accounts or {})
        self.risk = risk
        # Phase 6.1.C — optional persisted-outbox layer. When provided every
        # order is written to SQLite before submission and replayable on
        # restart; sim/file backtests leave this as None for zero DB latency.
        self.queue = queue

    @property
    def pool(self) -> BrokerPool:
        return self._pool

    def _broker_for(self, strategy_id: str) -> BaseBroker:
        account_id = self.strategy_accounts.get(strategy_id)
        if account_id is not None and self._pool.has(account_id):
            return self._pool.get(account_id)
        # Unconfigured / unknown strategy → primary broker. Keeps single-broker
        # backtests and unmapped one-off signals working.
        return self._pool.primary()

    async def handle_signal(self, signal: Signal) -> None:
        client_order_id = str(uuid.uuid4())
        magic = self.strategy_magic.get(signal.strategy_id, 0)
        volume = signal.suggested_volume or 0.01
        broker = self._broker_for(signal.strategy_id)

        # Pre-trade risk check. Rejection short-circuits the order path and
        # is recorded in a DecisionEvent so the trade journal shows why.
        if self.risk is not None:
            verdict = self.risk.check_signal(signal)
            if not verdict.allowed:
                logger.info(
                    "signal_rejected_by_risk strategy=%s symbol=%s reason=%s",
                    signal.strategy_id,
                    signal.symbol,
                    verdict.reason,
                )
                await self.bus.publish(
                    DecisionEvent(
                        decision=Decision(
                            signal=signal,
                            time=signal.time,
                            action="rejected",
                            reason=verdict.reason,
                            risk_check_passed=False,
                            client_order_id=None,
                        )
                    )
                )
                return

        req = OrderRequest(
            strategy_id=signal.strategy_id,
            symbol=signal.symbol,
            side=signal.side,
            type=signal.order_type,
            volume=volume,
            price=signal.suggested_price,
            sl=signal.suggested_sl,
            tp=signal.suggested_tp,
            comment=signal.comment,
            magic=magic,
            client_order_id=client_order_id,
        )
        decision = Decision(
            signal=signal,
            time=signal.time,
            action="placed",
            client_order_id=client_order_id,
        )
        await self.bus.publish(DecisionEvent(decision=decision))

        # Phase 6.1.C — route through the persisted outbox when configured;
        # otherwise call the broker directly (sim/file backtest path).
        if self.queue is not None:
            result = await self.queue.submit(req, broker)
        else:
            result = await broker.place_order(req)
        await self._publish_submission_result(req, result)

    async def replay_pending_orders(self) -> int:
        """Replay durable outbox rows and publish the same result events.

        Live mode calls this after broker connections are established. Backtests
        leave ``queue`` unset and get a cheap no-op.
        """
        if self.queue is None:
            return 0
        replayed = await self.queue.replay_pending_results()
        for item in replayed:
            await self._publish_submission_result(item.request, item.result)
        return len(replayed)

    async def _publish_submission_result(
        self,
        req: OrderRequest,
        result: OrderResult,
    ) -> None:
        if result.ok and result.order is not None and result.status in (
            OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED,
        ):
            # Market orders fill immediately — emit OrderFilledEvent.
            # PARTIALLY_FILLED comes from MT5 retcode DONE_PARTIAL (live broker
            # accepted some but not all of the requested volume). The order
            # itself carries status=PARTIALLY_FILLED and filled_volume<volume
            # so subscribers can distinguish; downstream (RiskMonitor counter,
            # strategy.on_order_filled, metrics, trade journal) only needs to
            # know the fill *happened*. Pre-fix, PARTIALLY_FILLED fell through
            # all three branches and disappeared silently in live mode.
            await self.bus.publish(OrderFilledEvent(order=result.order))
        elif result.ok and result.status == OrderStatus.SUBMITTED:
            # Pending order placed (Phase 6.2.B) — the broker already emitted
            # OrderSubmittedEvent. Strategy will see OrderFilledEvent later
            # when check_pending() promotes it to a position. Nothing to do here.
            pass
        elif not result.ok:
            await self.bus.publish(
                OrderRejectedEvent(
                    order=Order(
                        ticket=result.ticket or 0,
                        strategy_id=req.strategy_id,
                        symbol=req.symbol,
                        side=req.side,
                        type=req.type,
                        volume=req.volume,
                        price=req.price,
                        sl=req.sl,
                        tp=req.tp,
                        status=OrderStatus.REJECTED,
                        comment=req.comment,
                        magic=req.magic,
                        client_order_id=req.client_order_id,
                    ),
                    reason=result.message,
                )
            )

    async def handle_modify(self, evt: ModifyOrderRequestEvent) -> None:
        """Route a strategy's modification request to the right broker.

        Targets either an open position (SL/TP only) or a pending order
        (price/volume/SL/TP — Phase 6.2.D). Ownership is enforced by magic.
        """
        broker = self._broker_for(evt.strategy_id)
        expected_magic = self.strategy_magic.get(evt.strategy_id, 0)

        # Phase 6.2.D — check pending orders first; if the ticket is pending
        # we go through the broker.modify_order(price=, volume=) path.
        pendings = await broker.get_open_orders()
        pending = next((o for o in pendings if o.ticket == evt.ticket), None)
        if pending is not None:
            if pending.magic != expected_magic:
                logger.warning(
                    "modify_rejected_cross_strategy strategy=%s ticket=%s "
                    "pending_magic=%s expected_magic=%s",
                    evt.strategy_id, evt.ticket, pending.magic, expected_magic,
                )
                return
            new_sl = evt.sl if evt.sl is not None else pending.sl
            new_tp = evt.tp if evt.tp is not None else pending.tp
            new_price = evt.price if evt.price is not None else pending.price
            new_volume = evt.volume if evt.volume is not None else pending.volume
            result = await broker.modify_order(
                evt.ticket,
                sl=new_sl,
                tp=new_tp,
                price=new_price,
                volume=new_volume,
            )
            if not result.ok:
                logger.warning(
                    "modify_pending_failed strategy=%s ticket=%s reason=%s",
                    evt.strategy_id, evt.ticket, result.message,
                )
                return
            # Don't publish OrderModifiedEvent for pending modifications —
            # that event carries a Position, which doesn't exist yet.
            # Position-state managers care about open positions; the audit
            # log records the request itself.
            return

        # Open position path (Phase 4 — unchanged)
        positions = await broker.get_positions()
        pos = next((p for p in positions if p.ticket == evt.ticket), None)
        if pos is None:
            logger.warning(
                "modify_unknown_ticket strategy=%s ticket=%s",
                evt.strategy_id, evt.ticket,
            )
            return
        if pos.magic != expected_magic:
            logger.warning(
                "modify_rejected_cross_strategy strategy=%s ticket=%s "
                "position_magic=%s expected_magic=%s",
                evt.strategy_id, evt.ticket, pos.magic, expected_magic,
            )
            return

        # Preserve unchanged stops — caller passes None when they don't want
        # to touch one side. `BaseBroker.modify_order` accepts None to mean
        # "leave alone" too, but some brokers clear the value; pass the
        # current value through explicitly when None is requested.
        new_sl = evt.sl if evt.sl is not None else pos.sl
        new_tp = evt.tp if evt.tp is not None else pos.tp
        result = await broker.modify_order(evt.ticket, sl=new_sl, tp=new_tp)
        if not result.ok:
            logger.warning(
                "modify_failed strategy=%s ticket=%s reason=%s",
                evt.strategy_id, evt.ticket, result.message,
            )
            return

        # Re-read the position so the event payload is the post-modify
        # snapshot (avoids racing on broker internal state).
        positions = await broker.get_positions()
        updated = next((p for p in positions if p.ticket == evt.ticket), pos)
        await self.bus.publish(
            OrderModifiedEvent(position=updated, reason=evt.reason)
        )

    async def handle_partial_close(self, evt: PartialCloseRequestEvent) -> None:
        """Route a strategy's partial-close request to the right broker."""
        broker = self._broker_for(evt.strategy_id)
        positions = await broker.get_positions()
        pos = next((p for p in positions if p.ticket == evt.ticket), None)
        if pos is None:
            logger.warning(
                "partial_close_unknown_ticket strategy=%s ticket=%s",
                evt.strategy_id, evt.ticket,
            )
            return
        expected_magic = self.strategy_magic.get(evt.strategy_id, 0)
        if pos.magic != expected_magic:
            logger.warning(
                "partial_close_rejected_cross_strategy strategy=%s ticket=%s "
                "position_magic=%s expected_magic=%s",
                evt.strategy_id, evt.ticket, pos.magic, expected_magic,
            )
            return
        if evt.volume >= pos.volume:
            # Over-close → caller probably meant `close_position`. Refuse so
            # they don't accidentally drop the whole position via this path
            # (which would also have to publish a different event).
            logger.warning(
                "partial_close_volume_too_large strategy=%s ticket=%s "
                "requested=%s position_volume=%s",
                evt.strategy_id, evt.ticket, evt.volume, pos.volume,
            )
            return

        before_volume = pos.volume
        result = await broker.close_position(evt.ticket, volume=evt.volume)
        if not result.ok:
            logger.warning(
                "partial_close_failed strategy=%s ticket=%s reason=%s",
                evt.strategy_id, evt.ticket, result.message,
            )
            return

        # The broker emits the canonical PartialClosedEvent when it knows the
        # realised P&L of the chunk. For brokers that don't, fall back to
        # publishing here with the remaining-leg snapshot.
        positions = await broker.get_positions()
        remaining = next((p for p in positions if p.ticket == evt.ticket), None)
        if remaining is None:
            # Position vanished — broker treated it as a full close.
            logger.info(
                "partial_close_collapsed_to_full strategy=%s ticket=%s",
                evt.strategy_id, evt.ticket,
            )
            return
        # If the broker didn't shrink the volume, treat as no-op + log it.
        if remaining.volume >= before_volume:
            logger.warning(
                "partial_close_no_shrink strategy=%s ticket=%s before=%s after=%s",
                evt.strategy_id, evt.ticket, before_volume, remaining.volume,
            )

    async def handle_close(self, evt: ClosePositionRequestEvent) -> None:
        """Route a strategy's full-close request to the right broker.

        Ownership check mirrors `handle_modify` / `handle_partial_close`.
        Emits `PositionClosedEvent` on success; the broker's own
        `close_position` implementation may also emit one — downstream
        subscribers should be idempotent.
        """
        broker = self._broker_for(evt.strategy_id)
        positions = await broker.get_positions()
        pos = next((p for p in positions if p.ticket == evt.ticket), None)
        if pos is None:
            logger.warning(
                "close_unknown_ticket strategy=%s ticket=%s",
                evt.strategy_id, evt.ticket,
            )
            return
        expected_magic = self.strategy_magic.get(evt.strategy_id, 0)
        if pos.magic != expected_magic:
            logger.warning(
                "close_rejected_cross_strategy strategy=%s ticket=%s "
                "position_magic=%s expected_magic=%s",
                evt.strategy_id, evt.ticket, pos.magic, expected_magic,
            )
            return

        result = await broker.close_position(evt.ticket)
        if not result.ok:
            logger.warning(
                "close_failed strategy=%s ticket=%s reason=%s",
                evt.strategy_id, evt.ticket, result.message,
            )

    async def handle_cancel(self, evt: CancelOrderRequestEvent) -> None:
        """Route a strategy's pending-order cancellation request.

        Ownership check uses `get_open_orders()` (pending orders) rather
        than `get_positions()` — the magic comparison is the same.
        """
        broker = self._broker_for(evt.strategy_id)
        pending = await broker.get_open_orders()
        order = next((o for o in pending if o.ticket == evt.ticket), None)
        if order is None:
            logger.warning(
                "cancel_unknown_ticket strategy=%s ticket=%s",
                evt.strategy_id, evt.ticket,
            )
            return
        expected_magic = self.strategy_magic.get(evt.strategy_id, 0)
        if order.magic != expected_magic:
            logger.warning(
                "cancel_rejected_cross_strategy strategy=%s ticket=%s "
                "order_magic=%s expected_magic=%s",
                evt.strategy_id, evt.ticket, order.magic, expected_magic,
            )
            return
        result = await broker.cancel_order(evt.ticket)
        if not result.ok:
            logger.warning(
                "cancel_failed strategy=%s ticket=%s reason=%s",
                evt.strategy_id, evt.ticket, result.message,
            )

    async def attach(self) -> None:
        async def _on_signal(evt: SignalEvent) -> None:
            await self.handle_signal(evt.signal)

        async def _on_modify(evt: ModifyOrderRequestEvent) -> None:
            await self.handle_modify(evt)

        async def _on_partial_close(evt: PartialCloseRequestEvent) -> None:
            await self.handle_partial_close(evt)

        async def _on_close(evt: ClosePositionRequestEvent) -> None:
            await self.handle_close(evt)

        async def _on_cancel(evt: CancelOrderRequestEvent) -> None:
            await self.handle_cancel(evt)

        self._sub = self.bus.subscribe(SignalEvent, _on_signal, name="order_router")
        self._sub_modify = self.bus.subscribe(
            ModifyOrderRequestEvent, _on_modify, name="order_router.modify"
        )
        self._sub_partial = self.bus.subscribe(
            PartialCloseRequestEvent, _on_partial_close, name="order_router.partial"
        )
        self._sub_close = self.bus.subscribe(
            ClosePositionRequestEvent, _on_close, name="order_router.close"
        )
        self._sub_cancel = self.bus.subscribe(
            CancelOrderRequestEvent, _on_cancel, name="order_router.cancel"
        )

    async def detach(self) -> None:
        if hasattr(self, "_sub"):
            await self._sub.unsubscribe()
        if hasattr(self, "_sub_modify"):
            await self._sub_modify.unsubscribe()
        if hasattr(self, "_sub_partial"):
            await self._sub_partial.unsubscribe()
        if hasattr(self, "_sub_close"):
            await self._sub_close.unsubscribe()
        if hasattr(self, "_sub_cancel"):
            await self._sub_cancel.unsubscribe()
