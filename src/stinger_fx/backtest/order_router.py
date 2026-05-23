"""SignalEvent → broker OrderRequest.

A signal carries the strategy's intent; the router applies risk checks and
converts to an OrderRequest with a deterministic client_order_id.

This file lives under `backtest/` for now because the backtester is the first
caller, but the router itself is broker-agnostic — live mode will share it.
"""

from __future__ import annotations

import logging
import uuid

from stinger_fx.brokers.base import BaseBroker
from stinger_fx.brokers.pool import BrokerPool
from stinger_fx.core.event_bus import AsyncEventBus
from stinger_fx.core.events import (
    DecisionEvent,
    OrderFilledEvent,
    OrderRejectedEvent,
    SignalEvent,
)
from stinger_fx.domain import (
    Decision,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
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
            type=OrderType.MARKET,
            volume=volume,
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

        result = await broker.place_order(req)
        if result.ok and result.order is not None:
            await self.bus.publish(OrderFilledEvent(order=result.order))
        else:
            await self.bus.publish(
                OrderRejectedEvent(
                    order=Order(
                        ticket=result.ticket or 0,
                        strategy_id=signal.strategy_id,
                        symbol=signal.symbol,
                        side=signal.side,
                        type=OrderType.MARKET,
                        volume=volume,
                        sl=signal.suggested_sl,
                        tp=signal.suggested_tp,
                        status=OrderStatus.REJECTED,
                        comment=signal.comment,
                        magic=magic,
                        client_order_id=client_order_id,
                    ),
                    reason=result.message,
                )
            )

    async def attach(self) -> None:
        async def _on_signal(evt: SignalEvent) -> None:
            await self.handle_signal(evt.signal)

        self._sub = self.bus.subscribe(SignalEvent, _on_signal, name="order_router")

    async def detach(self) -> None:
        if hasattr(self, "_sub"):
            await self._sub.unsubscribe()
