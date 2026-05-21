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

logger = logging.getLogger("stinger.engine.router")


class OrderRouter:
    def __init__(
        self,
        bus: AsyncEventBus,
        broker: BaseBroker,
        *,
        strategy_magic: dict[str, int] | None = None,
    ) -> None:
        self.bus = bus
        self.broker = broker
        self.strategy_magic = strategy_magic or {}

    async def handle_signal(self, signal: Signal) -> None:
        client_order_id = str(uuid.uuid4())
        magic = self.strategy_magic.get(signal.strategy_id, 0)
        volume = signal.suggested_volume or 0.01

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

        result = await self.broker.place_order(req)
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
