"""OrderRouter aggregate open-risk cap — block a new order when total open
risk (existing positions + this order) would exceed equity x cap percent."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from stinger_fx.brokers.base import BaseBroker
from stinger_fx.config.models import RiskConfig
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import AccountSnapshotEvent
from stinger_fx.domain import (
    AccountSnapshot,
    OrderRequest,
    OrderResult,
    OrderStatus,
    Position,
    Side,
    Signal,
    SignalStrength,
    SymbolInfo,
)
from stinger_fx.execution.order_router import OrderRouter
from stinger_fx.risk import RiskMonitor


class _Broker(BaseBroker):
    name = "rec"

    def __init__(self, bus, positions: list[Position]) -> None:
        super().__init__(bus)
        self.placed: list[OrderRequest] = []
        self._positions = positions

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def is_connected(self) -> bool: return True
    async def get_account_info(self): raise NotImplementedError
    async def get_account_snapshot(self): raise NotImplementedError
    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        return SymbolInfo(
            symbol=symbol, digits=3, point=0.001, contract_size=100.0,
            volume_min=0.01, volume_max=100.0, volume_step=0.01,
            currency_base="XAU", currency_profit="USD", currency_margin="USD",
        )
    async def list_symbols(self): return []
    async def subscribe_ticks(self, symbol): ...
    async def subscribe_bars(self, symbol, tf): ...
    async def unsubscribe(self, symbol, tf=None): ...
    async def get_history_bars(self, *a, **kw): raise NotImplementedError
    async def get_history_ticks(self, *a, **kw): raise NotImplementedError
    async def place_order(self, req: OrderRequest) -> OrderResult:
        self.placed.append(req)
        return OrderResult(ok=True, ticket=1, status=OrderStatus.FILLED, order=None)
    async def modify_order(self, ticket, **kw): raise NotImplementedError
    async def close_position(self, ticket, volume=None): raise NotImplementedError
    async def cancel_order(self, ticket): raise NotImplementedError
    async def get_positions(self): return list(self._positions)
    async def get_open_orders(self): return []


def _open_pos(*, open_price: float, sl: float, volume: float) -> Position:
    return Position(
        ticket=99, symbol="XAUUSD", side=Side.BUY, volume=volume,
        open_price=open_price, open_time=datetime.now(UTC), sl=sl,
    )


def _signal(*, sl: float | None = 1990.0) -> Signal:
    return Signal(
        strategy_id="s1", time=datetime.now(UTC), symbol="XAUUSD",
        side=Side.BUY, strength=SignalStrength.NORMAL,
        suggested_volume=0.1, suggested_sl=sl, entry_ref_price=2000.0,
    )


async def _router(broker, *, max_pct: float) -> tuple[OrderRouter, RiskMonitor]:
    bus = broker.bus
    rm = RiskMonitor(bus, RiskConfig(max_aggregate_risk_pct=max_pct))
    await rm.start()
    await rm._on_snapshot(
        AccountSnapshotEvent(
            snapshot=AccountSnapshot(
                account_id="x", time=datetime.now(UTC), balance=10_000.0,
                equity=10_000.0, margin=0.0, free_margin=10_000.0,
            )
        )
    )
    return OrderRouter(bus, broker, strategy_magic={"s1": 1}, risk=rm), rm


@pytest.mark.asyncio
async def test_under_cap_places() -> None:
    # existing 100 + new 100 = 200 < cap 500 (5% of 10k) → placed.
    bus = AsyncEventBus()
    broker = _Broker(bus, [_open_pos(open_price=2000.0, sl=1990.0, volume=0.1)])
    router, rm = await _router(broker, max_pct=5.0)
    await router.handle_signal(_signal())
    assert len(broker.placed) == 1
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_over_cap_blocks() -> None:
    # existing 100 + new 100 = 200 > cap 100 (1% of 10k) → rejected.
    bus = AsyncEventBus()
    broker = _Broker(bus, [_open_pos(open_price=2000.0, sl=1990.0, volume=0.1)])
    router, rm = await _router(broker, max_pct=1.0)
    await router.handle_signal(_signal())
    assert broker.placed == []
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_disabled_places() -> None:
    # max_pct 0 = off → placed even with a big existing position.
    bus = AsyncEventBus()
    broker = _Broker(bus, [_open_pos(open_price=2000.0, sl=1000.0, volume=10.0)])
    router, rm = await _router(broker, max_pct=0.0)
    await router.handle_signal(_signal())
    assert len(broker.placed) == 1
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_stopless_new_order_not_gated() -> None:
    # New order with no SL can't be bounded → not gated even under a tiny cap.
    bus = AsyncEventBus()
    broker = _Broker(bus, [_open_pos(open_price=2000.0, sl=1990.0, volume=0.1)])
    router, rm = await _router(broker, max_pct=0.5)  # cap 50 < existing 100
    await router.handle_signal(_signal(sl=None))
    assert len(broker.placed) == 1
    await rm.stop()
    await bus.close()
