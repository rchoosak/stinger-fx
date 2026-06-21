"""OrderRouter engine-level trading filter — blocks orders on bad-fill
conditions and records a rejected DecisionEvent, else places normally."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from stinger_fx.brokers.base import BaseBroker
from stinger_fx.config.models import RiskConfig, TradingFilterConfig
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import DecisionEvent
from stinger_fx.domain import (
    OrderRequest,
    OrderResult,
    OrderStatus,
    Side,
    Signal,
    SignalStrength,
    SymbolInfo,
)
from stinger_fx.execution.order_router import OrderRouter
from stinger_fx.risk import RiskMonitor


class _RecordingBroker(BaseBroker):
    name = "rec"

    def __init__(self, bus, *, spread: int = 0) -> None:
        super().__init__(bus)
        self.placed: list[OrderRequest] = []
        self._spread = spread

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def is_connected(self) -> bool: return True
    async def get_account_info(self): raise NotImplementedError
    async def get_account_snapshot(self): raise NotImplementedError
    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        return SymbolInfo(
            symbol=symbol, digits=3, point=0.001, contract_size=100.0,
            volume_min=0.01, volume_max=100.0, volume_step=0.01, spread=self._spread,
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
    async def get_positions(self): return []
    async def get_open_orders(self): return []


def _signal(*, when: datetime | None = None) -> Signal:
    return Signal(
        strategy_id="s1",
        time=when or datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        symbol="XAUUSD",
        side=Side.BUY,
        strength=SignalStrength.NORMAL,
        suggested_volume=0.01,
    )


async def _router(bus, broker, tf: TradingFilterConfig) -> tuple[OrderRouter, list]:
    rm = RiskMonitor(bus, RiskConfig(trading_filter=tf))
    await rm.start()
    decisions: list[DecisionEvent] = []
    bus.subscribe(DecisionEvent, lambda e: decisions.append(e), name="probe")
    return OrderRouter(bus, broker, strategy_magic={"s1": 1}, risk=rm), decisions


async def _yield() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_spread_over_cap_blocks_and_records_rejection() -> None:
    bus = AsyncEventBus()
    broker = _RecordingBroker(bus, spread=50)
    router, decisions = await _router(
        bus, broker, TradingFilterConfig(enabled=True, max_spread_points=20)
    )
    await router.handle_signal(_signal())
    await _yield()
    assert broker.placed == []  # blocked
    rejected = [d for d in decisions if d.decision.action == "rejected"]
    assert len(rejected) == 1
    assert "trading_filter" in rejected[0].decision.reason
    assert "spread" in rejected[0].decision.reason
    await bus.close()


@pytest.mark.asyncio
async def test_spread_under_cap_places() -> None:
    bus = AsyncEventBus()
    broker = _RecordingBroker(bus, spread=10)
    router, _ = await _router(
        bus, broker, TradingFilterConfig(enabled=True, max_spread_points=20)
    )
    await router.handle_signal(_signal())
    assert len(broker.placed) == 1
    await bus.close()


@pytest.mark.asyncio
async def test_disabled_filter_places() -> None:
    bus = AsyncEventBus()
    broker = _RecordingBroker(bus, spread=9999)
    router, _ = await _router(
        bus, broker, TradingFilterConfig(enabled=False, max_spread_points=1)
    )
    await router.handle_signal(_signal())
    assert len(broker.placed) == 1
    await bus.close()


@pytest.mark.asyncio
async def test_rollover_window_blocks_on_signal_time() -> None:
    bus = AsyncEventBus()
    broker = _RecordingBroker(bus)
    router, decisions = await _router(
        bus,
        broker,
        TradingFilterConfig(
            enabled=True, block_rollover=True,
            rollover_hour_utc=21, rollover_block_minutes=5,
        ),
    )
    await router.handle_signal(_signal(when=datetime(2024, 1, 1, 21, 2, tzinfo=UTC)))
    await _yield()
    assert broker.placed == []
    assert any("rollover" in d.decision.reason for d in decisions)
    await bus.close()
