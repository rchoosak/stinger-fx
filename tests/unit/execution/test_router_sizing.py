"""OrderRouter risk-based sizing — the sized volume reaches the OrderRequest
when enabled, with safe fallback to the strategy's fixed volume otherwise."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from stinger_fx.brokers.base import BaseBroker
from stinger_fx.config.models import PositionSizingConfig, RiskConfig
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import AccountSnapshotEvent
from stinger_fx.domain import (
    AccountSnapshot,
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

    def __init__(self, bus) -> None:
        super().__init__(bus)
        self.placed: list[OrderRequest] = []

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
    async def get_positions(self): return []
    async def get_open_orders(self): return []


def _signal(*, sl: float | None = 1990.0, entry_ref: float | None = 2000.0) -> Signal:
    return Signal(
        strategy_id="s1",
        time=datetime.now(UTC),
        symbol="XAUUSD",
        side=Side.BUY,
        strength=SignalStrength.NORMAL,
        suggested_volume=0.01,
        suggested_sl=sl,
        entry_ref_price=entry_ref,
    )


async def _risk(*, enabled: bool, equity: float | None) -> RiskMonitor:
    rm = RiskMonitor(
        AsyncEventBus(),
        RiskConfig(position_sizing=PositionSizingConfig(enabled=enabled, risk_per_trade_pct=1.0)),
    )
    await rm.start()
    if equity is not None:
        await rm._on_snapshot(
            AccountSnapshotEvent(
                snapshot=AccountSnapshot(
                    account_id="x", time=datetime.now(UTC), balance=equity,
                    equity=equity, margin=0.0, free_margin=equity,
                )
            )
        )
    return rm


@pytest.mark.asyncio
async def test_enabled_sizes_volume() -> None:
    bus = AsyncEventBus()
    broker = _RecordingBroker(bus)
    rm = await _risk(enabled=True, equity=10_000)
    router = OrderRouter(bus, broker, strategy_magic={"s1": 1}, risk=rm)
    await router.handle_signal(_signal())  # stop 10 → 1% of 10k / (10×100) = 0.10
    assert broker.placed[0].volume == pytest.approx(0.10)
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_disabled_uses_fixed_volume() -> None:
    bus = AsyncEventBus()
    broker = _RecordingBroker(bus)
    rm = await _risk(enabled=False, equity=10_000)
    router = OrderRouter(bus, broker, strategy_magic={"s1": 1}, risk=rm)
    await router.handle_signal(_signal())
    assert broker.placed[0].volume == pytest.approx(0.01)  # suggested_volume
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_missing_sl_falls_back() -> None:
    bus = AsyncEventBus()
    broker = _RecordingBroker(bus)
    rm = await _risk(enabled=True, equity=10_000)
    router = OrderRouter(bus, broker, strategy_magic={"s1": 1}, risk=rm)
    await router.handle_signal(_signal(sl=None))
    assert broker.placed[0].volume == pytest.approx(0.01)
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_no_risk_monitor_falls_back() -> None:
    bus = AsyncEventBus()
    broker = _RecordingBroker(bus)
    router = OrderRouter(bus, broker, strategy_magic={"s1": 1})  # risk=None
    await router.handle_signal(_signal())
    assert broker.placed[0].volume == pytest.approx(0.01)
    await bus.close()


@pytest.mark.asyncio
async def test_equity_unknown_falls_back() -> None:
    bus = AsyncEventBus()
    broker = _RecordingBroker(bus)
    rm = await _risk(enabled=True, equity=None)  # no snapshot → equity None
    router = OrderRouter(bus, broker, strategy_magic={"s1": 1}, risk=rm)
    await router.handle_signal(_signal())
    assert broker.placed[0].volume == pytest.approx(0.01)
    await rm.stop()
    await bus.close()
