"""RiskMonitor — pre-trade enforcement of RiskConfig limits."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from stinger_fx.config.models import RiskConfig
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import (
    AccountSnapshotEvent,
    OrderFilledEvent,
    PositionClosedEvent,
)
from stinger_fx.domain import (
    AccountSnapshot,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
    Signal,
    SignalStrength,
)
from stinger_fx.risk import RiskMonitor


def _signal(strategy_id: str = "s1", *, side: Side = Side.BUY) -> Signal:
    return Signal(
        strategy_id=strategy_id,
        time=datetime.now(UTC),
        symbol="EURUSD",
        side=side,
        strength=SignalStrength.NORMAL,
        suggested_volume=0.1,
    )


def _filled(strategy_id: str = "s1") -> OrderFilledEvent:
    return OrderFilledEvent(
        order=Order(
            ticket=1,
            strategy_id=strategy_id,
            symbol="EURUSD",
            side=Side.BUY,
            type=OrderType.MARKET,
            volume=0.1,
            status=OrderStatus.FILLED,
        )
    )


def _closed(pnl: float, magic: int = 1) -> PositionClosedEvent:
    return PositionClosedEvent(
        position=Position(
            ticket=1,
            symbol="EURUSD",
            side=Side.BUY,
            volume=0.1,
            open_price=1.10,
            open_time=datetime.now(UTC),
            magic=magic,
        ),
        realized_pnl=pnl,
    )


def _snapshot(*, balance: float = 10_000, equity: float = 10_000) -> AccountSnapshotEvent:
    return AccountSnapshotEvent(
        snapshot=AccountSnapshot(
            account_id="x",
            time=datetime.now(UTC),
            balance=balance,
            equity=equity,
            margin=0.0,
            free_margin=equity,
        )
    )


@pytest.mark.asyncio
async def test_allows_signal_with_default_config() -> None:
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, RiskConfig())
    await rm.start()
    assert rm.check_signal(_signal()).allowed is True
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_blocks_when_strategy_at_position_cap() -> None:
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, RiskConfig(max_open_positions_per_strategy=2))
    await rm.start()
    # Two fills count as two open positions
    await rm._on_filled(_filled("s1"))
    await rm._on_filled(_filled("s1"))
    verdict = rm.check_signal(_signal("s1"))
    assert verdict.allowed is False
    assert "max_open_positions_per_strategy" in verdict.reason
    # Other strategies still allowed
    assert rm.check_signal(_signal("s2")).allowed is True
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_close_position_decrements_open_count() -> None:
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, RiskConfig(max_open_positions_per_strategy=1))
    await rm.start()
    await rm._on_filled(_filled("s1"))
    assert rm.check_signal(_signal("s1")).allowed is False
    await rm._on_closed(_closed(pnl=10.0))
    assert rm.check_signal(_signal("s1")).allowed is True
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_blocks_when_daily_loss_breached() -> None:
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, RiskConfig(max_daily_loss_pct=5.0))
    await rm.start()
    # Seed an opening balance via a snapshot
    await rm._on_snapshot(_snapshot(balance=10_000, equity=10_000))
    # Realize a $600 loss → -6% > 5% limit
    await rm._on_closed(_closed(pnl=-600))
    verdict = rm.check_signal(_signal())
    assert verdict.allowed is False
    assert "max_daily_loss_pct" in verdict.reason
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_kill_switch_trips_on_drawdown_and_blocks_everything() -> None:
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, RiskConfig(kill_switch_drawdown_pct=20.0))
    await rm.start()
    # Equity reaches peak 12_000, then drops to 9_000 — 25% drawdown → trip
    await rm._on_snapshot(_snapshot(balance=10_000, equity=12_000))
    await rm._on_snapshot(_snapshot(balance=10_000, equity=9_000))
    snap = rm.snapshot()
    assert snap["kill_switch_tripped"] is True
    # Every signal is now blocked regardless of strategy
    for sid in ("s1", "s2", "s3"):
        assert rm.check_signal(_signal(sid)).allowed is False
    # Manual reset clears it
    rm.reset_kill_switch()
    assert rm.check_signal(_signal("s1")).allowed is True
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_update_config_takes_effect_immediately() -> None:
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, RiskConfig(max_open_positions_per_strategy=5))
    await rm.start()
    await rm._on_filled(_filled("s1"))
    await rm._on_filled(_filled("s1"))
    # Currently 2 open under cap of 5 — allowed
    assert rm.check_signal(_signal("s1")).allowed is True
    # Tighten cap to 1 — should now block
    rm.update_config(RiskConfig(max_open_positions_per_strategy=1))
    assert rm.check_signal(_signal("s1")).allowed is False
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_rejected_decisions_published_by_router(monkeypatch) -> None:
    """OrderRouter should publish a 'rejected' DecisionEvent when risk denies."""
    from stinger_fx.backtest.order_router import OrderRouter
    from stinger_fx.brokers.base import BaseBroker
    from stinger_fx.core.events import DecisionEvent

    bus = AsyncEventBus()
    rm = RiskMonitor(bus, RiskConfig(max_open_positions_per_strategy=1))
    await rm.start()
    # Saturate the strategy
    await rm._on_filled(_filled("s1"))

    class _NoopBroker(BaseBroker):
        name = "noop"

        async def connect(self) -> None: ...
        async def disconnect(self) -> None: ...
        async def is_connected(self) -> bool: return True
        async def get_account_info(self): raise NotImplementedError
        async def get_account_snapshot(self): raise NotImplementedError
        async def get_symbol_info(self, symbol): raise NotImplementedError
        async def list_symbols(self): return []
        async def subscribe_ticks(self, symbol): ...
        async def subscribe_bars(self, symbol, tf): ...
        async def unsubscribe(self, symbol, tf=None): ...
        async def get_history_bars(self, *a, **kw): raise NotImplementedError
        async def get_history_ticks(self, *a, **kw): raise NotImplementedError
        async def place_order(self, req): raise AssertionError("must not be called")
        async def modify_order(self, ticket, **kw): raise NotImplementedError
        async def close_position(self, ticket, volume=None): raise NotImplementedError
        async def cancel_order(self, ticket): raise NotImplementedError
        async def get_positions(self): return []
        async def get_open_orders(self): return []

    decisions: list[DecisionEvent] = []

    async def collect(evt: DecisionEvent) -> None:
        decisions.append(evt)

    bus.subscribe(DecisionEvent, collect)

    router = OrderRouter(bus, _NoopBroker(bus), risk=rm)
    await router.handle_signal(_signal("s1"))

    # Yield so the bus delivers to the subscriber
    import asyncio

    for _ in range(3):
        await asyncio.sleep(0)

    assert len(decisions) == 1
    assert decisions[0].decision.action == "rejected"
    assert "max_open_positions" in decisions[0].decision.reason

    await rm.stop()
    await bus.close()
