"""RiskMonitor — per-symbol position and daily-loss limits."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from stinger_fx.config.models import RiskConfig, SymbolRiskConfig
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import OrderFilledEvent, PartialClosedEvent, PositionClosedEvent
from stinger_fx.domain import Order, OrderStatus, OrderType, Position, Side, Signal, SignalStrength
from stinger_fx.risk.monitor import RiskMonitor


def _make_signal(symbol: str = "EURUSD", strategy_id: str = "strat") -> Signal:
    return Signal(
        strategy_id=strategy_id,
        time=datetime(2024, 1, 1, tzinfo=UTC),
        symbol=symbol,
        side=Side.BUY,
        strength=SignalStrength.NORMAL,
        suggested_volume=0.1,
    )


def _make_filled(symbol: str, strategy_id: str = "strat") -> OrderFilledEvent:
    order = Order(
        ticket=1,
        strategy_id=strategy_id,
        symbol=symbol,
        side=Side.BUY,
        type=OrderType.MARKET,
        volume=0.1,
        status=OrderStatus.FILLED,
        fill_price=1.10,
        filled_at=datetime(2024, 1, 1, tzinfo=UTC),
        magic=0,
    )
    return OrderFilledEvent(order=order)


def _make_closed(symbol: str, pnl: float) -> PositionClosedEvent:
    pos = Position(
        ticket=1,
        symbol=symbol,
        side=Side.BUY,
        volume=0.1,
        open_price=1.10,
        open_time=datetime(2024, 1, 1, tzinfo=UTC),
        magic=0,
    )
    return PositionClosedEvent(position=pos, realized_pnl=pnl)


async def _drain(bus: AsyncEventBus) -> None:
    for _ in range(3):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_per_symbol_max_positions_blocks_new_signal() -> None:
    """After max_open_positions on EURUSD is reached, further EURUSD signals
    must be rejected."""
    bus = AsyncEventBus()
    cfg = RiskConfig(
        max_open_positions_per_strategy=10,   # account-wide is high
        per_symbol={"EURUSD": SymbolRiskConfig(max_open_positions=1)},
    )
    monitor = RiskMonitor(bus, cfg)
    await monitor.start()

    try:
        # First signal — no positions yet → allowed
        v1 = monitor.check_signal(_make_signal("EURUSD"))
        assert v1.allowed

        # Simulate one EURUSD fill
        await bus.publish(_make_filled("EURUSD"))
        await _drain(bus)

        # Second EURUSD signal — cap=1 reached → rejected
        v2 = monitor.check_signal(_make_signal("EURUSD"))
        assert not v2.allowed
        assert "per_symbol.max_open_positions=1" in v2.reason

        # GBPUSD is unaffected — should still be allowed
        v3 = monitor.check_signal(_make_signal("GBPUSD"))
        assert v3.allowed
    finally:
        await monitor.stop()
        await bus.close()


@pytest.mark.asyncio
async def test_per_symbol_counter_decrements_on_close() -> None:
    """Closing a position must decrement the per-symbol counter so a new
    signal can be accepted again."""
    bus = AsyncEventBus()
    cfg = RiskConfig(
        max_open_positions_per_strategy=10,
        per_symbol={"EURUSD": SymbolRiskConfig(max_open_positions=1)},
    )
    monitor = RiskMonitor(bus, cfg)
    await monitor.start()

    try:
        await bus.publish(_make_filled("EURUSD"))
        await _drain(bus)

        # Cap reached
        assert not monitor.check_signal(_make_signal("EURUSD")).allowed

        # Close the position
        await bus.publish(_make_closed("EURUSD", pnl=5.0))
        await _drain(bus)

        # Cap released → allowed again
        assert monitor.check_signal(_make_signal("EURUSD")).allowed
    finally:
        await monitor.stop()
        await bus.close()


@pytest.mark.asyncio
async def test_per_symbol_daily_loss_blocks_signal() -> None:
    """Once the per-symbol daily loss exceeds max_daily_loss_usd, new signals
    on that symbol must be rejected."""
    bus = AsyncEventBus()
    cfg = RiskConfig(
        max_daily_loss_pct=100.0,   # account-wide very high
        per_symbol={"EURUSD": SymbolRiskConfig(max_daily_loss_usd=50.0)},
    )
    monitor = RiskMonitor(bus, cfg)
    await monitor.start()

    try:
        # Simulate two losing closes totalling $60
        await bus.publish(_make_closed("EURUSD", pnl=-30.0))
        await _drain(bus)
        await bus.publish(_make_closed("EURUSD", pnl=-30.0))
        await _drain(bus)

        v = monitor.check_signal(_make_signal("EURUSD"))
        assert not v.allowed
        assert "per_symbol.max_daily_loss_usd=50.0" in v.reason

        # GBPUSD is unaffected
        assert monitor.check_signal(_make_signal("GBPUSD")).allowed
    finally:
        await monitor.stop()
        await bus.close()


@pytest.mark.asyncio
async def test_partial_close_loss_counts_without_releasing_position_cap() -> None:
    bus = AsyncEventBus()
    cfg = RiskConfig(
        max_open_positions_per_strategy=1,
        per_symbol={
            "EURUSD": SymbolRiskConfig(
                max_open_positions=1,
                max_daily_loss_usd=50.0,
            )
        },
    )
    monitor = RiskMonitor(bus, cfg)
    await monitor.start()

    try:
        await bus.publish(_make_filled("EURUSD"))
        await _drain(bus)
        pos = _make_closed("EURUSD", pnl=0.0).position.model_copy(
            update={"volume": 0.05}
        )
        await bus.publish(
            PartialClosedEvent(
                position=pos,
                closed_volume=0.05,
                realized_pnl=-60.0,
            )
        )
        await _drain(bus)

        snap = monitor.snapshot()
        assert snap["daily_realized_pnl"] == pytest.approx(-60.0)
        assert snap["daily_pnl_by_symbol"] == {"EURUSD": pytest.approx(-60.0)}
        assert snap["open_positions"] == {"strat": 1}
        assert snap["open_positions_by_symbol"] == {"EURUSD": 1}
        assert not monitor.check_signal(_make_signal("EURUSD")).allowed
    finally:
        await monitor.stop()
        await bus.close()


@pytest.mark.asyncio
async def test_per_symbol_no_config_is_permissive() -> None:
    """When per_symbol is empty all symbols should pass the per-symbol checks."""
    bus = AsyncEventBus()
    cfg = RiskConfig(per_symbol={})
    monitor = RiskMonitor(bus, cfg)
    await monitor.start()

    try:
        for sym in ("EURUSD", "GBPUSD", "USDJPY"):
            assert monitor.check_signal(_make_signal(sym)).allowed
    finally:
        await monitor.stop()
        await bus.close()
