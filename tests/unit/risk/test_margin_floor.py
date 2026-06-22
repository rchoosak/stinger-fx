"""RiskMonitor margin floor — block new orders when margin headroom is thin."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from stinger_fx.config.models import MarginFloorConfig, RiskConfig
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import AccountSnapshotEvent
from stinger_fx.domain import AccountSnapshot, Side, Signal, SignalStrength
from stinger_fx.risk import RiskMonitor


def _snap(*, margin_level: float, free_margin: float) -> AccountSnapshotEvent:
    return AccountSnapshotEvent(
        snapshot=AccountSnapshot(
            account_id="x", time=datetime.now(UTC), balance=10_000.0,
            equity=10_000.0, margin=1_000.0, free_margin=free_margin,
            margin_level=margin_level,
        )
    )


def _signal() -> Signal:
    return Signal(
        strategy_id="s1", time=datetime.now(UTC), symbol="EURUSD",
        side=Side.BUY, strength=SignalStrength.NORMAL, suggested_volume=0.1,
    )


def _cfg(**over) -> RiskConfig:
    # other account gates off so the margin floor is isolated.
    return RiskConfig(
        max_daily_loss_pct=0.0,
        kill_switch_drawdown_pct=0.0,
        margin_floor=MarginFloorConfig(enabled=True, **over),
    )


@pytest.mark.asyncio
async def test_margin_level_below_floor_blocks() -> None:
    rm = RiskMonitor(AsyncEventBus(), _cfg(min_margin_level_pct=200.0))
    await rm.start()
    await rm._on_snapshot(_snap(margin_level=150.0, free_margin=5_000.0))
    v = rm.check_signal(_signal())
    assert v.allowed is False and "min_margin_level_pct" in v.reason
    await rm.stop()


@pytest.mark.asyncio
async def test_margin_level_above_floor_allowed() -> None:
    rm = RiskMonitor(AsyncEventBus(), _cfg(min_margin_level_pct=200.0))
    await rm.start()
    await rm._on_snapshot(_snap(margin_level=250.0, free_margin=5_000.0))
    assert rm.check_signal(_signal()).allowed is True
    await rm.stop()


@pytest.mark.asyncio
async def test_zero_margin_level_is_allowed() -> None:
    # margin_level 0 = no open margin → never a shortage (fail-open).
    rm = RiskMonitor(AsyncEventBus(), _cfg(min_margin_level_pct=200.0))
    await rm.start()
    await rm._on_snapshot(_snap(margin_level=0.0, free_margin=10_000.0))
    assert rm.check_signal(_signal()).allowed is True
    await rm.stop()


@pytest.mark.asyncio
async def test_free_margin_below_floor_blocks() -> None:
    rm = RiskMonitor(AsyncEventBus(), _cfg(min_free_margin=2_000.0))
    await rm.start()
    await rm._on_snapshot(_snap(margin_level=300.0, free_margin=500.0))
    v = rm.check_signal(_signal())
    assert v.allowed is False and "min_free_margin" in v.reason
    await rm.stop()


@pytest.mark.asyncio
async def test_no_snapshot_yet_is_allowed() -> None:
    # Before the first snapshot the margin fields are None → fail-open.
    rm = RiskMonitor(AsyncEventBus(), _cfg(min_margin_level_pct=200.0, min_free_margin=2_000.0))
    await rm.start()
    assert rm.check_signal(_signal()).allowed is True
    await rm.stop()


@pytest.mark.asyncio
async def test_disabled_is_allowed() -> None:
    cfg = RiskConfig(
        max_daily_loss_pct=0.0, kill_switch_drawdown_pct=0.0,
        margin_floor=MarginFloorConfig(enabled=False, min_margin_level_pct=200.0),
    )
    rm = RiskMonitor(AsyncEventBus(), cfg)
    await rm.start()
    await rm._on_snapshot(_snap(margin_level=50.0, free_margin=100.0))
    assert rm.check_signal(_signal()).allowed is True
    await rm.stop()
