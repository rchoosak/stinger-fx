"""RiskMonitor profit-lock — lock in gains, trip on giveback, persist + reset."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from stinger_fx.config.models import ProfitLockConfig, RiskConfig
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import AccountSnapshotEvent
from stinger_fx.data import RiskStateRepo, in_memory_store
from stinger_fx.domain import AccountSnapshot, Side, Signal, SignalStrength
from stinger_fx.risk import RiskMonitor


def _snap(equity: float) -> AccountSnapshotEvent:
    return AccountSnapshotEvent(
        snapshot=AccountSnapshot(
            account_id="x", time=datetime.now(UTC), balance=equity,
            equity=equity, margin=0.0, free_margin=equity,
        )
    )


def _signal() -> Signal:
    return Signal(
        strategy_id="s1", time=datetime.now(UTC), symbol="EURUSD",
        side=Side.BUY, strength=SignalStrength.NORMAL, suggested_volume=0.1,
    )


def _cfg(**over) -> RiskConfig:
    # kill switch off so profit-lock is isolated.
    return RiskConfig(
        kill_switch_drawdown_pct=0.0,
        profit_lock=ProfitLockConfig(enabled=True, activate_pct=10.0, giveback_pct=50.0, **over),
    )


@pytest.mark.asyncio
async def test_armed_then_giveback_trips_and_blocks() -> None:
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, _cfg())
    await rm.start()
    await rm._on_snapshot(_snap(10_000))   # session open
    await rm._on_snapshot(_snap(12_000))   # high, armed (>= 11_000); gain 2_000, floor 11_000
    assert rm.snapshot()["profit_lock_tripped"] is False
    await rm._on_snapshot(_snap(10_900))   # < floor 11_000 → trip
    assert rm.snapshot()["profit_lock_tripped"] is True
    v = rm.check_signal(_signal())
    assert v.allowed is False and "profit_lock_tripped" in v.reason
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_not_armed_no_trip() -> None:
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, _cfg())
    await rm.start()
    await rm._on_snapshot(_snap(10_000))
    await rm._on_snapshot(_snap(10_500))   # high < 11_000 arm level → never armed
    await rm._on_snapshot(_snap(9_000))    # big drop, but not armed → no trip
    assert rm.snapshot()["profit_lock_tripped"] is False
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_giveback_within_allowance_no_trip() -> None:
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, _cfg())
    await rm.start()
    await rm._on_snapshot(_snap(10_000))
    await rm._on_snapshot(_snap(12_000))   # floor 11_000
    await rm._on_snapshot(_snap(11_500))   # gave back 500 < 1_000 allowance → ok
    assert rm.snapshot()["profit_lock_tripped"] is False
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_reset_rebases_and_clears() -> None:
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, _cfg())
    await rm.start()
    await rm._on_snapshot(_snap(10_000))
    await rm._on_snapshot(_snap(12_000))
    await rm._on_snapshot(_snap(10_900))   # trip
    assert rm.snapshot()["profit_lock_tripped"] is True
    rm.reset_profit_lock()
    assert rm.snapshot()["profit_lock_tripped"] is False
    # Re-armed from 10_900 — a same-equity snapshot doesn't re-trip.
    await rm._on_snapshot(_snap(10_900))
    assert rm.snapshot()["profit_lock_tripped"] is False
    assert rm.check_signal(_signal()).allowed is True
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_persisted_trip_survives_restart() -> None:
    store = in_memory_store()
    repo = RiskStateRepo(store)
    repo.save(peak_equity=12_000.0, kill_switch_tripped=False, profit_lock_tripped=True)
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, _cfg(), state_repo=repo)
    await rm.start()
    await rm.rehydrate(open_positions=[], daily_realized_pnl=0.0, daily_pnl_by_symbol={})
    assert rm.snapshot()["profit_lock_tripped"] is True
    assert rm.check_signal(_signal()).allowed is False
    await rm.stop()
    await bus.close()
