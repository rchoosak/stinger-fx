"""RiskMonitor crash recovery — rehydrate() + persisted peak/kill-switch.

These cover the Tier-1 production-safety gap: after an engine restart the
RiskMonitor must rebuild its in-memory state (open positions, today's realized
P&L) and reload its persisted derived state (peak equity, a tripped kill
switch) so the safety limits are NOT silently reset.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from stinger_fx.config.models import RiskConfig
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import AccountSnapshotEvent, PositionClosedEvent
from stinger_fx.data import RiskStateRepo, in_memory_store
from stinger_fx.domain import AccountSnapshot, Position, Side, Signal, SignalStrength
from stinger_fx.risk import RiskMonitor


def _signal(strategy_id: str = "s1", symbol: str = "EURUSD") -> Signal:
    return Signal(
        strategy_id=strategy_id,
        time=datetime.now(UTC),
        symbol=symbol,
        side=Side.BUY,
        strength=SignalStrength.NORMAL,
        suggested_volume=0.1,
    )


def _snapshot(*, balance: float, equity: float) -> AccountSnapshotEvent:
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


def _closed(ticket: int, *, symbol: str = "XAUUSD", pnl: float = 0.0) -> PositionClosedEvent:
    return PositionClosedEvent(
        position=Position(
            ticket=ticket,
            symbol=symbol,
            side=Side.BUY,
            volume=0.1,
            open_price=1.1,
            open_time=datetime.now(UTC),
        ),
        realized_pnl=pnl,
    )


# --- open-position rehydration --------------------------------------------


@pytest.mark.asyncio
async def test_rehydrate_rebuilds_open_position_counts() -> None:
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, RiskConfig(max_open_positions_per_strategy=2))
    await rm.start()
    await rm.rehydrate(
        open_positions=[
            ("s1", "XAUUSD", 101),
            ("s1", "XAUUSD", 102),
            ("s2", "EURUSD", 103),
            (None, "GBPUSD", 104),  # unknown magic — symbol bucket only
        ],
        daily_realized_pnl=0.0,
        daily_pnl_by_symbol={},
    )
    snap = rm.snapshot()
    assert snap["open_positions"] == {"s1": 2, "s2": 1}
    assert snap["open_positions_by_symbol"] == {"XAUUSD": 2, "EURUSD": 1, "GBPUSD": 1}

    # s1 is at its cap of 2 → blocked; s2 (1 open) still allowed.
    assert rm.check_signal(_signal("s1")).allowed is False
    assert rm.check_signal(_signal("s2")).allowed is True
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_rehydrated_ticket_maps_back_to_strategy_on_close() -> None:
    """ticket→strategy must be rebuilt so a later close decrements the right
    bucket rather than logging an unknown-ticket warning."""
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, RiskConfig(max_open_positions_per_strategy=2))
    await rm.start()
    await rm.rehydrate(
        open_positions=[("s1", "XAUUSD", 101), ("s1", "XAUUSD", 102)],
        daily_realized_pnl=0.0,
        daily_pnl_by_symbol={},
    )
    assert rm.check_signal(_signal("s1")).allowed is False  # 2/2 open
    await rm._on_closed(_closed(101))
    assert rm.check_signal(_signal("s1")).allowed is True  # back to 1/2
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_rehydrate_with_unknown_magic_does_not_crash() -> None:
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, RiskConfig())
    await rm.start()
    await rm.rehydrate(
        open_positions=[(None, "GBPUSD", 104)],
        daily_realized_pnl=0.0,
        daily_pnl_by_symbol={},
    )
    snap = rm.snapshot()
    assert snap["open_positions"] == {}
    assert snap["open_positions_by_symbol"] == {"GBPUSD": 1}
    await rm.stop()
    await bus.close()


# --- daily P&L rehydration + opening-balance reconstruction ----------------


@pytest.mark.asyncio
async def test_rehydrate_restores_daily_loss_and_blocks() -> None:
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, RiskConfig(max_daily_loss_pct=5.0))
    await rm.start()
    # Today already lost $600 before the restart.
    await rm.rehydrate(
        open_positions=[],
        daily_realized_pnl=-600.0,
        daily_pnl_by_symbol={"XAUUSD": -600.0},
    )
    # First snapshot post-restart: balance already reflects the loss. The
    # day's opening balance must be reconstructed to 10_000 (9_400 + 600).
    await rm._on_snapshot(_snapshot(balance=9_400, equity=9_400))
    snap = rm.snapshot()
    assert snap["daily_opening_balance"] == 10_000.0
    assert snap["daily_realized_pnl"] == -600.0
    # -600 / 10_000 = 6% > 5% limit → blocked.
    verdict = rm.check_signal(_signal())
    assert verdict.allowed is False
    assert "max_daily_loss_pct" in verdict.reason
    await rm.stop()
    await bus.close()


# --- persisted derived state (peak equity + kill switch) -------------------


@pytest.mark.asyncio
async def test_persisted_tripped_kill_switch_survives_restart() -> None:
    """The whole point: a tripped kill switch must NOT be cleared by a restart."""
    store = in_memory_store()
    repo = RiskStateRepo(store)
    repo.save(peak_equity=12_000.0, kill_switch_tripped=True)

    bus = AsyncEventBus()
    rm = RiskMonitor(bus, RiskConfig(kill_switch_drawdown_pct=20.0), state_repo=repo)
    await rm.start()
    await rm.rehydrate(open_positions=[], daily_realized_pnl=0.0, daily_pnl_by_symbol={})

    assert rm.snapshot()["kill_switch_tripped"] is True
    for sid in ("s1", "s2", "s3"):
        v = rm.check_signal(_signal(sid))
        assert v.allowed is False
        assert "kill_switch_tripped" in v.reason
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_persisted_peak_equity_is_drawdown_baseline() -> None:
    """Drawdown must be measured against the true (persisted) peak, not the
    restart-time equity — otherwise restarting mid-drawdown hides the loss."""
    store = in_memory_store()
    repo = RiskStateRepo(store)
    repo.save(peak_equity=12_000.0, kill_switch_tripped=False)

    bus = AsyncEventBus()
    rm = RiskMonitor(bus, RiskConfig(kill_switch_drawdown_pct=20.0), state_repo=repo)
    await rm.start()
    await rm.rehydrate(open_positions=[], daily_realized_pnl=0.0, daily_pnl_by_symbol={})
    # Equity 9_000 vs persisted peak 12_000 = 25% drawdown → trips. Without the
    # persisted peak, the first snapshot would set peak=9_000 and never trip.
    await rm._on_snapshot(_snapshot(balance=10_000, equity=9_000))
    assert rm.snapshot()["kill_switch_tripped"] is True
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_peak_and_trip_are_persisted_for_next_restart() -> None:
    store = in_memory_store()
    repo = RiskStateRepo(store)

    bus = AsyncEventBus()
    rm = RiskMonitor(bus, RiskConfig(kill_switch_drawdown_pct=20.0), state_repo=repo)
    await rm.start()

    # New high → peak persisted.
    await rm._on_snapshot(_snapshot(balance=10_000, equity=11_000))
    assert repo.load().peak_equity == 11_000.0  # type: ignore[union-attr]

    # 27% drawdown → trip persisted.
    await rm._on_snapshot(_snapshot(balance=10_000, equity=8_000))
    assert repo.load().kill_switch_tripped is True  # type: ignore[union-attr]

    # Operator reset → persisted back to False.
    rm.reset_kill_switch()
    assert repo.load().kill_switch_tripped is False  # type: ignore[union-attr]
    await rm.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_rehydrate_without_state_repo_is_safe() -> None:
    """Backtest / unit path passes no repo — rehydrate must still work."""
    bus = AsyncEventBus()
    rm = RiskMonitor(bus, RiskConfig())  # no state_repo
    await rm.start()
    await rm.rehydrate(
        open_positions=[("s1", "XAUUSD", 1)],
        daily_realized_pnl=-5.0,
        daily_pnl_by_symbol={"XAUUSD": -5.0},
    )
    assert rm.snapshot()["open_positions"] == {"s1": 1}
    await rm.stop()
    await bus.close()
