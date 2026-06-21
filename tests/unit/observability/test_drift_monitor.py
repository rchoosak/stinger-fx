"""DriftMonitor — alerts when recent live win-rate/expectancy degrades vs the
backtest baseline, with hysteresis."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from stinger_fx.config.models import DriftMonitorConfig
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import PositionClosedEvent, StrategyDriftEvent
from stinger_fx.data import BacktestRepo, TradeRepo, in_memory_store
from stinger_fx.domain import Position, Side
from stinger_fx.observability.drift_monitor import DriftMonitor

BASE = datetime(2024, 1, 1, tzinfo=UTC)


def _seed_baseline(
    store, *, win_rate: float, expectancy_per_lot: float, sid: str = "s1"
) -> None:
    repo = BacktestRepo(store)
    rid = repo.start_run("bt", sid, {})
    repo.finish_run(
        rid,
        {"win_rate": win_rate, "expectancy_per_lot": expectancy_per_lot},
        "bt.json",
    )


class _TradeSeeder:
    def __init__(self, store) -> None:
        self._repo = TradeRepo(store)
        self._i = 0

    def add(self, *, sid: str, pnl: float) -> None:
        self._i += 1
        self._repo.add(
            position_id=self._i, strategy_id=sid, symbol="XAUUSD", side="buy",
            open_ts=BASE + timedelta(minutes=self._i),
            close_ts=BASE + timedelta(minutes=self._i),
            open_price=1.0, close_price=1.1, volume=0.1, pnl=pnl,
        )


def _close_evt() -> PositionClosedEvent:
    return PositionClosedEvent(
        position=Position(
            ticket=1, symbol="XAUUSD", side=Side.BUY, volume=0.1,
            open_price=1.0, open_time=BASE, magic=7,
        ),
        realized_pnl=0.0,
    )


def _monitor(store, bus, **over) -> tuple[DriftMonitor, list[StrategyDriftEvent]]:
    cfg = DriftMonitorConfig(enabled=True, window=5, min_trades=5, **over)
    dm = DriftMonitor(bus, store, strategy_for_magic=lambda m: "s1", cfg=cfg)
    events: list[StrategyDriftEvent] = []
    bus.subscribe(StrategyDriftEvent, lambda e: events.append(e), name="probe")
    return dm, events


async def _yield() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_degraded_win_rate_emits_drift() -> None:
    store = in_memory_store()
    _seed_baseline(store, win_rate=0.5, expectancy_per_lot=0.0)  # isolate win-rate
    seeder = _TradeSeeder(store)
    for _ in range(5):  # all losers → live win-rate 0.0 < 0.5*0.7
        seeder.add(sid="s1", pnl=-1.0)
    bus = AsyncEventBus()
    dm, events = _monitor(store, bus)
    await dm._on_closed(_close_evt())
    await _yield()
    assert len(events) == 1
    assert "win-rate" in events[0].reason
    assert events[0].sample_size == 5
    await bus.close()


@pytest.mark.asyncio
async def test_degraded_expectancy_emits_drift() -> None:
    store = in_memory_store()
    _seed_baseline(store, win_rate=0.5, expectancy_per_lot=10.0)
    seeder = _TradeSeeder(store)
    # win-rate healthy (0.6) but per-lot expectancy negative:
    # pnls/0.1 = [10,10,10,-15,-15] → mean -2 < floor 10*0.5
    for pnl in [1.0, 1.0, 1.0, -1.5, -1.5]:
        seeder.add(sid="s1", pnl=pnl)
    bus = AsyncEventBus()
    dm, events = _monitor(store, bus)
    await dm._on_closed(_close_evt())
    await _yield()
    assert len(events) == 1
    assert "expectancy" in events[0].reason
    await bus.close()


@pytest.mark.asyncio
async def test_healthy_no_event() -> None:
    store = in_memory_store()
    _seed_baseline(store, win_rate=0.5, expectancy_per_lot=10.0)
    seeder = _TradeSeeder(store)
    for _ in range(5):  # all winners, per-lot exp 120 ≥ floor
        seeder.add(sid="s1", pnl=12.0)
    bus = AsyncEventBus()
    dm, events = _monitor(store, bus)
    await dm._on_closed(_close_evt())
    await _yield()
    assert events == []
    await bus.close()


@pytest.mark.asyncio
async def test_below_min_trades_no_event() -> None:
    store = in_memory_store()
    _seed_baseline(store, win_rate=0.5, expectancy_per_lot=10.0)
    seeder = _TradeSeeder(store)
    for _ in range(4):  # < min_trades (5)
        seeder.add(sid="s1", pnl=-1.0)
    bus = AsyncEventBus()
    dm, events = _monitor(store, bus)
    await dm._on_closed(_close_evt())
    await _yield()
    assert events == []
    await bus.close()


@pytest.mark.asyncio
async def test_no_baseline_no_event() -> None:
    store = in_memory_store()  # no backtest run seeded
    seeder = _TradeSeeder(store)
    for _ in range(5):
        seeder.add(sid="s1", pnl=-1.0)
    bus = AsyncEventBus()
    dm, events = _monitor(store, bus)
    await dm._on_closed(_close_evt())
    await _yield()
    assert events == []
    await bus.close()


@pytest.mark.asyncio
async def test_hysteresis_one_alert_then_rearm_on_recovery() -> None:
    store = in_memory_store()
    _seed_baseline(store, win_rate=0.5, expectancy_per_lot=0.0)  # isolate win-rate
    seeder = _TradeSeeder(store)
    bus = AsyncEventBus()
    dm, events = _monitor(store, bus)

    for _ in range(5):  # degraded (losers)
        seeder.add(sid="s1", pnl=-1.0)
    await dm._on_closed(_close_evt())
    await _yield()
    assert len(events) == 1  # first alert

    await dm._on_closed(_close_evt())  # still degraded, no new trades
    await _yield()
    assert len(events) == 1  # silenced by hysteresis

    for _ in range(5):  # recovery — last 5 are winners now
        seeder.add(sid="s1", pnl=12.0)
    await dm._on_closed(_close_evt())
    await _yield()
    assert len(events) == 1  # healthy → re-armed, no event

    for _ in range(5):  # degrade again
        seeder.add(sid="s1", pnl=-1.0)
    await dm._on_closed(_close_evt())
    await _yield()
    assert len(events) == 2  # re-alerts
    await bus.close()
