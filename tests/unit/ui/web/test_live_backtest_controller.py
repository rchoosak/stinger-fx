"""Lifecycle tests for ``LiveBacktestController``.

Three properties under test:
  1. ``start()`` spawns an asyncio.Task and ``is_running()`` flips to True.
  2. ``stop()`` cancels cleanly and the controller is ready to start again.
  3. Concurrent ``start()`` calls / unknown run ids raise
     ``LiveBacktestError`` (the HTTP layer converts to 4xx).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stinger_fx.brokers.pool import BrokerPool
from stinger_fx.config.models import (
    AppConfig,
    BacktestConfig,
    BacktestRunConfig,
    BrokerConfig,
    FullConfig,
    StrategiesConfig,
    StrategyEntry,
)
from stinger_fx.core import AsyncEventBus
from stinger_fx.data import in_memory_store
from stinger_fx.data.parquet_store import ParquetStore
from stinger_fx.domain import Tick, Timeframe
from stinger_fx.ui.handle import EngineHandle
from stinger_fx.ui.web.live_backtest import LiveBacktestController, LiveBacktestError


@pytest.fixture
def tick_root(tmp_path: Path) -> Path:
    """Tiny tick parquet so the controller test can spin a real backtest
    end-to-end inside a few hundred ms."""
    root = tmp_path / "parquet"
    base = datetime(2024, 1, 1, tzinfo=UTC)
    bids = [1.1000 + 0.0001 * i for i in range(60)]
    store = ParquetStore(root)
    store.append_ticks(
        "EURUSD",
        [
            Tick(symbol="EURUSD", time=base + timedelta(seconds=i),
                 bid=b, ask=b + 2e-5)
            for i, b in enumerate(bids)
        ],
    )
    return root


def _strategy_entry() -> StrategyEntry:
    return StrategyEntry(
        id="ma_tick",
        class_path="stinger_fx.strategies.examples.ma_crossover:MACrossover",
        enabled=True,
        params={
            "symbol": "EURUSD", "timeframe": "M1",
            "fast": 2, "slow": 5, "volume": 0.1,
        },
    )


def _run_cfg(tick_root: Path, run_id: str = "ut") -> BacktestRunConfig:
    return BacktestRunConfig(
        id=run_id,
        mode="file",
        strategy_id="ma_tick",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=2),
        initial_balance=10_000.0,
        granularity="tick",
        data_source=tick_root,
    )


def _full_config(tick_root: Path, *, run_id: str = "ut") -> FullConfig:
    return FullConfig(
        app=AppConfig(broker=BrokerConfig(type="mt5")),
        strategies=StrategiesConfig(strategies=[_strategy_entry()]),
        backtest=BacktestConfig(runs=[_run_cfg(tick_root, run_id=run_id)]),
    )


def _handle() -> EngineHandle:
    return EngineHandle(bus=AsyncEventBus(), brokers=BrokerPool())


@pytest.mark.asyncio
async def test_start_spawns_task_and_flips_running(tick_root: Path) -> None:
    handle = _handle()
    controller = LiveBacktestController(
        handle=handle,
        cfg=_full_config(tick_root),
        sqlite_store=in_memory_store(),
    )
    assert not controller.is_running()
    # speed=100 → 2-minute sim window finishes in ~1.2s wall.
    await controller.start("ut", speed=100.0)
    assert controller.is_running()
    status = controller.status
    assert status["running"] is True
    assert status["run_id"] == "ut"
    assert status["speed"] == 100.0
    # Cleanup — let the task finish on its own to also test natural completion.
    await controller.stop()
    assert not controller.is_running()
    await handle.bus.close()


@pytest.mark.asyncio
async def test_stop_cancels_running_task(tick_root: Path) -> None:
    """Stop while a long replay is in-flight — cancellation must be clean."""
    handle = _handle()
    controller = LiveBacktestController(
        handle=handle,
        cfg=_full_config(tick_root),
        sqlite_store=in_memory_store(),
    )
    # Slow replay so the task is definitely mid-run when we cancel.
    await controller.start("ut", speed=1.0)
    assert controller.is_running()
    await controller.stop()
    assert not controller.is_running()
    # And we can start again afterward.
    await controller.start("ut", speed=100.0)
    assert controller.is_running()
    await controller.stop()
    await handle.bus.close()


@pytest.mark.asyncio
async def test_double_start_raises(tick_root: Path) -> None:
    handle = _handle()
    controller = LiveBacktestController(
        handle=handle,
        cfg=_full_config(tick_root),
        sqlite_store=in_memory_store(),
    )
    await controller.start("ut", speed=1.0)
    with pytest.raises(LiveBacktestError, match="already running"):
        await controller.start("ut", speed=10.0)
    await controller.stop()
    await handle.bus.close()


@pytest.mark.asyncio
async def test_unknown_run_id_raises(tick_root: Path) -> None:
    handle = _handle()
    controller = LiveBacktestController(
        handle=handle,
        cfg=_full_config(tick_root),
        sqlite_store=in_memory_store(),
    )
    with pytest.raises(LiveBacktestError, match="no backtest run"):
        await controller.start("nope-doesnt-exist", speed=1.0)
    assert not controller.is_running()
    await handle.bus.close()


@pytest.mark.asyncio
async def test_speed_zero_rejected(tick_root: Path) -> None:
    """speed=0 (max) would flood the SSE queue — live mode requires throttle."""
    handle = _handle()
    controller = LiveBacktestController(
        handle=handle,
        cfg=_full_config(tick_root),
        sqlite_store=in_memory_store(),
    )
    with pytest.raises(LiveBacktestError, match="speed > 0"):
        await controller.start("ut", speed=0.0)
    await handle.bus.close()


@pytest.mark.asyncio
async def test_stop_is_noop_when_idle(tick_root: Path) -> None:
    """Calling stop() before start() must not raise — endpoint may invoke
    it defensively on shutdown."""
    handle = _handle()
    controller = LiveBacktestController(
        handle=handle,
        cfg=_full_config(tick_root),
        sqlite_store=in_memory_store(),
    )
    await controller.stop()
    await controller.stop()
    assert not controller.is_running()
    await handle.bus.close()
