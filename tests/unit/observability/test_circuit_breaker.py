"""StrategyCircuitBreaker — auto-pause on drift / losing streak."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from stinger_fx.config.models import CircuitBreakerConfig
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import PositionClosedEvent, StrategyDriftEvent
from stinger_fx.domain import Position, Side
from stinger_fx.observability.circuit_breaker import StrategyCircuitBreaker


def _drift(sid: str = "s1") -> StrategyDriftEvent:
    return StrategyDriftEvent(
        strategy_id=sid, sample_size=20, live_win_rate=0.2, live_expectancy_per_lot=-5.0,
        baseline_win_rate=0.5, baseline_expectancy_per_lot=10.0, reason="win-rate low",
    )


def _close(pnl: float, magic: int = 7) -> PositionClosedEvent:
    return PositionClosedEvent(
        position=Position(
            ticket=1, symbol="XAUUSD", side=Side.BUY, volume=0.1,
            open_price=2000.0, open_time=datetime(2024, 1, 1, tzinfo=UTC), magic=magic,
        ),
        realized_pnl=pnl,
    )


def _breaker(**over) -> tuple[StrategyCircuitBreaker, list[str]]:
    paused: list[str] = []

    async def pause_strategy(sid: str) -> None:
        paused.append(sid)

    cfg = CircuitBreakerConfig(enabled=True, **over)
    cb = StrategyCircuitBreaker(
        AsyncEventBus(), strategy_for_magic=lambda m: "s1",
        pause_strategy=pause_strategy, cfg=cfg,
    )
    return cb, paused


@pytest.mark.asyncio
async def test_drift_pauses_strategy() -> None:
    cb, paused = _breaker(pause_on_drift=True)
    await cb._on_drift(_drift())
    assert paused == ["s1"]


@pytest.mark.asyncio
async def test_drift_no_pause_when_disabled() -> None:
    cb, paused = _breaker(pause_on_drift=False)
    await cb._on_drift(_drift())
    assert paused == []


@pytest.mark.asyncio
async def test_consecutive_losses_pause() -> None:
    cb, paused = _breaker(pause_on_drift=False, max_consecutive_losses=3)
    await cb._on_closed(_close(-1.0))
    await cb._on_closed(_close(-1.0))
    assert paused == []          # 2 < 3
    await cb._on_closed(_close(-1.0))
    assert paused == ["s1"]       # 3rd loss → pause


@pytest.mark.asyncio
async def test_a_win_resets_the_streak() -> None:
    cb, paused = _breaker(pause_on_drift=False, max_consecutive_losses=3)
    await cb._on_closed(_close(-1.0))
    await cb._on_closed(_close(-1.0))
    await cb._on_closed(_close(5.0))   # win resets
    await cb._on_closed(_close(-1.0))
    await cb._on_closed(_close(-1.0))
    assert paused == []                 # only 2 in a row after the win


@pytest.mark.asyncio
async def test_already_paused_not_repaused() -> None:
    cb, paused = _breaker(pause_on_drift=True, max_consecutive_losses=2)
    await cb._on_drift(_drift())        # pauses
    await cb._on_closed(_close(-1.0))
    await cb._on_closed(_close(-1.0))   # would pause again, but already paused
    assert paused == ["s1"]             # only once


@pytest.mark.asyncio
async def test_unknown_magic_ignored() -> None:
    paused: list[str] = []

    async def pause_strategy(sid: str) -> None:
        paused.append(sid)

    cb = StrategyCircuitBreaker(
        AsyncEventBus(), strategy_for_magic=lambda m: None,  # unknown
        pause_strategy=pause_strategy,
        cfg=CircuitBreakerConfig(enabled=True, max_consecutive_losses=1),
    )
    await cb._on_closed(_close(-1.0))
    assert paused == []
