"""StrategyRunner — quarantine + hot-reload of params."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from stinger_fx.core import AsyncEventBus, LiveClock
from stinger_fx.core.events import BarEvent
from stinger_fx.domain import Bar, Subscription, Timeframe
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.parameters import StrategyParams
from stinger_fx.strategies.runner import (
    MAX_ERRORS_PER_MINUTE,
    StrategyRunner,
    derive_magic,
)


class _Params(StrategyParams):
    x: int = 1


class _RaisingStrategy(BaseStrategy):
    name = "raiser"
    Params = _Params

    @classmethod
    def subscriptions(cls, params: StrategyParams) -> list[Subscription]:
        return [Subscription(symbol="EURUSD", timeframe=Timeframe.M15)]

    async def on_bar(self, ctx, bar):
        raise RuntimeError("boom")


class _ReloadAware(BaseStrategy):
    name = "reload_aware"
    Params = _Params
    seen_reloads: list[tuple[int, int]] = []

    @classmethod
    def subscriptions(cls, params: StrategyParams) -> list[Subscription]:
        return [Subscription(symbol="EURUSD", timeframe=Timeframe.M15)]

    async def on_params_reloaded(self, ctx, old, new):
        type(self).seen_reloads.append((old.x, new.x))


def _bar(ts: datetime) -> Bar:
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.M15,
        time=ts,
        open=1.10,
        high=1.11,
        low=1.09,
        close=1.105,
        is_closed=True,
    )


@pytest.mark.asyncio
async def test_quarantine_after_too_many_errors() -> None:
    bus = AsyncEventBus()
    runner = StrategyRunner(
        strategy_id="raiser_1",
        strategy=_RaisingStrategy(),
        params=_Params(),
        bus=bus,
        clock=LiveClock(),
        reload_lock=asyncio.Lock(),
        signal_sink=lambda _s: asyncio.sleep(0),  # type: ignore[arg-type, return-value]
    )
    await runner.start()
    base = datetime(2024, 1, 1, tzinfo=UTC)
    for i in range(MAX_ERRORS_PER_MINUTE + 2):
        await bus.publish(BarEvent(bar=_bar(base.replace(minute=15 * i % 60))))
        await asyncio.sleep(0.02)
    assert runner._quarantined is True
    await runner.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_update_params_fires_on_params_reloaded() -> None:
    _ReloadAware.seen_reloads = []
    bus = AsyncEventBus()
    runner = StrategyRunner(
        strategy_id="reload_1",
        strategy=_ReloadAware(),
        params=_Params(x=1),
        bus=bus,
        clock=LiveClock(),
        reload_lock=asyncio.Lock(),
        signal_sink=lambda _s: asyncio.sleep(0),  # type: ignore[arg-type, return-value]
    )
    await runner.start()
    await runner.update_params(_Params(x=42))
    await asyncio.sleep(0.05)
    assert _ReloadAware.seen_reloads == [(1, 42)]
    await runner.stop()
    await bus.close()


def test_derive_magic_is_deterministic_and_positive() -> None:
    a = derive_magic("ma_eurusd_m15")
    b = derive_magic("ma_eurusd_m15")
    c = derive_magic("ma_gbpusd_m15")
    assert a == b
    assert a != c
    assert a > 0
    assert a < (1 << 63)
