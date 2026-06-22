"""StrategyCircuitBreaker — auto-pause a degrading strategy.

Turns the drift monitor's alert (and a consecutive-loss streak) into an action:
it calls ``StrategyRunner.pause()``, which the runner enforces via ``_active()``
so the strategy stops trading until an operator resumes it. Alert-only behaviour
is unchanged — the ``StrategyDriftEvent`` still fires; this just adds the pause.

Best-effort: a pause failure is logged and never breaks the engine. Idempotent —
an already-paused strategy isn't paused again.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from stinger_fx.config.models import CircuitBreakerConfig
from stinger_fx.core.event_bus import AsyncEventBus, Subscription
from stinger_fx.core.events import PositionClosedEvent, StrategyDriftEvent

logger = logging.getLogger("stinger.observability.circuit_breaker")


class StrategyCircuitBreaker:
    def __init__(
        self,
        bus: AsyncEventBus,
        *,
        strategy_for_magic: Callable[[int], str | None],
        pause_strategy: Callable[[str], Awaitable[None]],
        cfg: CircuitBreakerConfig,
    ) -> None:
        self._bus = bus
        self._strategy_for_magic = strategy_for_magic
        self._pause_strategy = pause_strategy
        self._cfg = cfg
        self._subs: list[Subscription] = []
        self._losing_streak: dict[str, int] = {}
        self._paused: set[str] = set()

    async def start(self) -> None:
        if self._cfg.pause_on_drift:
            self._subs.append(
                self._bus.subscribe(
                    StrategyDriftEvent, self._on_drift, name="circuit_breaker.drift"
                )
            )
        if self._cfg.max_consecutive_losses > 0:
            self._subs.append(
                self._bus.subscribe(
                    PositionClosedEvent, self._on_closed, name="circuit_breaker.close"
                )
            )
        logger.info("circuit_breaker_started")

    async def stop(self) -> None:
        for sub in self._subs:
            await sub.unsubscribe()
        self._subs.clear()

    async def _on_drift(self, evt: StrategyDriftEvent) -> None:
        if not self._cfg.pause_on_drift:
            return
        await self._pause(evt.strategy_id, f"drift: {evt.reason}")

    async def _on_closed(self, evt: PositionClosedEvent) -> None:
        if self._cfg.max_consecutive_losses <= 0:
            return
        sid = self._strategy_for_magic(evt.position.magic)
        if sid is None:
            return
        if evt.realized_pnl < 0:
            self._losing_streak[sid] = self._losing_streak.get(sid, 0) + 1
        else:
            self._losing_streak[sid] = 0
        streak = self._losing_streak.get(sid, 0)
        if streak >= self._cfg.max_consecutive_losses:
            await self._pause(sid, f"{streak} consecutive losses")

    async def _pause(self, strategy_id: str, reason: str) -> None:
        if strategy_id in self._paused:
            return
        self._paused.add(strategy_id)
        logger.warning("circuit_breaker_pause strategy=%s reason=%s", strategy_id, reason)
        try:
            await self._pause_strategy(strategy_id)
        except Exception:
            logger.exception("circuit_breaker_pause_failed strategy=%s", strategy_id)
