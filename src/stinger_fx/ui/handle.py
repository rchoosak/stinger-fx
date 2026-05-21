"""EngineHandle — the read+control facade UIs use to talk to the engine.

All three UI modes (normal/TUI/web) share this so most of their code stays
agnostic of how the engine is wired internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stinger_fx.brokers.base import BaseBroker
from stinger_fx.core.event_bus import AsyncEventBus
from stinger_fx.domain import AccountInfo, Position
from stinger_fx.strategies.runner import StrategyRunner


@dataclass
class StrategyState:
    id: str
    name: str
    state: str          # started | stopped | paused | quarantined
    symbol: str
    timeframe: str


@dataclass
class EngineHandle:
    """Slim read+control view of the engine, injected into each UI runner."""

    bus: AsyncEventBus
    broker: BaseBroker
    runners: dict[str, StrategyRunner] = field(default_factory=dict)

    async def list_strategies(self) -> list[StrategyState]:
        out: list[StrategyState] = []
        for sid, runner in self.runners.items():
            ctx = runner._ctx
            state = "started"
            if runner._stopped:
                state = "stopped"
            elif runner._quarantined:
                state = "quarantined"
            elif runner._paused:
                state = "paused"
            out.append(
                StrategyState(
                    id=sid,
                    name=runner.strategy.name or sid,
                    state=state,
                    symbol=ctx.symbol if ctx else "",
                    timeframe=ctx.timeframe.value if ctx else "",
                )
            )
        return out

    async def pause_strategy(self, sid: str) -> None:
        runner = self.runners.get(sid)
        if runner is None:
            raise KeyError(sid)
        await runner.pause()

    async def resume_strategy(self, sid: str) -> None:
        runner = self.runners.get(sid)
        if runner is None:
            raise KeyError(sid)
        await runner.resume()

    async def get_positions(self) -> list[Position]:
        return await self.broker.get_positions()

    async def get_account(self) -> AccountInfo:
        return await self.broker.get_account_info()
