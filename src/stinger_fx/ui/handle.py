"""EngineHandle — the read+control facade UIs use to talk to the engine.

All three UI modes (normal/TUI/web) share this so most of their code stays
agnostic of how the engine is wired internally.

In a multi-account setup the handle holds a `BrokerPool` and exposes the
primary broker via `.broker` for UI components that show a single account.
Components that need per-account data can iterate `handle.brokers.items()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stinger_fx.brokers.base import BaseBroker
from stinger_fx.brokers.pool import BrokerPool
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
    account: str = "default"


@dataclass
class EngineHandle:
    """Slim read+control view of the engine, injected into each UI runner."""

    bus: AsyncEventBus
    brokers: BrokerPool
    runners: dict[str, StrategyRunner] = field(default_factory=dict)
    # `strategy_accounts` maps strategy_id → account_id; used by list_strategies
    # to label which broker each strategy trades against.
    strategy_accounts: dict[str, str] = field(default_factory=dict)

    @property
    def broker(self) -> BaseBroker:
        """Backward-compat alias — returns the primary broker."""
        return self.brokers.primary()

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
                    account=self.strategy_accounts.get(sid, "default"),
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
        """Aggregate positions across every broker in the pool."""
        out: list[Position] = []
        for _, broker in self.brokers.items():
            out.extend(await broker.get_positions())
        return out

    async def get_account(self) -> AccountInfo:
        """Primary account — used by UIs that show a single account."""
        return await self.brokers.primary().get_account_info()

    async def list_accounts(self) -> list[tuple[str, AccountInfo]]:
        """One (account_id, info) per configured broker."""
        out: list[tuple[str, AccountInfo]] = []
        for account_id, broker in self.brokers.items():
            try:
                info = await broker.get_account_info()
            except Exception:  # noqa: BLE001
                continue
            out.append((account_id, info))
        return out
