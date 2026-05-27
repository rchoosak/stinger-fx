"""Hot-reload diff & apply.

Walks the diff between the previous and new `FullConfig` snapshots and decides,
per category, whether the change can be applied live or requires a restart.

The reload runs under `engine.reload_lock`, which strategy runners also hold
briefly while dispatching a single event — guaranteeing that params can't be
swapped mid-handler.

Apply callbacks are injected by the engine assembler so that this module stays
free of broker / strategy-runner imports.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from stinger_fx.config.models import (
    AppConfig,
    FullConfig,
    StrategyEntry,
)

logger = logging.getLogger("stinger.config.reload")


@dataclass
class ReloadActions:
    """Apply hooks owned by the engine. All async."""

    # Strategies
    add_strategy: Callable[[StrategyEntry], Awaitable[None]]
    remove_strategy: Callable[[str], Awaitable[None]]                     # id
    replace_strategy: Callable[[StrategyEntry], Awaitable[None]]          # class_path change → stop+start
    update_params: Callable[[str, dict[str, Any]], Awaitable[None]]       # id, new params
    set_enabled: Callable[[str, bool], Awaitable[None]]                   # id, enabled

    # App
    update_log_level: Callable[[str], Awaitable[None]]
    update_risk: Callable[[Any], Awaitable[None]]                         # RiskConfig


@dataclass
class ReloadResult:
    applied: list[str] = field(default_factory=list)
    needs_restart: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class ConfigReloader:
    def __init__(self, actions: ReloadActions) -> None:
        self._actions = actions

    async def diff_and_apply(self, old: FullConfig, new: FullConfig) -> ReloadResult:
        result = ReloadResult()

        await self._diff_app(old.app, new.app, result)
        await self._diff_strategies(old, new, result)
        # Backtest config is read on-demand by `stinger-fx backtest run`; no hot apply needed.

        if result.applied:
            logger.info("config reload applied=%s", result.applied)
        if result.needs_restart:
            logger.warning("config reload needs restart for=%s", result.needs_restart)
        return result

    # -- App ------------------------------------------------------------------

    async def _diff_app(self, old: AppConfig, new: AppConfig, result: ReloadResult) -> None:
        # Broker switch — requires restart (broker re-connect mid-run is unsafe).
        # `primary_broker_config` is safe for both single- and multi-account
        # configs; for multi-account we compare the whole `broker_list` below
        # to also catch additions / removals.
        old_primary = old.primary_broker_config
        new_primary = new.primary_broker_config
        if old_primary.type != new_primary.type:
            result.needs_restart.append(
                f"broker.type ({old_primary.type}→{new_primary.type})"
            )
            return

        # Broker subconfig changes within the same type — also restart for safety.
        if old.broker_list != new.broker_list:
            result.needs_restart.append("broker.*")

        # Web bind host/port — restart-required; everything else in WebConfig is hot.
        if (old.web.host, old.web.port) != (new.web.host, new.web.port):
            result.needs_restart.append("web.host/port")

        # Log level — hot.
        if old.log_level != new.log_level:
            await self._safe(
                self._actions.update_log_level(new.log_level),
                f"log_level={new.log_level}",
                result,
            )

        # Risk — hot.
        if old.risk != new.risk:
            await self._safe(self._actions.update_risk(new.risk), "risk.*", result)

        if old.mode != new.mode:
            result.needs_restart.append(f"mode ({old.mode}→{new.mode})")

        if old.timezone != new.timezone:
            result.needs_restart.append(f"timezone ({old.timezone}→{new.timezone})")

        if old.data_dir != new.data_dir:
            result.needs_restart.append(f"data_dir ({old.data_dir}→{new.data_dir})")

    # -- Strategies -----------------------------------------------------------

    async def _diff_strategies(
        self,
        old: FullConfig,
        new: FullConfig,
        result: ReloadResult,
    ) -> None:
        old_by_id = {s.id: s for s in old.strategies.strategies}
        new_by_id = {s.id: s for s in new.strategies.strategies}

        # Removed entries
        for sid in old_by_id.keys() - new_by_id.keys():
            await self._safe(self._actions.remove_strategy(sid), f"strategy[-{sid}]", result)

        # Added entries
        for sid in new_by_id.keys() - old_by_id.keys():
            entry = new_by_id[sid]
            if not entry.enabled:
                continue  # disabled-on-arrival = nothing to start
            await self._safe(self._actions.add_strategy(entry), f"strategy[+{sid}]", result)

        # Modified entries
        for sid in old_by_id.keys() & new_by_id.keys():
            o = old_by_id[sid]
            n = new_by_id[sid]
            if o == n:
                continue

            if o.class_path != n.class_path:
                await self._safe(self._actions.replace_strategy(n), f"strategy[~{sid}].class_path", result)
                continue

            if o.enabled != n.enabled:
                await self._safe(
                    self._actions.set_enabled(sid, n.enabled),
                    f"strategy[{sid}].enabled={n.enabled}",
                    result,
                )

            if o.params != n.params and n.enabled:
                await self._safe(
                    self._actions.update_params(sid, n.params),
                    f"strategy[{sid}].params",
                    result,
                )

    # -- helpers --------------------------------------------------------------

    async def _safe(self, coro: Awaitable[None], label: str, result: ReloadResult) -> None:
        try:
            await coro
            result.applied.append(label)
        except Exception as e:
            logger.exception("reload apply failed label=%s", label)
            result.errors.append(f"{label}: {e}")
