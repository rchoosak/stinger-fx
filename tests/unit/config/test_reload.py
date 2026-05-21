"""Hot-reload diff_and_apply — verify the categorization logic."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from stinger_fx.config import ConfigReloader, ReloadActions
from stinger_fx.config.models import (
    AppConfig,
    BacktestConfig,
    BacktestRunConfig,
    BrokerConfig,
    FullConfig,
    MT5Config,
    RiskConfig,
    StrategiesConfig,
    StrategyEntry,
    WebConfig,
)
from stinger_fx.domain import Timeframe


def _full(strategies: list[StrategyEntry], **app_overrides: Any) -> FullConfig:
    return FullConfig(
        app=AppConfig(
            broker=BrokerConfig(type="mt5", mt5=MT5Config()),
            web=WebConfig(),
            risk=RiskConfig(),
            **app_overrides,
        ),
        strategies=StrategiesConfig(strategies=strategies),
        backtest=BacktestConfig(
            runs=[
                BacktestRunConfig(
                    id="r1",
                    mode="file",
                    strategy_id="s1",
                    symbol="EURUSD",
                    timeframe=Timeframe.M15,
                    start=datetime(2024, 1, 1, tzinfo=UTC),
                    end=datetime(2024, 2, 1, tzinfo=UTC),
                )
            ]
        ),
    )


class _Recorder:
    def __init__(self) -> None:
        self.added: list[str] = []
        self.removed: list[str] = []
        self.replaced: list[str] = []
        self.params: list[tuple[str, dict]] = []
        self.enabled: list[tuple[str, bool]] = []
        self.log_levels: list[str] = []
        self.risk_calls: int = 0

    def as_actions(self) -> ReloadActions:
        async def add(e: StrategyEntry) -> None:
            self.added.append(e.id)

        async def remove(sid: str) -> None:
            self.removed.append(sid)

        async def replace(e: StrategyEntry) -> None:
            self.replaced.append(e.id)

        async def params(sid: str, raw: dict) -> None:
            self.params.append((sid, raw))

        async def enabled(sid: str, val: bool) -> None:
            self.enabled.append((sid, val))

        async def log_level(level: str) -> None:
            self.log_levels.append(level)

        async def risk(_: Any) -> None:
            self.risk_calls += 1

        return ReloadActions(
            add_strategy=add,
            remove_strategy=remove,
            replace_strategy=replace,
            update_params=params,
            set_enabled=enabled,
            update_log_level=log_level,
            update_risk=risk,
        )


@pytest.mark.asyncio
async def test_param_change_calls_update_params() -> None:
    s1_old = StrategyEntry(id="s1", class_path="a:B", enabled=True, params={"x": 1})
    s1_new = StrategyEntry(id="s1", class_path="a:B", enabled=True, params={"x": 2})
    rec = _Recorder()
    reloader = ConfigReloader(rec.as_actions())
    result = await reloader.diff_and_apply(_full([s1_old]), _full([s1_new]))
    assert result.ok
    assert rec.params == [("s1", {"x": 2})]
    assert not rec.added and not rec.removed


@pytest.mark.asyncio
async def test_added_strategy_triggers_add() -> None:
    s_new = StrategyEntry(id="s_new", class_path="a:B", enabled=True, params={})
    rec = _Recorder()
    reloader = ConfigReloader(rec.as_actions())
    result = await reloader.diff_and_apply(_full([]), _full([s_new]))
    assert result.ok
    assert rec.added == ["s_new"]


@pytest.mark.asyncio
async def test_removed_strategy_triggers_remove() -> None:
    s_old = StrategyEntry(id="s_old", class_path="a:B", enabled=True, params={})
    rec = _Recorder()
    reloader = ConfigReloader(rec.as_actions())
    result = await reloader.diff_and_apply(_full([s_old]), _full([]))
    assert result.ok
    assert rec.removed == ["s_old"]


@pytest.mark.asyncio
async def test_class_path_change_triggers_replace() -> None:
    s_old = StrategyEntry(id="s1", class_path="a:B", enabled=True, params={})
    s_new = StrategyEntry(id="s1", class_path="a:C", enabled=True, params={})
    rec = _Recorder()
    reloader = ConfigReloader(rec.as_actions())
    result = await reloader.diff_and_apply(_full([s_old]), _full([s_new]))
    assert result.ok
    assert rec.replaced == ["s1"]


@pytest.mark.asyncio
async def test_broker_change_marks_needs_restart() -> None:
    rec = _Recorder()
    reloader = ConfigReloader(rec.as_actions())
    old = _full([])
    new = _full([])
    new.app.broker.mt5 = MT5Config(login=9999)  # change broker subconfig
    result = await reloader.diff_and_apply(old, new)
    assert result.ok
    assert any("broker" in r for r in result.needs_restart)


@pytest.mark.asyncio
async def test_log_level_change_is_hot() -> None:
    rec = _Recorder()
    reloader = ConfigReloader(rec.as_actions())
    new = _full([], log_level="DEBUG")
    result = await reloader.diff_and_apply(_full([]), new)
    assert result.ok
    assert rec.log_levels == ["DEBUG"]
