"""Pydantic v2 schemas for the YAML config files.

Three top-level shapes, each in its own YAML file:
  • AppConfig          ← config/app.yaml
  • StrategiesConfig   ← config/strategies.yaml
  • BacktestConfig     ← config/backtest.yaml (a list of runs)

Validation errors are raised as `ConfigError` by the loader so callers don't
need to know about pydantic's exception types.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from stinger_fx.domain.timeframes import Timeframe

# --- App ----------------------------------------------------------------------


class MT5Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    terminal_path: str = ""        # path to terminal64.exe; "" = auto-detect
    login: int = 0                 # 0 = use currently-logged-in account
    password: str = ""
    server: str = ""
    timeout_ms: int = Field(60_000, gt=0)


class MT4Config(BaseModel):
    """Phase-2 placeholder; presence is validated but settings are unused for now."""

    model_config = ConfigDict(extra="forbid")

    bridge_host: str = "127.0.0.1"
    bridge_port: int = Field(15555, gt=0, lt=65536)


class BrokerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["mt5", "mt4"] = "mt5"
    mt5: MT5Config | None = None
    mt4: MT4Config | None = None


class WebConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(8765, gt=0, lt=65536)


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_open_positions_per_strategy: int = Field(5, ge=0)
    max_daily_loss_pct: float = Field(5.0, ge=0)
    kill_switch_drawdown_pct: float = Field(20.0, ge=0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["normal", "tui", "web"] = "normal"
    broker: BrokerConfig
    data_dir: Path = Path("./data")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    timezone: str = "UTC"
    web: WebConfig = WebConfig()
    risk: RiskConfig = RiskConfig()


# --- Strategies ---------------------------------------------------------------


class StrategyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    class_path: str = Field(min_length=1)
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("class_path")
    @classmethod
    def _has_colon(cls, v: str) -> str:
        if ":" not in v:
            raise ValueError("class_path must be of the form 'pkg.module:ClassName'")
        return v


class StrategiesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategies: list[StrategyEntry] = Field(default_factory=list)

    @field_validator("strategies")
    @classmethod
    def _unique_ids(cls, v: list[StrategyEntry]) -> list[StrategyEntry]:
        ids = [s.id for s in v]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate strategy ids: {dupes}")
        return v


# --- Backtest -----------------------------------------------------------------


class BacktestRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    mode: Literal["file", "mt5_tester", "mt4_tester"] = "file"
    strategy_id: str
    symbol: str
    timeframe: Timeframe
    start: datetime
    end: datetime
    initial_balance: float = Field(10_000.0, gt=0)
    data_source: Path | None = None
    slippage_pips: float = Field(0.0, ge=0)

    @field_validator("start", "end")
    @classmethod
    def _require_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("start/end must include a timezone (e.g. ...T00:00:00Z)")
        return v


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs: list[BacktestRunConfig] = Field(default_factory=list)

    @field_validator("runs")
    @classmethod
    def _unique_ids(cls, v: list[BacktestRunConfig]) -> list[BacktestRunConfig]:
        ids = [r.id for r in v]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate backtest run ids: {dupes}")
        return v


# --- Snapshot of full config (used by the reloader) --------------------------


class FullConfig(BaseModel):
    """Combined snapshot used by the hot-reloader to diff old vs. new."""

    model_config = ConfigDict(extra="forbid")

    app: AppConfig
    strategies: StrategiesConfig
    backtest: BacktestConfig
