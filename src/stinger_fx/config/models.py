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


MetricName = Literal[
    "net_pnl",
    "sharpe",
    "profit_factor",
    "win_rate",
    "expectancy",
    "max_drawdown",      # smaller is better; ranking flips automatically
    "trades",
]


class SweepRunConfig(BaseModel):
    """A parameter-sweep run — same as BacktestRunConfig plus a grid of values
    to try. Each cell of the cartesian product is backtested in turn."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    strategy_id: str
    symbol: str
    timeframe: Timeframe
    start: datetime
    end: datetime
    initial_balance: float = Field(10_000.0, gt=0)
    slippage_pips: float = Field(0.0, ge=0)
    data_source: Path
    parameter_grid: dict[str, list[Any]] = Field(default_factory=dict)
    rank_by: MetricName = "net_pnl"
    top_n: int = Field(10, ge=1, le=1000)

    @field_validator("start", "end")
    @classmethod
    def _require_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("start/end must include a timezone (e.g. ...T00:00:00Z)")
        return v

    @field_validator("parameter_grid")
    @classmethod
    def _grid_nonempty(cls, v: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if not v:
            raise ValueError("parameter_grid must declare at least one parameter")
        for name, values in v.items():
            if not values:
                raise ValueError(f"parameter_grid['{name}'] is empty — list one value at minimum")
        return v


class WalkForwardConfig(BaseModel):
    """Walk-forward optimisation: time-sliced sweeps with out-of-sample
    evaluation. References a sweep config for the parameter grid + backtest
    settings; this config layers the time slicing on top.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    sweep_id: str           # the SweepRunConfig that supplies the parameter grid
    folds: int = Field(4, ge=2, le=50)
    scheme: Literal["expanding", "rolling"] = "expanding"
    in_sample_ratio: float = Field(0.7, gt=0.0, lt=1.0)  # only used for rolling
    rank_by: MetricName | None = None     # default: inherit from sweep


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs: list[BacktestRunConfig] = Field(default_factory=list)
    sweeps: list[SweepRunConfig] = Field(default_factory=list)
    walk_forwards: list[WalkForwardConfig] = Field(default_factory=list)

    @field_validator("runs")
    @classmethod
    def _unique_ids(cls, v: list[BacktestRunConfig]) -> list[BacktestRunConfig]:
        ids = [r.id for r in v]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate backtest run ids: {dupes}")
        return v

    @field_validator("sweeps")
    @classmethod
    def _unique_sweep_ids(cls, v: list[SweepRunConfig]) -> list[SweepRunConfig]:
        ids = [r.id for r in v]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate sweep ids: {dupes}")
        return v

    @field_validator("walk_forwards")
    @classmethod
    def _unique_wf_ids(cls, v: list[WalkForwardConfig]) -> list[WalkForwardConfig]:
        ids = [r.id for r in v]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate walk-forward ids: {dupes}")
        return v


# --- Snapshot of full config (used by the reloader) --------------------------


class FullConfig(BaseModel):
    """Combined snapshot used by the hot-reloader to diff old vs. new."""

    model_config = ConfigDict(extra="forbid")

    app: AppConfig
    strategies: StrategiesConfig
    backtest: BacktestConfig
