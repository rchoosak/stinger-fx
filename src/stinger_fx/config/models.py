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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from stinger_fx.domain.symbols import Subscription
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

    # `id` identifies the broker in a multi-account setup; strategies pick which
    # broker to trade against via StrategyEntry.account. The legacy singular
    # `broker:` block omits this field and gets id="default" assigned by the
    # AppConfig validator.
    id: str = Field("default", min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    type: Literal["mt5", "mt4"] = "mt5"
    mt5: MT5Config | None = None
    mt4: MT4Config | None = None


class WebConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(8765, gt=0, lt=65536)


class SymbolRiskConfig(BaseModel):
    """Per-symbol risk overrides (nested inside `RiskConfig.per_symbol`)."""

    model_config = ConfigDict(extra="forbid")

    max_open_positions: int = Field(0, ge=0)
    """Maximum concurrent positions for this symbol. 0 = unlimited."""

    max_daily_loss_usd: float = Field(0.0, ge=0)
    """Maximum realized loss in USD for this symbol per UTC day. 0 = unlimited."""


class PositionSizingConfig(BaseModel):
    """Risk-based position sizing (account-level, applies to every strategy)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    """When off (default), orders use the strategy's fixed `volume`."""

    risk_per_trade_pct: float = Field(1.0, gt=0, le=100)
    """Equity fraction risked per trade at its stop, e.g. 1.0 = 1%."""


class NewsBlackout(BaseModel):
    """A UTC window during which no new orders may be placed."""

    model_config = ConfigDict(extra="forbid")

    start: datetime
    end: datetime

    @model_validator(mode="after")
    def _check_order(self) -> NewsBlackout:
        if self.end <= self.start:
            raise ValueError("news blackout end must be after start")
        return self


class TradingFilterConfig(BaseModel):
    """Account-level pre-trade guard against bad-fill conditions every strategy
    faces. All checks are independently toggleable; all off by default."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_spread_points: int = Field(0, ge=0)
    """Block when the broker spread exceeds this (points). 0 = no spread cap."""

    session_start_hour_utc: int | None = Field(default=None, ge=0, le=23)
    session_end_hour_utc: int | None = Field(default=None, ge=1, le=24)
    """Trade only within [start, end) UTC hours. Both None = no session gate.
    A start > end wraps past midnight (e.g. 22→6)."""

    block_rollover: bool = False
    rollover_hour_utc: int = Field(21, ge=0, le=23)
    rollover_block_minutes: int = Field(5, ge=0)
    """Block within ± this many minutes of `rollover_hour:00` UTC."""

    news_blackouts: list[NewsBlackout] = Field(default_factory=list)

    @model_validator(mode="after")
    def _session_pair(self) -> TradingFilterConfig:
        a, b = self.session_start_hour_utc, self.session_end_hour_utc
        if (a is None) != (b is None):
            raise ValueError(
                "session_start_hour_utc and session_end_hour_utc must be set "
                "together (or both omitted)"
            )
        return self


class DriftMonitorConfig(BaseModel):
    """Live-vs-backtest drift monitor — alert when a strategy's recent live
    win-rate/expectancy falls materially below its backtest baseline."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    window: int = Field(50, ge=1)
    """Most recent live trades per strategy used to compute live stats."""

    min_trades: int = Field(20, ge=1)
    """Need at least this many recent live trades before alerting."""

    min_win_rate_ratio: float = Field(0.7, gt=0, le=1)
    """Alert when live win-rate < baseline win-rate x this."""

    min_expectancy_ratio: float = Field(0.5, gt=0, le=1)
    """Alert when live expectancy < baseline expectancy x this."""


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_open_positions_per_strategy: int = Field(5, ge=0)
    max_daily_loss_pct: float = Field(5.0, ge=0)
    kill_switch_drawdown_pct: float = Field(20.0, ge=0)

    per_symbol: dict[str, SymbolRiskConfig] = Field(default_factory=dict)
    """Per-symbol overrides.  Keys are symbol names (e.g. ``"EURUSD"``)."""

    position_sizing: PositionSizingConfig = Field(default_factory=PositionSizingConfig)
    """Account-level risk-based sizing; off by default → fixed lots."""

    trading_filter: TradingFilterConfig = Field(default_factory=TradingFilterConfig)
    """Account-level pre-trade spread/session/rollover/news guard; off by default."""

    drift_monitor: DriftMonitorConfig = Field(default_factory=DriftMonitorConfig)
    """Live-vs-backtest drift alerting; off by default."""


class MetricsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = Field(9100, gt=0, lt=65536)


class NotificationChannelConfig(BaseModel):
    """One Telegram / Discord notification target."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["telegram", "discord"]
    enabled: bool = True
    # Telegram
    bot_token: str = ""
    chat_id: int | str = ""
    # Discord
    webhook_url: str = ""
    # Event filters — see notifications.known_event_names() for the catalogue.
    events: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["normal", "tui", "web"] = "normal"
    # Backward-compat singleton (Phase 1–2). New configs may use `brokers:` for
    # multi-account; either path resolves to `broker_list` via the validator.
    broker: BrokerConfig | None = None
    brokers: list[BrokerConfig] | None = None
    data_dir: Path = Path("./data")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    timezone: str = "UTC"
    web: WebConfig = WebConfig()
    risk: RiskConfig = RiskConfig()
    metrics: MetricsConfig = MetricsConfig()
    notifications: list[NotificationChannelConfig] = Field(default_factory=list)
    allow_unsafe_inprocess_mt5_multi_account: bool = False

    @field_validator("brokers")
    @classmethod
    def _unique_broker_ids(
        cls, v: list[BrokerConfig] | None
    ) -> list[BrokerConfig] | None:
        if not v:
            return v
        ids = [b.id for b in v]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate broker ids: {dupes}")
        return v

    @model_validator(mode="after")
    def _normalise_brokers(self) -> AppConfig:
        # Exactly one of `broker:` or `brokers:` must be present.
        if self.broker is None and not self.brokers:
            raise ValueError(
                "AppConfig requires `broker:` (singular) or `brokers:` (list of accounts)"
            )
        if self.broker is not None and self.brokers:
            raise ValueError(
                "AppConfig has both `broker:` and `brokers:` — keep only one"
            )
        mt5_count = sum(1 for broker in self.broker_list if broker.type == "mt5")
        if mt5_count > 1 and not self.allow_unsafe_inprocess_mt5_multi_account:
            raise ValueError(
                "multiple in-process MT5 brokers are disabled by default because "
                "the MetaTrader5 Python SDK is process-global and not safely "
                "isolated per account. Use one MT5 account per process/terminal, "
                "or set allow_unsafe_inprocess_mt5_multi_account: true only for "
                "explicit testing."
            )
        return self

    @property
    def broker_list(self) -> list[BrokerConfig]:
        """Unified accessor used by the runtime — returns the broker list
        regardless of which YAML shape was supplied."""
        if self.brokers:
            return list(self.brokers)
        assert self.broker is not None  # guarded by _normalise_brokers
        return [self.broker]

    @property
    def primary_broker_config(self) -> BrokerConfig:
        """First broker config regardless of single/multi mode.

        Use this anywhere the legacy code path read ``app_cfg.broker.X``
        for logging / display / Strategy-Tester defaulting — in multi-
        account mode ``broker`` is None and direct access raises, but
        ``broker_list`` is guaranteed non-empty by ``_normalise_brokers``.
        """
        return self.broker_list[0]


# --- Strategies ---------------------------------------------------------------


class StrategyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    class_path: str = Field(min_length=1)
    enabled: bool = True
    # In a multi-account setup, picks which broker.id the strategy trades
    # against. Single-broker configs leave this at the default — it resolves
    # to the (only) broker via the broker pool.
    account: str = Field("default", min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
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


def _coerce_feed_list(
    *,
    symbol: str | None,
    timeframe: Timeframe | None,
    symbols: list[str] | None,
    timeframes: list[Timeframe] | None,
    feeds: list[Subscription] | None,
) -> list[Subscription]:
    """Normalise a config's feed shape into a deterministic list of subscriptions.

    Accepts exactly ONE of these three input shapes:

      • `symbol` + `timeframe`            — legacy single-feed (Phase 1–3)
      • `symbols` + `timeframes`          — cross product (every symbol × every tf)
      • `feeds`                           — explicit list (mixed symbol × tf)

    Mixing shapes is an error. The returned list is sorted by `(symbol, tf.value)`
    so backtest replay is deterministic across runs.
    """
    sources: list[str] = []
    if symbol is not None and timeframe is not None:
        sources.append("singular")
    if symbols or timeframes:
        if not (symbols and timeframes):
            raise ValueError(
                "`symbols` and `timeframes` must both be supplied together"
            )
        sources.append("plural")
    if feeds:
        sources.append("feeds")
    if not sources:
        raise ValueError(
            "config must declare ONE of: `symbol`+`timeframe`, "
            "`symbols`+`timeframes`, or `feeds`"
        )
    if len(sources) > 1:
        raise ValueError(
            f"config mixes feed shapes {sources!r}; keep exactly one"
        )

    if sources == ["singular"]:
        assert symbol is not None
        assert timeframe is not None
        out = [Subscription(symbol=symbol, timeframe=timeframe)]
    elif sources == ["plural"]:
        assert symbols is not None
        assert timeframes is not None
        out = [Subscription(symbol=s, timeframe=tf) for s in symbols for tf in timeframes]
    else:
        assert feeds is not None
        out = list(feeds)

    # Deterministic ordering for backtest replay tie-breaks.
    out.sort(key=lambda f: (f.symbol, f.timeframe.value))
    # Dedupe — same (symbol, tf) declared twice would only confuse the merge.
    seen: set[tuple[str, str]] = set()
    deduped: list[Subscription] = []
    for f in out:
        key = (f.symbol, f.timeframe.value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)
    return deduped


def _validate_symbol_contract_sizes(v: dict[str, float]) -> dict[str, float]:
    for symbol, contract_size in v.items():
        if contract_size <= 0:
            raise ValueError(
                f"symbol_contract_sizes[{symbol!r}] must be > 0"
            )
    return v


class BacktestRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    mode: Literal["file", "mt5_tester", "mt4_tester"] = "file"
    strategy_id: str
    # Singular (legacy, Phase 1–3) — keep optional so plural/feeds configs can omit.
    symbol: str | None = None
    timeframe: Timeframe | None = None
    # Plural (Phase 4) — cross product
    symbols: list[str] | None = None
    timeframes: list[Timeframe] | None = None
    # Explicit list (Phase 4) — mixed symbol/tf
    feeds: list[Subscription] | None = None
    start: datetime
    end: datetime
    initial_balance: float = Field(10_000.0, gt=0)
    symbol_contract_sizes: dict[str, float] = Field(default_factory=dict)
    data_source: Path | None = None
    slippage_pips: float = Field(0.0, ge=0)
    slippage_model: Literal["fixed", "spread", "volatility"] = "fixed"
    slippage_volatility_factor: float = Field(0.25, gt=0)
    # Trading costs (account currency). All default to 0 → existing runs are
    # unchanged. `volume` is in lots, so cost = rate × volume.
    #   commission_per_lot — charged per lot *per side* (open AND close), so a
    #     round-turn costs 2×. Set to your broker's per-lot-per-side commission.
    #   swap_{long,short}_per_lot — per lot *per night held*, signed (negative =
    #     you pay, positive = you receive). Long and short usually differ.
    #   swap_rollover_hour_utc — the UTC hour a night is counted (the rollover).
    commission_per_lot: float = Field(0.0, ge=0)
    swap_long_per_lot: float = 0.0
    swap_short_per_lot: float = 0.0
    swap_rollover_hour_utc: int = Field(21, ge=0, le=23)
    granularity: Literal["bar", "tick"] = "bar"
    # Playback throttle for the backtest replay loop.
    #   0   → max speed (default, current behavior — replay as fast as possible)
    #   1   → real-time (1 sim-second per wall-second)
    #   >1  → faster than real-time (e.g. 60 = 1 sim-minute per wall-second)
    #   <1  → slow-motion (e.g. 0.5 = half real-time)
    # The throttle paces *event delivery* to the bus; it does not affect
    # SimClock or any computed metric. Use it for UI replay, demos, debug
    # sessions, or stress-testing async paths at sub-real-time pacing.
    speed: float = Field(0.0, ge=0)

    @field_validator("start", "end")
    @classmethod
    def _require_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("start/end must include a timezone (e.g. ...T00:00:00Z)")
        return v

    @field_validator("symbol_contract_sizes")
    @classmethod
    def _contract_sizes_positive(cls, v: dict[str, float]) -> dict[str, float]:
        return _validate_symbol_contract_sizes(v)

    @model_validator(mode="after")
    def _normalise_feeds(self) -> BacktestRunConfig:
        # Compute and stash on a private attribute; expose via `feed_list`.
        feeds = _coerce_feed_list(
            symbol=self.symbol,
            timeframe=self.timeframe,
            symbols=self.symbols,
            timeframes=self.timeframes,
            feeds=self.feeds,
        )
        # Back-fill `symbol`/`timeframe` with the primary feed so code that
        # still reads them directly (MT5 tester, sidecar JSON, web viewer)
        # keeps working. Pydantic frozen=False allows this; we set via
        # __dict__ to bypass the model_validator running again.
        object.__setattr__(self, "symbol", feeds[0].symbol)
        object.__setattr__(self, "timeframe", feeds[0].timeframe)
        object.__setattr__(self, "_feed_list", feeds)
        return self

    @property
    def feed_list(self) -> list[Subscription]:
        """Unified accessor — returns the normalised feed list regardless of
        which input shape was supplied (singular / plural / explicit)."""
        # `_feed_list` is set by the model_validator; if we're called before
        # validation (shouldn't happen in practice) fall back to recomputation.
        existing = getattr(self, "_feed_list", None)
        if existing is not None:
            return list(existing)
        return _coerce_feed_list(
            symbol=self.symbol,
            timeframe=self.timeframe,
            symbols=self.symbols,
            timeframes=self.timeframes,
            feeds=self.feeds,
        )


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
    # Singular (legacy) — optional so plural/feeds configs can omit
    symbol: str | None = None
    timeframe: Timeframe | None = None
    # Plural (Phase 4)
    symbols: list[str] | None = None
    timeframes: list[Timeframe] | None = None
    # Explicit list (Phase 4)
    feeds: list[Subscription] | None = None
    start: datetime
    end: datetime
    initial_balance: float = Field(10_000.0, gt=0)
    symbol_contract_sizes: dict[str, float] = Field(default_factory=dict)
    slippage_pips: float = Field(0.0, ge=0)
    slippage_model: Literal["fixed", "spread", "volatility"] = "fixed"
    slippage_volatility_factor: float = Field(0.25, gt=0)
    data_source: Path
    parameter_grid: dict[str, list[Any]] = Field(default_factory=dict)
    # rank_by accepts a built-in MetricName OR any name defined in
    # custom_metrics (Phase 7.C). Relaxed from Literal to str + runtime check.
    rank_by: str = "net_pnl"
    top_n: int = Field(10, ge=1, le=1000)
    granularity: Literal["bar", "tick"] = "bar"
    # Phase 6.3.A/B — pluggable search backend
    algo: Literal["grid", "optuna", "random", "genetic"] = "grid"
    n_trials: int = Field(100, ge=1)        # for optuna/random; grid uses cartesian; genetic uses pop×gen
    random_seed: int | None = None          # reproducibility for adaptive algos
    # Genetic-specific knobs (ignored by other algos)
    population_size: int = Field(20, ge=2)
    generations: int = Field(10, ge=1)
    elite_size: int = Field(2, ge=0)
    mutation_rate: float = Field(0.1, ge=0.0, le=1.0)
    # Phase 7.B — multi-objective (Pareto frontier). When supplied,
    # the sweep computes the Pareto frontier across these objectives
    # AND still returns the single-objective ranking (rank_by). Each
    # entry is {"metric": name, "direction": "max" | "min"}.
    objectives: list[dict[str, str]] | None = None
    # Phase 7.C — user-defined metrics. Each entry maps a custom metric
    # name to a DSL expression (e.g. "sharpe - 0.5 * max_drawdown / 10").
    # Custom metrics are computed per cell post-backtest and are then
    # available to rank_by + objectives. Expressions can only reference
    # built-in metrics (no custom→custom chains).
    custom_metrics: dict[str, str] = Field(default_factory=dict)

    @field_validator("start", "end")
    @classmethod
    def _require_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("start/end must include a timezone (e.g. ...T00:00:00Z)")
        return v

    @field_validator("symbol_contract_sizes")
    @classmethod
    def _contract_sizes_positive(cls, v: dict[str, float]) -> dict[str, float]:
        return _validate_symbol_contract_sizes(v)

    @field_validator("parameter_grid")
    @classmethod
    def _grid_nonempty(cls, v: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if not v:
            raise ValueError("parameter_grid must declare at least one parameter")
        for name, values in v.items():
            if not values:
                raise ValueError(f"parameter_grid['{name}'] is empty — list one value at minimum")
        return v

    @model_validator(mode="after")
    def _normalise_feeds(self) -> SweepRunConfig:
        feeds = _coerce_feed_list(
            symbol=self.symbol,
            timeframe=self.timeframe,
            symbols=self.symbols,
            timeframes=self.timeframes,
            feeds=self.feeds,
        )
        object.__setattr__(self, "symbol", feeds[0].symbol)
        object.__setattr__(self, "timeframe", feeds[0].timeframe)
        object.__setattr__(self, "_feed_list", feeds)
        return self

    @property
    def feed_list(self) -> list[Subscription]:
        existing = getattr(self, "_feed_list", None)
        if existing is not None:
            return list(existing)
        return _coerce_feed_list(
            symbol=self.symbol,
            timeframe=self.timeframe,
            symbols=self.symbols,
            timeframes=self.timeframes,
            feeds=self.feeds,
        )


class WalkForwardConfig(BaseModel):
    """Walk-forward optimization config — fit on rolling/expanding windows,
    evaluate on the trailing out-of-sample slice of each fold (Phase 6.3.C)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    strategy_id: str
    # Feed shape — same singular/plural/feeds normalisation as sweep configs.
    symbol: str | None = None
    timeframe: Timeframe | None = None
    symbols: list[str] | None = None
    timeframes: list[Timeframe] | None = None
    feeds: list[Subscription] | None = None
    start: datetime
    end: datetime
    initial_balance: float = Field(10_000.0, gt=0)
    symbol_contract_sizes: dict[str, float] = Field(default_factory=dict)
    data_source: Path
    # Walk-forward specific
    n_folds: int = Field(5, ge=2)
    in_sample_pct: float = Field(0.7, gt=0.0, lt=1.0)
    scheme: Literal["rolling", "expanding"] = "rolling"
    # Inner-sweep configuration (same algos / knobs as SweepRunConfig)
    parameter_grid: dict[str, list[Any]] = Field(default_factory=dict)
    # rank_by accepts a built-in MetricName OR any name defined in
    # custom_metrics (Phase 7.C). Plain str type with validation in the
    # model_validator below.
    rank_by: str = "net_pnl"
    algo: Literal["grid", "optuna", "random", "genetic"] = "grid"
    n_trials: int = Field(100, ge=1)
    random_seed: int | None = None
    population_size: int = Field(20, ge=2)
    generations: int = Field(10, ge=1)
    elite_size: int = Field(2, ge=0)
    mutation_rate: float = Field(0.1, ge=0.0, le=1.0)

    @field_validator("start", "end")
    @classmethod
    def _require_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("start/end must include a timezone (e.g. ...T00:00:00Z)")
        return v

    @field_validator("symbol_contract_sizes")
    @classmethod
    def _contract_sizes_positive(cls, v: dict[str, float]) -> dict[str, float]:
        return _validate_symbol_contract_sizes(v)

    @field_validator("parameter_grid")
    @classmethod
    def _grid_nonempty(cls, v: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if not v:
            raise ValueError("parameter_grid must declare at least one parameter")
        for name, values in v.items():
            if not values:
                raise ValueError(f"parameter_grid['{name}'] is empty")
        return v

    @model_validator(mode="after")
    def _normalise_feeds(self) -> WalkForwardConfig:
        feeds = _coerce_feed_list(
            symbol=self.symbol,
            timeframe=self.timeframe,
            symbols=self.symbols,
            timeframes=self.timeframes,
            feeds=self.feeds,
        )
        object.__setattr__(self, "symbol", feeds[0].symbol)
        object.__setattr__(self, "timeframe", feeds[0].timeframe)
        object.__setattr__(self, "_feed_list", feeds)
        return self

    @property
    def feed_list(self) -> list[Subscription]:
        existing = getattr(self, "_feed_list", None)
        if existing is not None:
            return list(existing)
        return _coerce_feed_list(
            symbol=self.symbol,
            timeframe=self.timeframe,
            symbols=self.symbols,
            timeframes=self.timeframes,
            feeds=self.feeds,
        )


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


# --- Snapshot of full config (used by the reloader) --------------------------


class FullConfig(BaseModel):
    """Combined snapshot used by the hot-reloader to diff old vs. new."""

    model_config = ConfigDict(extra="forbid")

    app: AppConfig
    strategies: StrategiesConfig
    backtest: BacktestConfig
