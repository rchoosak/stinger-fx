"""YAML → Pydantic config models. Raises `ConfigError` on any failure."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from pydantic import ValidationError

from stinger_fx.config.models import (
    AppConfig,
    BacktestConfig,
    FullConfig,
    StrategiesConfig,
)
from stinger_fx.core.errors import ConfigError


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return data


def _validate[T](model: type[T], data: dict, source: Path) -> T:
    try:
        return model.model_validate(data)  # type: ignore[attr-defined]
    except ValidationError as e:
        raise ConfigError(f"validation failed for {source}:\n{e}") from e


def load_app(path: Path) -> AppConfig:
    return _validate(AppConfig, _load_yaml(path), path)


def load_strategies(path: Path) -> StrategiesConfig:
    return _validate(StrategiesConfig, _load_yaml(path), path)


def load_backtest(path: Path) -> BacktestConfig:
    return _validate(BacktestConfig, _load_yaml(path), path)


def load_all(config_dir: Path) -> FullConfig:
    """Load every YAML in a config directory into a single snapshot."""
    return FullConfig(
        app=load_app(config_dir / "app.yaml"),
        strategies=load_strategies(config_dir / "strategies.yaml"),
        backtest=load_backtest(config_dir / "backtest.yaml"),
    )


def hash_file(path: Path) -> str:
    """SHA-256 of a config file — used by the audit log."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()
