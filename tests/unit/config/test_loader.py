from __future__ import annotations

from pathlib import Path

import pytest

from stinger_fx.config import load_all, load_app, load_strategies
from stinger_fx.core.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"


def test_load_all_uses_repo_defaults() -> None:
    cfg = load_all(CONFIG_DIR)
    assert cfg.app.mode in {"normal", "tui", "web"}
    assert cfg.app.broker.type == "mt5"
    assert any(s.id == "ma_eurusd_m15" for s in cfg.strategies.strategies)


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_app(tmp_path / "nope.yaml")


def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "app.yaml"
    bad.write_text("mode: normal\nbroker:\n  type: !!python/object:foo\n")
    with pytest.raises(ConfigError):
        load_app(bad)


def test_invalid_field_raises_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "strategies.yaml"
    bad.write_text("strategies:\n  - id: bad\n    class_path: no_colon\n    enabled: true\n")
    with pytest.raises(ConfigError):
        load_strategies(bad)


def test_duplicate_strategy_ids_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "strategies.yaml"
    bad.write_text(
        "strategies:\n"
        "  - id: dup\n    class_path: a.b:C\n    params: {}\n"
        "  - id: dup\n    class_path: a.b:C\n    params: {}\n"
    )
    with pytest.raises(ConfigError):
        load_strategies(bad)
