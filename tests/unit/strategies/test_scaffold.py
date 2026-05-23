"""Scaffold generator — input validation, file creation, generated file is loadable."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from stinger_fx.strategies.scaffold import (
    ScaffoldError,
    derive_module_path,
    scaffold,
    snake_to_pascal,
    yaml_snippet,
)


def test_snake_to_pascal_examples() -> None:
    assert snake_to_pascal("foo") == "Foo"
    assert snake_to_pascal("ma_crossover") == "MaCrossover"
    assert snake_to_pascal("rsi_v2") == "RsiV2"


def test_scaffold_rejects_invalid_name(tmp_path: Path) -> None:
    for bad in ["1bad", "Bad", "with-dash", "with space", ""]:
        with pytest.raises(ScaffoldError):
            scaffold(bad, tmp_path)


def test_scaffold_writes_file_and_init(tmp_path: Path) -> None:
    out = scaffold("my_strategy", tmp_path / "user")
    assert out == tmp_path / "user" / "my_strategy.py"
    assert out.exists()
    assert (tmp_path / "user" / "__init__.py").exists()
    body = out.read_text()
    assert "class MyStrategy(BaseStrategy)" in body
    assert "class MyStrategyParams(StrategyParams)" in body
    assert 'name = "my_strategy"' in body


def test_scaffold_refuses_to_overwrite(tmp_path: Path) -> None:
    scaffold("foo", tmp_path)
    with pytest.raises(ScaffoldError):
        scaffold("foo", tmp_path)
    # --force overrides
    scaffold("foo", tmp_path, force=True)


def test_generated_module_is_importable_and_loads_via_registry(tmp_path: Path) -> None:
    """The strategy registry should be able to load_strategy_class on the
    freshly-generated file with no further edits — proving the template is
    syntactically + semantically valid against the current API."""
    from stinger_fx.strategies.registry import load_strategy_class, validate_params

    out = scaffold("regression_strat", tmp_path)
    # Make tmp_path importable so we can resolve "regression_strat:RegressionStrat"
    sys.path.insert(0, str(tmp_path))
    try:
        cls = load_strategy_class("regression_strat:RegressionStrat")
        assert cls.name == "regression_strat"
        params = validate_params(cls, {})
        subs = cls.subscriptions(params)
        assert subs and subs[0].symbol == "EURUSD"
    finally:
        sys.path.remove(str(tmp_path))
        # Clean module cache so subsequent runs don't see the stale class
        sys.modules.pop("regression_strat", None)
    assert out.exists()


def test_yaml_snippet_has_required_fields() -> None:
    snip = yaml_snippet("foo", "stinger_fx.strategies.user.foo")
    assert "id: foo" in snip
    assert "class_path: stinger_fx.strategies.user.foo:Foo" in snip
    assert "enabled: true" in snip


def test_derive_module_path_handles_src_layout(tmp_path: Path) -> None:
    # Simulate the canonical layout
    f = tmp_path / "src" / "stinger_fx" / "strategies" / "user" / "x.py"
    f.parent.mkdir(parents=True)
    f.write_text("")
    assert derive_module_path(f, tmp_path) == "stinger_fx.strategies.user.x"


def test_derive_module_path_handles_outside_repo(tmp_path: Path) -> None:
    other = tmp_path / "anywhere" / "y.py"
    other.parent.mkdir(parents=True)
    other.write_text("")
    # When the file isn't inside repo_root, return the stem.
    unrelated_root = tmp_path.parent / "no_such_root"
    assert derive_module_path(other, unrelated_root) == "y"
