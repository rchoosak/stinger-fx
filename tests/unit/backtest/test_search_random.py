"""RandomSearch — uniform random sampling, deterministic with seed."""

from __future__ import annotations

import pytest

from stinger_fx.backtest.search import (
    RandomSearch,
    SearchStrategy,
    build_search_strategy,
)


def test_random_yields_n_trials_then_none() -> None:
    rs = RandomSearch({"a": [1, 2, 3]}, n_trials=5, random_seed=0)
    assert rs.total_trials == 5
    out = []
    while (s := rs.suggest()) is not None:
        out.append(s)
    assert len(out) == 5


def test_random_samples_from_grid_only() -> None:
    """Every suggestion must use values from the supplied grid lists."""
    grid = {"a": [10, 20], "b": ["x", "y", "z"]}
    rs = RandomSearch(grid, n_trials=50, random_seed=42)
    while (s := rs.suggest()) is not None:
        assert s["a"] in (10, 20)
        assert s["b"] in ("x", "y", "z")


def test_random_satisfies_protocol() -> None:
    rs = RandomSearch({"a": [1]}, n_trials=1)
    assert isinstance(rs, SearchStrategy)


def test_random_reproducible_with_seed() -> None:
    """Same seed → same sequence."""
    def run(seed: int) -> list:
        rs = RandomSearch({"a": [1, 2, 3], "b": [4, 5, 6]}, n_trials=10, random_seed=seed)
        return [rs.suggest() for _ in range(10)]
    assert run(42) == run(42)


def test_random_rejects_invalid_n_trials() -> None:
    with pytest.raises(ValueError):
        RandomSearch({"a": [1]}, n_trials=0)


def test_random_rejects_empty_grid() -> None:
    with pytest.raises(ValueError):
        RandomSearch({}, n_trials=5)


def test_random_rejects_empty_values_list() -> None:
    with pytest.raises(ValueError):
        RandomSearch({"a": []}, n_trials=5)


def test_factory_builds_random() -> None:
    search = build_search_strategy(
        "random", parameter_grid={"a": [1, 2]}, n_trials=3, random_seed=0
    )
    assert isinstance(search, RandomSearch)


def test_factory_requires_n_trials_for_random() -> None:
    with pytest.raises(ValueError, match="n_trials"):
        build_search_strategy("random", parameter_grid={"a": [1]})
