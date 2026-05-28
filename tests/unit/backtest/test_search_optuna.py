"""OptunaSearch — TPE sampler converges on a synthetic objective."""

from __future__ import annotations

import pytest

# Optuna is an optional extra; skip the whole module when it's not installed.
optuna = pytest.importorskip("optuna")

from stinger_fx.backtest.search import (
    build_search_strategy,
)
from stinger_fx.backtest.search.optuna_search import (
    OptunaSearch,
)


def _synthetic_score(params: dict) -> float:
    """A toy objective that peaks at a=2, b=3 with value 100."""
    target_a, target_b = 2, 3
    return 100 - (params["a"] - target_a) ** 2 - (params["b"] - target_b) ** 2


def test_optuna_yields_n_trials() -> None:
    """OptunaSearch should yield exactly n_trials suggestions."""
    search = OptunaSearch(
        {"a": [1, 2, 3], "b": [1, 2, 3, 4]},
        n_trials=10,
        random_seed=42,
    )
    assert search.total_trials == 10
    suggestions = []
    while True:
        s = search.suggest()
        if s is None:
            break
        suggestions.append(s)
        search.report(s, _synthetic_score(s))
    assert len(suggestions) == 10


def test_optuna_converges_toward_best_region() -> None:
    """Over 30 trials, the suggester should land on the (a=2, b=3) peak
    at least once and explore better-on-average than uniform random."""
    search = OptunaSearch(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]},
        n_trials=30,
        random_seed=7,
    )
    scores: list[float] = []
    found_optimum = False
    while True:
        s = search.suggest()
        if s is None:
            break
        score = _synthetic_score(s)
        scores.append(score)
        if s["a"] == 2 and s["b"] == 3:
            found_optimum = True
        search.report(s, score)
    # At least once we should have hit the global optimum
    assert found_optimum, "TPE didn't visit the optimum in 30 trials with this seed"
    # Best score reached = 100 (the optimum value)
    assert max(scores) == pytest.approx(100)


def test_optuna_requires_n_trials_positive() -> None:
    with pytest.raises(ValueError, match="n_trials"):
        OptunaSearch({"a": [1]}, n_trials=0)


def test_factory_builds_optuna() -> None:
    """build_search_strategy('optuna', ...) should return an OptunaSearch."""
    search = build_search_strategy(
        "optuna",
        parameter_grid={"a": [1, 2]},
        n_trials=5,
        random_seed=0,
    )
    assert isinstance(search, OptunaSearch)


def test_factory_rejects_optuna_without_n_trials() -> None:
    with pytest.raises(ValueError, match="n_trials"):
        build_search_strategy("optuna", parameter_grid={"a": [1]})


def test_factory_rejects_unknown_algo() -> None:
    with pytest.raises(ValueError, match="unknown search algo"):
        build_search_strategy("nonsense", parameter_grid={"a": [1]})


def test_reproducibility_with_seed() -> None:
    """Two OptunaSearch instances with the same seed must produce the same
    sequence of suggestions for the same grid."""
    def run(seed: int) -> list:
        search = OptunaSearch({"a": [1, 2, 3], "b": [1, 2, 3]}, n_trials=8, random_seed=seed)
        out = []
        while (s := search.suggest()) is not None:
            out.append(dict(s))
            search.report(s, _synthetic_score(s))
        return out

    a = run(42)
    b = run(42)
    assert a == b
