"""GeneticSearch — tournament selection, crossover, mutation."""

from __future__ import annotations

import pytest

from stinger_fx.backtest.search import (
    GeneticSearch,
    SearchStrategy,
    build_search_strategy,
)


def _quadratic_objective(params: dict) -> float:
    """Toy fitness: peaks at a=2, b=3, value 100."""
    return 100 - (params["a"] - 2) ** 2 - (params["b"] - 3) ** 2


def test_ga_yields_pop_size_x_generations() -> None:
    ga = GeneticSearch(
        {"a": [1, 2, 3], "b": [1, 2, 3]},
        population_size=5,
        generations=3,
        random_seed=0,
    )
    assert ga.total_trials == 15
    count = 0
    while (s := ga.suggest()) is not None:
        ga.report(s, _quadratic_objective(s))
        count += 1
    assert count == 15


def test_ga_satisfies_protocol() -> None:
    ga = GeneticSearch({"a": [1]}, population_size=4, generations=1)
    assert isinstance(ga, SearchStrategy)


def test_ga_converges_toward_optimum() -> None:
    """Over 5×6 = 30 trials with a smooth objective, GA should find or
    get very close to the peak (a=2, b=3, score=100)."""
    ga = GeneticSearch(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]},
        population_size=5,
        generations=6,
        elite_size=1,
        mutation_rate=0.2,
        random_seed=7,
    )
    best_score = float("-inf")
    while (s := ga.suggest()) is not None:
        score = _quadratic_objective(s)
        ga.report(s, score)
        best_score = max(best_score, score)
    # With this seed and depth, we should find the peak (score=100)
    assert best_score == pytest.approx(100), f"GA didn't find optimum, best={best_score}"


def test_ga_elite_preserved_to_next_gen() -> None:
    """The top-elite_size individuals from gen N must appear unchanged in gen N+1."""
    ga = GeneticSearch(
        {"a": [1, 2, 3], "b": [1, 2, 3]},
        population_size=5,
        generations=2,
        elite_size=2,
        mutation_rate=0.0,  # disable mutation so we can track elite cleanly
        random_seed=42,
    )
    gen0: list[dict] = []
    for _ in range(5):
        s = ga.suggest()
        ga.report(s, _quadratic_objective(s))
        gen0.append(s)
    # Best 2 of gen0 by score
    ranked = sorted(gen0, key=_quadratic_objective, reverse=True)
    elite = ranked[:2]
    # First 2 suggestions of gen1 should be the elite (in order of score)
    gen1_first = ga.suggest()
    gen1_second = ga.suggest()
    assert gen1_first in elite
    assert gen1_second in elite


def test_ga_reproducible_with_seed() -> None:
    def run(seed: int) -> list:
        ga = GeneticSearch(
            {"a": [1, 2, 3], "b": [4, 5, 6]},
            population_size=4,
            generations=3,
            random_seed=seed,
        )
        out = []
        while (s := ga.suggest()) is not None:
            ga.report(s, _quadratic_objective(s))
            out.append(dict(s))
        return out

    assert run(123) == run(123)


def test_ga_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        GeneticSearch({"a": [1]}, population_size=0, generations=1)
    with pytest.raises(ValueError):
        GeneticSearch({"a": [1]}, population_size=5, generations=0)
    with pytest.raises(ValueError, match="elite_size"):
        # elite_size must be < population_size
        GeneticSearch({"a": [1]}, population_size=5, generations=1, elite_size=5)
    with pytest.raises(ValueError, match="mutation_rate"):
        GeneticSearch({"a": [1]}, population_size=5, generations=1, mutation_rate=1.5)


def test_factory_builds_genetic() -> None:
    search = build_search_strategy(
        "genetic",
        parameter_grid={"a": [1, 2]},
        population_size=4,
        generations=2,
        random_seed=0,
    )
    assert isinstance(search, GeneticSearch)


def test_factory_rejects_unknown_algo() -> None:
    """The error message must mention all supported algos."""
    with pytest.raises(ValueError, match="grid.*optuna.*random.*genetic"):
        build_search_strategy("magic", parameter_grid={"a": [1]})
