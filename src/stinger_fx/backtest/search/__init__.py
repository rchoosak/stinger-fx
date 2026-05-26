"""Parameter-search strategies for sweep / walk-forward / future optimizers.

Define a single ``SearchStrategy`` Protocol so the sweep runner is decoupled
from the actual search algorithm. Implementations:

  * :class:`GridSearch`   — exhaustive cartesian enumeration (default,
                            deterministic, no third-party deps)
  * :class:`OptunaSearch` — TPE Bayesian-style sampling over discrete
                            choices (requires ``optuna`` extra)
  * (future) :class:`RandomSearch`, :class:`GeneticSearch`

The sweep runner calls ``suggest()`` until it returns ``None``, then
publishes the cell score via ``report()`` so algorithms that learn from
feedback (Optuna, GA) can guide future suggestions.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from stinger_fx.backtest.search.genetic import GeneticSearch
from stinger_fx.backtest.search.grid import GridSearch
from stinger_fx.backtest.search.random_search import RandomSearch


@runtime_checkable
class SearchStrategy(Protocol):
    """Pluggable parameter-search algorithm.

    The runner pattern is::

        search = build_search_strategy(cfg)
        i = 0
        while True:
            params = search.suggest()
            if params is None:
                break
            score = run_backtest(params)  # cell metric, larger-is-better
            search.report(params, score)
            i += 1
    """

    @property
    def total_trials(self) -> int | None:
        """Upper bound on trials, or ``None`` if adaptive / unbounded."""
        ...

    def suggest(self) -> dict[str, Any] | None:
        """Return the next params to evaluate, or ``None`` when done."""
        ...

    def report(self, params: dict[str, Any], score: float) -> None:
        """Receive the score for a previously-suggested params dict.

        Score convention: larger is better. The sweep runner flips the
        sign for metrics like ``max_drawdown`` so this contract holds
        uniformly across algorithms.
        """
        ...


def build_search_strategy(
    algo: str,
    *,
    parameter_grid: dict[str, list[Any]],
    n_trials: int | None = None,
    random_seed: int | None = None,
    population_size: int = 20,
    generations: int = 10,
    elite_size: int = 2,
    mutation_rate: float = 0.1,
) -> SearchStrategy:
    """Factory — dispatch to the right SearchStrategy implementation.

    Parameters
    ----------
    algo:
        ``"grid"``, ``"optuna"``, ``"random"``, or ``"genetic"``.
    parameter_grid:
        Map of param name → list of candidate values.
    n_trials:
        Required for ``optuna`` and ``random``; ignored by grid/genetic
        (genetic computes total as ``population_size × generations``).
    random_seed:
        Reproducibility seed for adaptive / random backends.
    population_size, generations, elite_size, mutation_rate:
        Genetic-specific knobs; ignored by other backends.
    """
    if algo == "grid":
        return GridSearch(parameter_grid)
    if algo == "optuna":
        from stinger_fx.backtest.search.optuna_search import OptunaSearch

        if n_trials is None:
            raise ValueError("optuna algo requires n_trials")
        return OptunaSearch(
            parameter_grid, n_trials=n_trials, random_seed=random_seed
        )
    if algo == "random":
        if n_trials is None:
            raise ValueError("random algo requires n_trials")
        return RandomSearch(
            parameter_grid, n_trials=n_trials, random_seed=random_seed
        )
    if algo == "genetic":
        return GeneticSearch(
            parameter_grid,
            population_size=population_size,
            generations=generations,
            elite_size=elite_size,
            mutation_rate=mutation_rate,
            random_seed=random_seed,
        )
    raise ValueError(
        f"unknown search algo {algo!r}; "
        f"expected 'grid', 'optuna', 'random', or 'genetic'"
    )


__all__ = [
    "GeneticSearch",
    "GridSearch",
    "RandomSearch",
    "SearchStrategy",
    "build_search_strategy",
]
