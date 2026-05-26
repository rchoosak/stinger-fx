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

from stinger_fx.backtest.search.grid import GridSearch


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
) -> SearchStrategy:
    """Factory — dispatch to the right SearchStrategy implementation.

    Parameters
    ----------
    algo:
        ``"grid"``, ``"optuna"``, ``"random"`` (random and genetic land
        in 6.3.B).
    parameter_grid:
        Map of param name → list of candidate values.
    n_trials:
        Required for non-grid algorithms; ignored by grid.
    random_seed:
        Reproducibility seed. Passed to Optuna's sampler.
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
    raise ValueError(
        f"unknown search algo {algo!r}; expected 'grid' or 'optuna'"
    )


__all__ = ["GridSearch", "SearchStrategy", "build_search_strategy"]
