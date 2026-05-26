"""RandomSearch — uniform random sampling over a discrete parameter grid.

A faster baseline than exhaustive grid when the parameter space is large
and most cells are likely uninformative. ``report()`` is a no-op since
random search is non-adaptive (Optuna/GA are the adaptive backends).

Reproducibility: seed via ``random_seed`` for deterministic test runs.
Duplicates are allowed — if you suggest the same combo twice that's just
the price of pure random.
"""

from __future__ import annotations

import random
from typing import Any


class RandomSearch:
    """Uniform sample from the parameter grid for ``n_trials`` iterations."""

    def __init__(
        self,
        parameter_grid: dict[str, list[Any]],
        *,
        n_trials: int,
        random_seed: int | None = None,
    ) -> None:
        if n_trials <= 0:
            raise ValueError(f"n_trials must be > 0, got {n_trials}")
        if not parameter_grid:
            raise ValueError("parameter_grid must be non-empty")
        for name, values in parameter_grid.items():
            if not values:
                raise ValueError(f"parameter_grid[{name!r}] must have at least one value")
        self._grid = dict(parameter_grid)
        self._n_trials = n_trials
        self._yielded = 0
        self._rng = random.Random(random_seed)

    @property
    def total_trials(self) -> int | None:
        return self._n_trials

    def suggest(self) -> dict[str, Any] | None:
        if self._yielded >= self._n_trials:
            return None
        self._yielded += 1
        return {name: self._rng.choice(values) for name, values in self._grid.items()}

    def report(self, params: dict[str, Any], score: float) -> None:
        # Random search ignores feedback.
        return None
