"""GridSearch — exhaustive cartesian enumeration of a parameter grid.

Wraps the original ``enumerate_grid`` logic as a ``SearchStrategy``. Output
is deterministic: same grid produces the same sequence of suggestions.

``report()`` is a no-op — grid search doesn't learn from feedback. We
keep the method present for Protocol compliance.
"""

from __future__ import annotations

import itertools
from typing import Any


class GridSearch:
    """Cartesian product of the parameter grid."""

    def __init__(self, parameter_grid: dict[str, list[Any]]) -> None:
        self._grid = dict(parameter_grid)
        self._iter = iter(self._enumerate())
        self._total = self._compute_total()

    def _enumerate(self):
        if not self._grid:
            return iter([])
        keys = list(self._grid.keys())
        values_lists = [self._grid[k] for k in keys]
        return (
            dict(zip(keys, combo, strict=True))
            for combo in itertools.product(*values_lists)
        )

    def _compute_total(self) -> int:
        if not self._grid:
            return 0
        total = 1
        for vs in self._grid.values():
            total *= len(vs)
        return total

    @property
    def total_trials(self) -> int | None:
        return self._total

    def suggest(self) -> dict[str, Any] | None:
        return next(self._iter, None)

    def report(self, params: dict[str, Any], score: float) -> None:
        # Grid is non-adaptive — feedback is irrelevant.
        return None
