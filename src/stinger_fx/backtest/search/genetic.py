"""GeneticSearch — simple genetic algorithm over a discrete parameter grid.

A textbook GA in ~120 lines, no third-party deps. Good for very large
parameter spaces where TPE (Optuna) struggles, or when you want to
emphasise exploration over the smooth-objective assumption Bayesian
samplers make.

Generation flow:
  1. Generation 0 is filled with random individuals (uniform from grid).
  2. The runner suggests each one, runs a backtest, reports the score.
  3. Once the whole population is scored, the next generation is built:
      * Top ``elite_size`` survive unchanged (elitism)
      * Remaining slots filled by tournament selection of two parents,
        single-point crossover (each gene randomly inherited from either
        parent), and per-gene mutation at ``mutation_rate``
  4. Repeat for ``generations`` total generations.

Total trials = ``population_size × generations``.

The cross-generation state lives in the search instance itself; the
runner just calls ``suggest()`` / ``report()`` in the usual loop and the
genetic logic happens transparently between generations.
"""

from __future__ import annotations

import random
from typing import Any


class GeneticSearch:
    """Tournament-selection GA with elitism."""

    def __init__(
        self,
        parameter_grid: dict[str, list[Any]],
        *,
        population_size: int = 20,
        generations: int = 10,
        elite_size: int = 2,
        mutation_rate: float = 0.1,
        tournament_size: int = 3,
        random_seed: int | None = None,
    ) -> None:
        if population_size <= 0:
            raise ValueError(f"population_size must be > 0, got {population_size}")
        if generations <= 0:
            raise ValueError(f"generations must be > 0, got {generations}")
        if not (0 <= elite_size < population_size):
            raise ValueError(
                f"elite_size must be in [0, population_size), got {elite_size}"
            )
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError(f"mutation_rate must be in [0, 1], got {mutation_rate}")
        if not parameter_grid:
            raise ValueError("parameter_grid must be non-empty")
        for name, values in parameter_grid.items():
            if not values:
                raise ValueError(f"parameter_grid[{name!r}] must have at least one value")

        self._grid = dict(parameter_grid)
        self._pop_size = population_size
        self._generations = generations
        self._elite_size = elite_size
        self._mutation_rate = mutation_rate
        self._tournament_size = max(2, min(tournament_size, population_size))
        self._rng = random.Random(random_seed)

        # Per-generation state
        self._current_gen = 0
        self._current_pop: list[dict[str, Any]] = self._initial_population()
        self._suggested_in_gen = 0
        # Scores keyed by repr(sorted(params.items())) so identical individuals
        # produced by elitism/crossover collide gracefully
        self._scores: dict[str, float] = {}

    @property
    def total_trials(self) -> int | None:
        return self._pop_size * self._generations

    # --- Search API ---------------------------------------------------------

    def suggest(self) -> dict[str, Any] | None:
        if self._current_gen >= self._generations:
            return None
        if self._suggested_in_gen >= self._pop_size:
            # Generation complete — breed the next one and continue.
            self._advance_generation()
            self._current_gen += 1
            self._suggested_in_gen = 0
            if self._current_gen >= self._generations:
                return None
        individual = self._current_pop[self._suggested_in_gen]
        self._suggested_in_gen += 1
        return dict(individual)  # defensive copy

    def report(self, params: dict[str, Any], score: float) -> None:
        self._scores[self._key(params)] = score

    # --- Genetic operators --------------------------------------------------

    def _initial_population(self) -> list[dict[str, Any]]:
        return [self._random_individual() for _ in range(self._pop_size)]

    def _random_individual(self) -> dict[str, Any]:
        return {name: self._rng.choice(values) for name, values in self._grid.items()}

    def _advance_generation(self) -> None:
        """Score current pop, keep elites, breed children, replace."""
        # Rank current population by score (descending)
        scored: list[tuple[float, dict[str, Any]]] = []
        for ind in self._current_pop:
            score = self._scores.get(self._key(ind), float("-inf"))
            scored.append((score, ind))
        scored.sort(key=lambda t: t[0], reverse=True)

        # Elitism — top survivors carry over unchanged
        next_pop: list[dict[str, Any]] = [dict(ind) for _, ind in scored[: self._elite_size]]

        # Fill the rest via tournament + crossover + mutation
        while len(next_pop) < self._pop_size:
            p1 = self._tournament(scored)
            p2 = self._tournament(scored)
            child = self._crossover(p1, p2)
            child = self._mutate(child)
            next_pop.append(child)

        self._current_pop = next_pop
        # New generation starts clean — old scores are stale
        self._scores.clear()

    def _tournament(
        self, scored: list[tuple[float, dict[str, Any]]]
    ) -> dict[str, Any]:
        contestants = self._rng.sample(
            scored, min(self._tournament_size, len(scored))
        )
        return max(contestants, key=lambda t: t[0])[1]

    def _crossover(
        self, parent_a: dict[str, Any], parent_b: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            name: (parent_a[name] if self._rng.random() < 0.5 else parent_b[name])
            for name in self._grid
        }

    def _mutate(self, individual: dict[str, Any]) -> dict[str, Any]:
        out = dict(individual)
        for name, values in self._grid.items():
            if self._rng.random() < self._mutation_rate:
                out[name] = self._rng.choice(values)
        return out

    # --- Helpers ------------------------------------------------------------

    @staticmethod
    def _key(params: dict[str, Any]) -> str:
        return repr(sorted(params.items()))
