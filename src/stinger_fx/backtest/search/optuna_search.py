"""OptunaSearch — TPE (Tree-structured Parzen Estimator) sampling.

Requires the ``optuna`` package — install via the ``optimize`` extra::

    uv sync --extra optimize

TPE is a Bayesian-style sampler that builds a probabilistic model of the
objective from past trials and biases future suggestions toward promising
regions. For a discrete parameter grid (the only shape the sweep config
supports today) it acts as a smart explorer of the cartesian space —
visiting bad combinations less often than uniform random would.

Score convention matches the SearchStrategy contract: larger is better.
The Optuna study is configured with ``direction="maximize"``; the sweep
runner flips the sign for smaller-is-better metrics like max_drawdown.
"""

from __future__ import annotations

from typing import Any


class OptunaSearch:
    """TPE sampler over a discrete parameter grid."""

    def __init__(
        self,
        parameter_grid: dict[str, list[Any]],
        *,
        n_trials: int,
        random_seed: int | None = None,
    ) -> None:
        if n_trials <= 0:
            raise ValueError(f"n_trials must be > 0, got {n_trials}")
        try:
            import optuna
        except ImportError as e:
            raise RuntimeError(
                "optuna is required for OptunaSearch; install with "
                "`uv sync --extra optimize`"
            ) from e
        # Silence optuna's INFO logs by default — the sweep runner produces
        # its own per-cell summary. Operators can re-enable via env if needed.
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self._optuna = optuna
        self._grid = dict(parameter_grid)
        self._n_trials = n_trials
        self._study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_seed),
        )
        # Trials are async — we track which trial each suggestion belongs to
        # so report() can tell() the right one.
        self._pending: dict[str, Any] = {}  # params_key → trial
        self._completed = 0

    @property
    def total_trials(self) -> int | None:
        return self._n_trials

    def suggest(self) -> dict[str, Any] | None:
        if self._completed >= self._n_trials:
            return None
        trial = self._study.ask()
        params: dict[str, Any] = {}
        for name, candidates in self._grid.items():
            params[name] = trial.suggest_categorical(name, candidates)
        # Stash the trial keyed by a hashable form of the params so report()
        # can find it. Optuna allows multiple identical suggestions; we just
        # keep the most recent and tell() in completion order.
        self._pending[self._key(params)] = trial
        return params

    def report(self, params: dict[str, Any], score: float) -> None:
        key = self._key(params)
        trial = self._pending.pop(key, None)
        if trial is None:
            # Shouldn't happen in the normal runner flow; skip rather than
            # raise so a single mismatched report doesn't kill the sweep.
            return
        self._study.tell(trial, score)
        self._completed += 1

    # --- Helpers ------------------------------------------------------------

    @staticmethod
    def _key(params: dict[str, Any]) -> str:
        # Sort for determinism; JSON-able primitives are the only thing we
        # ever put in a parameter grid.
        items = sorted(params.items())
        return repr(items)
