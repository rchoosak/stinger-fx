"""Walk-forward optimization — fit on past data, evaluate on future.

A backtest result that includes the parameters chosen by an in-sample
sweep is by definition over-fit to that window. Walk-forward checks
whether the chosen parameters generalise: it splits the full date range
into ``n_folds`` chunks, then for each fold runs a parameter sweep on
the in-sample portion and applies the winning parameters to the
out-of-sample portion. Aggregating OOS performance across folds gives
an honest estimate of how the strategy actually performs in production.

Two schemes:

  * **Rolling**: each fold uses only its own window for in-sample.
    Mimics "re-fit every period from a fixed look-back".
  * **Expanding**: each fold uses all data from ``cfg.start`` up to the
    current window's in-sample end. Mimics "use every piece of history
    you have" — the in-sample window grows over time.

The runner uses the pluggable :class:`SearchStrategy` from Phase 6.3.A/B
so you can choose grid / optuna / random / genetic per fold by setting
``cfg.algo``.

Consistency between in-sample and out-of-sample performance is reported
as the Pearson correlation across folds: a strategy that overfits gets a
high IS metric but low (or negative) OOS metric across folds, dragging
the correlation toward zero or below.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from stinger_fx.backtest.file_backtester import FileBacktester
from stinger_fx.backtest.sweep import ParameterSweep, SweepReport
from stinger_fx.config.models import (
    BacktestRunConfig,
    MetricName,
    StrategyEntry,
    SweepRunConfig,
    WalkForwardConfig,
)
from stinger_fx.data import SqliteStore

logger = logging.getLogger("stinger.backtest.walk_forward")


# --- Data classes -----------------------------------------------------------


@dataclass
class WalkForwardFold:
    """One fold's date windows (before any backtesting happens)."""

    index: int
    in_sample: tuple[datetime, datetime]
    out_of_sample: tuple[datetime, datetime]


@dataclass
class WalkForwardFoldResult:
    """One fold's result after both in-sample sweep and OOS backtest."""

    index: int
    in_sample_start: datetime
    in_sample_end: datetime
    oos_start: datetime
    oos_end: datetime
    best_params: dict[str, Any]
    in_sample_metrics: dict[str, float]
    oos_metrics: dict[str, float]

    def to_json(self) -> dict:
        return {
            "index": self.index,
            "in_sample": [self.in_sample_start.isoformat(), self.in_sample_end.isoformat()],
            "out_of_sample": [self.oos_start.isoformat(), self.oos_end.isoformat()],
            "best_params": self.best_params,
            "in_sample_metrics": self.in_sample_metrics,
            "oos_metrics": self.oos_metrics,
        }


@dataclass
class WalkForwardReport:
    """Aggregate of all fold results plus consistency score."""

    id: str
    strategy_id: str
    started_at: datetime
    finished_at: datetime
    rank_by: MetricName
    scheme: str
    n_folds: int
    folds: list[WalkForwardFoldResult] = field(default_factory=list)

    @property
    def consistency_score(self) -> float:
        """Pearson correlation of IS metric vs OOS metric across folds.

        Range: -1 (perfectly anti-correlated, strong overfit signal)
                0 (no relationship)
              +1 (perfectly correlated, strong generalisation)

        Returns 0.0 when there are fewer than 2 folds or when either
        series is constant (no variance to correlate).
        """
        if len(self.folds) < 2:
            return 0.0
        xs = [f.in_sample_metrics.get(self.rank_by, 0.0) for f in self.folds]
        ys = [f.oos_metrics.get(self.rank_by, 0.0) for f in self.folds]
        return _pearson(xs, ys)

    @property
    def avg_oos_metric(self) -> float | None:
        if not self.folds:
            return None
        return sum(f.oos_metrics.get(self.rank_by, 0.0) for f in self.folds) / len(self.folds)

    def to_summary(self) -> dict:
        return {
            "id": self.id,
            "strategy_id": self.strategy_id,
            "scheme": self.scheme,
            "n_folds": self.n_folds,
            "rank_by": self.rank_by,
            "consistency_score": self.consistency_score,
            "avg_oos_metric": self.avg_oos_metric,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "folds": [f.to_json() for f in self.folds],
        }


# --- Fold slicing -----------------------------------------------------------


def slice_folds(
    start: datetime,
    end: datetime,
    n_folds: int,
    *,
    in_sample_pct: float = 0.7,
    scheme: str = "rolling",
) -> list[WalkForwardFold]:
    """Divide ``[start, end]`` into ``n_folds`` equal windows; within each,
    the first ``in_sample_pct`` is the in-sample period and the remainder
    is out-of-sample.

    For ``scheme="expanding"`` each fold's in-sample window starts from
    the global ``start`` instead of just its own slice.
    """
    if n_folds <= 0:
        raise ValueError(f"n_folds must be > 0, got {n_folds}")
    if not (0.0 < in_sample_pct < 1.0):
        raise ValueError(f"in_sample_pct must be in (0, 1), got {in_sample_pct}")
    if scheme not in ("rolling", "expanding"):
        raise ValueError(f"scheme must be 'rolling' or 'expanding', got {scheme!r}")
    if end <= start:
        raise ValueError("end must be after start")

    total_seconds = (end - start).total_seconds()
    step_seconds = total_seconds / n_folds
    folds: list[WalkForwardFold] = []
    for i in range(n_folds):
        win_start = start + timedelta(seconds=i * step_seconds)
        win_end = start + timedelta(seconds=(i + 1) * step_seconds)
        is_end = win_start + timedelta(seconds=step_seconds * in_sample_pct)
        is_start = start if scheme == "expanding" else win_start
        folds.append(
            WalkForwardFold(
                index=i,
                in_sample=(is_start, is_end),
                out_of_sample=(is_end, win_end),
            )
        )
    return folds


# --- Runner -----------------------------------------------------------------


class WalkForward:
    """Sweep-on-fold-then-OOS-evaluate workflow runner."""

    def __init__(
        self,
        *,
        base_strategy: StrategyEntry,
        sqlite_store: SqliteStore | None = None,
        report_dir: Path | None = None,
    ) -> None:
        self._base_strategy = base_strategy
        self._sqlite = sqlite_store
        self._report_dir = report_dir or Path("./data/walk_forward")

    async def run(self, cfg: WalkForwardConfig) -> WalkForwardReport:
        folds = slice_folds(
            cfg.start,
            cfg.end,
            cfg.n_folds,
            in_sample_pct=cfg.in_sample_pct,
            scheme=cfg.scheme,
        )
        started_at = datetime.now(UTC)
        results: list[WalkForwardFoldResult] = []
        logger.info(
            "walk_forward_started id=%s scheme=%s n_folds=%d strategy=%s",
            cfg.id, cfg.scheme, cfg.n_folds, cfg.strategy_id,
        )

        for fold in folds:
            # --- Fit: sweep on in-sample window -------------------------
            sweep_cfg = self._build_sweep_cfg(cfg, fold)
            sweep_report: SweepReport = await ParameterSweep(
                base_strategy=self._base_strategy,
                sqlite_store=self._sqlite,
                report_dir=self._report_dir / cfg.id / f"fold_{fold.index}_sweep",
            ).run(sweep_cfg)
            best = sweep_report.best()
            if best is None:
                logger.warning(
                    "walk_forward_fold_empty_sweep id=%s fold=%d", cfg.id, fold.index
                )
                continue

            # --- Evaluate: backtest on OOS window using best params -----
            oos_metrics = await self._evaluate_oos(cfg, fold, best.params)

            results.append(
                WalkForwardFoldResult(
                    index=fold.index,
                    in_sample_start=fold.in_sample[0],
                    in_sample_end=fold.in_sample[1],
                    oos_start=fold.out_of_sample[0],
                    oos_end=fold.out_of_sample[1],
                    best_params=best.params,
                    in_sample_metrics=best.metrics,
                    oos_metrics=oos_metrics,
                )
            )
            logger.info(
                "walk_forward_fold_done id=%s fold=%d best=%s is_%s=%s oos_%s=%s",
                cfg.id, fold.index, best.params,
                cfg.rank_by, best.metrics.get(cfg.rank_by),
                cfg.rank_by, oos_metrics.get(cfg.rank_by),
            )

        report = WalkForwardReport(
            id=cfg.id,
            strategy_id=cfg.strategy_id,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            rank_by=cfg.rank_by,
            scheme=cfg.scheme,
            n_folds=cfg.n_folds,
            folds=results,
        )
        await self._persist(cfg, report)
        return report

    # --- Internals ----------------------------------------------------------

    def _build_sweep_cfg(
        self, cfg: WalkForwardConfig, fold: WalkForwardFold
    ) -> SweepRunConfig:
        return SweepRunConfig(
            id=f"{cfg.id}_fold_{fold.index}",
            strategy_id=cfg.strategy_id,
            feeds=cfg.feed_list,
            start=fold.in_sample[0],
            end=fold.in_sample[1],
            initial_balance=cfg.initial_balance,
            data_source=cfg.data_source,
            parameter_grid=cfg.parameter_grid,
            rank_by=cfg.rank_by,
            algo=cfg.algo,
            n_trials=cfg.n_trials,
            random_seed=cfg.random_seed,
            population_size=cfg.population_size,
            generations=cfg.generations,
            elite_size=cfg.elite_size,
            mutation_rate=cfg.mutation_rate,
        )

    async def _evaluate_oos(
        self,
        cfg: WalkForwardConfig,
        fold: WalkForwardFold,
        best_params: dict[str, Any],
    ) -> dict[str, float]:
        # Build a per-fold strategy entry whose `params` is base + best overrides.
        merged_params = {**self._base_strategy.params, **best_params}
        oos_strategy = self._base_strategy.model_copy(
            update={
                "id": f"{self._base_strategy.id}__wf_{cfg.id}_fold_{fold.index}",
                "params": merged_params,
            }
        )
        oos_run_cfg = BacktestRunConfig(
            id=f"{cfg.id}_fold_{fold.index}_oos",
            mode="file",
            strategy_id=oos_strategy.id,
            feeds=cfg.feed_list,
            start=fold.out_of_sample[0],
            end=fold.out_of_sample[1],
            initial_balance=cfg.initial_balance,
            data_source=cfg.data_source,
        )
        bt = FileBacktester(
            strategy=oos_strategy,
            parquet_root=cfg.data_source,
            sqlite_store=self._sqlite,
            report_dir=self._report_dir / cfg.id / f"fold_{fold.index}_oos",
        )
        oos_report = await bt.run(oos_run_cfg)
        return oos_report.to_metrics_dict()

    async def _persist(self, cfg: WalkForwardConfig, report: WalkForwardReport) -> None:
        self._report_dir.mkdir(parents=True, exist_ok=True)
        path = self._report_dir / f"{cfg.id}_summary.json"
        path.write_text(json.dumps(report.to_summary(), indent=2, default=str))
        logger.info("walk_forward_persisted path=%s", path)


# --- Helpers ----------------------------------------------------------------


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson product-moment correlation. Returns 0.0 when either series is
    constant (no variance) or lengths differ."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mx) ** 2 for x in xs)
    var_y = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return 0.0
    return num / denom
