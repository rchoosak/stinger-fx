"""Parameter sweep — cartesian product of strategy params, ranked by metric.

Each cell of the grid is materialised into a one-off StrategyEntry whose
`params` is a merge of the base strategy's params with the cell's overrides,
then run through the existing FileBacktester. Results are collected, ranked,
and persisted alongside the regular `backtest_runs` rows.

Designed to plug into the existing backtest infrastructure with no changes
to the core engine — the sweep just calls FileBacktester many times.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stinger_fx.backtest.file_backtester import FileBacktester
from stinger_fx.backtest.metric_dsl import MetricDSLError, compile_metric
from stinger_fx.backtest.pareto import Objective, ParetoResult, extract_pareto_frontier
from stinger_fx.backtest.reports import BacktestReport
from stinger_fx.backtest.search import build_search_strategy
from stinger_fx.config.models import (
    BacktestRunConfig,
    MetricName,
    StrategyEntry,
    SweepRunConfig,
)
from stinger_fx.data import SqliteStore

logger = logging.getLogger("stinger.backtest.sweep")


# Metrics where smaller is better (max_drawdown). Everything else: bigger.
_SMALLER_IS_BETTER: set[MetricName] = {"max_drawdown"}


@dataclass
class SweepCellResult:
    params: dict[str, Any]
    metrics: dict[str, float]


@dataclass
class SweepReport:
    sweep_id: str
    strategy_id: str
    started_at: datetime
    finished_at: datetime
    rank_by: MetricName | str  # MetricName built-in OR custom DSL expression
    total_combos: int
    results: list[SweepCellResult] = field(default_factory=list)
    # Phase 7.B — multi-objective Pareto frontier (None when no objectives
    # were declared on the sweep config). When set, summary JSON includes
    # the frontier subset + the full Pareto-tagged point list.
    pareto: ParetoResult | None = None

    @property
    def ranked(self) -> list[SweepCellResult]:
        reverse = self.rank_by not in _SMALLER_IS_BETTER
        return sorted(
            self.results,
            key=lambda r: r.metrics.get(self.rank_by, float("-inf") if reverse else float("inf")),
            reverse=reverse,
        )

    def best(self) -> SweepCellResult | None:
        return self.ranked[0] if self.results else None

    def top(self, n: int) -> list[SweepCellResult]:
        return self.ranked[:n]

    def to_summary(self) -> dict:
        best = self.best()
        return {
            "sweep_id": self.sweep_id,
            "strategy_id": self.strategy_id,
            "rank_by": self.rank_by,
            "total_combos": self.total_combos,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "best_params": best.params if best else None,
            "best_metric_value": best.metrics.get(self.rank_by) if best else None,
        }


def enumerate_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of the grid → list of param dicts.

    Example:
      {"fast": [5, 10], "slow": [20, 30]}
      → [{"fast":5,"slow":20}, {"fast":5,"slow":30}, {"fast":10,"slow":20}, {"fast":10,"slow":30}]
    """
    if not grid:
        return []
    keys = list(grid.keys())
    values_lists = [grid[k] for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values_lists)]


class ParameterSweep:
    """Run a sweep config: backtest every cell, rank, persist, return report."""

    def __init__(
        self,
        *,
        base_strategy: StrategyEntry,
        sqlite_store: SqliteStore | None = None,
        report_dir: Path | None = None,
    ) -> None:
        self._base_strategy = base_strategy
        self._sqlite = sqlite_store
        self._report_dir = report_dir or Path("./data/sweeps")

    async def run(self, cfg: SweepRunConfig) -> SweepReport:
        # Phase 6.3.A — pluggable search backend.  Grid is the default and
        # behaves identically to the original cartesian enumeration.
        search = build_search_strategy(
            cfg.algo,
            parameter_grid=cfg.parameter_grid,
            n_trials=cfg.n_trials,
            random_seed=cfg.random_seed,
            population_size=cfg.population_size,
            generations=cfg.generations,
            elite_size=cfg.elite_size,
            mutation_rate=cfg.mutation_rate,
        )
        started_at = datetime.now(UTC)
        report = SweepReport(
            sweep_id=cfg.id,
            strategy_id=cfg.strategy_id,
            started_at=started_at,
            finished_at=started_at,  # filled at end
            rank_by=cfg.rank_by,
            total_combos=search.total_trials or 0,
        )
        logger.info(
            "sweep_started sweep_id=%s algo=%s expected_trials=%s strategy=%s",
            cfg.id, cfg.algo, search.total_trials, cfg.strategy_id,
        )

        # Whether the rank metric is smaller-is-better. We invert the score
        # before reporting so adaptive search backends (Optuna, GA) can use
        # "maximize" uniformly.
        minimize = cfg.rank_by in _SMALLER_IS_BETTER

        # Phase 7.C — compile custom-metric expressions once up front so any
        # syntax / disallowed-node error fails fast before any backtest runs.
        compiled_customs = {
            name: compile_metric(expr) for name, expr in cfg.custom_metrics.items()
        }

        i = 0
        while True:
            overrides = search.suggest()
            if overrides is None:
                break
            i += 1
            cell_report = await self._run_cell(cfg, overrides, i)
            metrics = cell_report.to_metrics_dict()
            # Phase 7.C — evaluate each custom metric against the cell's
            # base metrics and merge into the metrics dict. Custom metrics
            # are then visible to rank_by / objectives downstream.
            for name, expr in compiled_customs.items():
                try:
                    metrics[name] = expr.evaluate(metrics)
                except MetricDSLError as e:
                    logger.warning(
                        "custom_metric_eval_failed sweep_id=%s cell=%d name=%s err=%s",
                        cfg.id, i, name, e,
                    )
                    metrics[name] = float("nan")
            raw_score = metrics.get(cfg.rank_by, 0.0)
            # Optuna handles None/nan poorly; coerce to a finite worst-case
            if raw_score is None or (isinstance(raw_score, float) and raw_score != raw_score):
                raw_score = float("-inf") if not minimize else float("inf")
            search.report(overrides, -raw_score if minimize else raw_score)
            report.results.append(
                SweepCellResult(params=overrides, metrics=metrics)
            )
            logger.info(
                "sweep_cell_done sweep_id=%s i=%d params=%s %s=%s",
                cfg.id, i, overrides, cfg.rank_by, raw_score,
            )

        # Now that the search is done we know the actual trial count
        report.total_combos = i
        report.finished_at = datetime.now(UTC)
        # Phase 7.B — compute Pareto frontier when objectives are declared
        if cfg.objectives:
            objectives = [
                Objective(
                    metric=obj["metric"],
                    direction=obj.get("direction", "max"),  # type: ignore[arg-type]
                )
                for obj in cfg.objectives
            ]
            cells = [(r.params, r.metrics) for r in report.results]
            report.pareto = extract_pareto_frontier(cells, objectives)
            logger.info(
                "sweep_pareto sweep_id=%s frontier_size=%d total=%d",
                cfg.id,
                len(report.pareto.frontier),
                len(report.pareto.points),
            )
        await self._persist(cfg, report)
        return report

    # --- Internals -----------------------------------------------------------

    async def _run_cell(
        self,
        cfg: SweepRunConfig,
        overrides: dict[str, Any],
        index: int,
    ) -> BacktestReport:
        merged_params = {**self._base_strategy.params, **overrides}
        # Use a unique id per cell so the per-run output files don't clobber.
        cell_strategy = self._base_strategy.model_copy(
            update={"id": f"{self._base_strategy.id}__sweep_{index}", "params": merged_params}
        )
        # Forward the sweep's full feed list so multi-feed sweeps don't
        # collapse to a single symbol. The explicit `feeds=` shape mirrors
        # whatever the sweep config declared (singular, plural, or feeds).
        run_cfg = BacktestRunConfig(
            id=f"{cfg.id}_cell_{index}",
            mode="file",
            strategy_id=cell_strategy.id,
            feeds=cfg.feed_list,
            start=cfg.start,
            end=cfg.end,
            initial_balance=cfg.initial_balance,
            symbol_contract_sizes=cfg.symbol_contract_sizes,
            slippage_pips=cfg.slippage_pips,
            data_source=cfg.data_source,
        )
        bt = FileBacktester(
            strategy=cell_strategy,
            parquet_root=cfg.data_source,
            sqlite_store=self._sqlite,
            report_dir=self._report_dir / cfg.id,
        )
        return await bt.run(run_cfg)

    async def _persist(self, cfg: SweepRunConfig, report: SweepReport) -> None:
        self._report_dir.mkdir(parents=True, exist_ok=True)
        path = self._report_dir / f"{cfg.id}_summary.json"
        summary_payload: dict[str, Any] = {
            **report.to_summary(),
            "top_n": [
                {"params": r.params, "metrics": r.metrics}
                for r in report.top(cfg.top_n)
            ],
            "all": [
                {"params": r.params, "metrics": r.metrics} for r in report.ranked
            ],
        }
        if report.pareto is not None:
            summary_payload["pareto"] = report.pareto.to_summary()
        path.write_text(json.dumps(summary_payload, indent=2, default=str))
        if self._sqlite is not None:
            from stinger_fx.data.repositories import SweepRepo

            SweepRepo(self._sqlite).record_sweep(cfg.id, cfg.strategy_id, report)
