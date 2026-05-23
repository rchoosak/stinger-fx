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
from stinger_fx.backtest.reports import BacktestReport
from stinger_fx.config.models import (
    BacktestRunConfig,
    MetricName,
    StrategyEntry,
    SweepRunConfig,
)
from stinger_fx.data import SqliteStore
from stinger_fx.domain import Timeframe

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
    rank_by: MetricName
    total_combos: int
    results: list[SweepCellResult] = field(default_factory=list)

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
        combos = enumerate_grid(cfg.parameter_grid)
        started_at = datetime.now(UTC)
        report = SweepReport(
            sweep_id=cfg.id,
            strategy_id=cfg.strategy_id,
            started_at=started_at,
            finished_at=started_at,  # filled at end
            rank_by=cfg.rank_by,
            total_combos=len(combos),
        )
        logger.info(
            "sweep_started sweep_id=%s combos=%d strategy=%s",
            cfg.id,
            len(combos),
            cfg.strategy_id,
        )

        for i, overrides in enumerate(combos, start=1):
            cell_report = await self._run_cell(cfg, overrides, i)
            report.results.append(
                SweepCellResult(
                    params=overrides,
                    metrics=cell_report.to_metrics_dict(),
                )
            )
            logger.info(
                "sweep_cell_done sweep_id=%s i=%d/%d params=%s %s=%s",
                cfg.id, i, len(combos), overrides, cfg.rank_by,
                cell_report.to_metrics_dict().get(cfg.rank_by),
            )

        report.finished_at = datetime.now(UTC)
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
        run_cfg = BacktestRunConfig(
            id=f"{cfg.id}_cell_{index}",
            mode="file",
            strategy_id=cell_strategy.id,
            symbol=cfg.symbol,
            timeframe=cfg.timeframe,
            start=cfg.start,
            end=cfg.end,
            initial_balance=cfg.initial_balance,
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
        path.write_text(
            json.dumps(
                {
                    **report.to_summary(),
                    "top_n": [
                        {"params": r.params, "metrics": r.metrics}
                        for r in report.top(cfg.top_n)
                    ],
                    "all": [
                        {"params": r.params, "metrics": r.metrics} for r in report.ranked
                    ],
                },
                indent=2,
                default=str,
            )
        )
        if self._sqlite is not None:
            from stinger_fx.data.repositories import SweepRepo

            SweepRepo(self._sqlite).record_sweep(cfg.id, cfg.strategy_id, report)
