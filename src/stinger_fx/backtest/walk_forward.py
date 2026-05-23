"""Walk-forward optimisation — time-sliced sweeps with out-of-sample evaluation.

For each of N folds:
  1. Run a parameter sweep on the **in-sample** window → pick best params by
     `rank_by` (inherited from the sweep, or overridden in the WF config).
  2. Run a single backtest on the **out-of-sample** window using those params.
  3. Record both the in-sample metric and the out-of-sample metric for the fold.

The output report has one row per fold; comparing in-sample vs out-of-sample
metrics tells you whether the optimisation is overfitting.

Two schemes:

  • **expanding** (default) — in-sample grows over time, out-of-sample is the
    immediately-following segment. Best for stationary strategies with rare
    regime shifts.

  • **rolling** — in-sample is a fixed window that slides forward; same
    out-of-sample placement. Best for non-stationary strategies that should
    only "remember" the recent past.

Diagram (`folds=3`, `expanding`):

    [seg0][seg1][seg2][seg3]   (segments of equal duration)
    Fold 1 in=[seg0]      out=[seg1]
    Fold 2 in=[seg0..1]   out=[seg2]
    Fold 3 in=[seg0..2]   out=[seg3]
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from stinger_fx.backtest.file_backtester import FileBacktester
from stinger_fx.backtest.sweep import ParameterSweep
from stinger_fx.config.models import (
    BacktestRunConfig,
    MetricName,
    StrategyEntry,
    SweepRunConfig,
    WalkForwardConfig,
)
from stinger_fx.data import SqliteStore

logger = logging.getLogger("stinger.backtest.walk_forward")


@dataclass
class WalkForwardFold:
    fold: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_of_sample_start: datetime
    out_of_sample_end: datetime
    best_params: dict[str, Any]
    in_sample_metric: float | None
    out_of_sample_metrics: dict[str, float]


@dataclass
class WalkForwardReport:
    wf_id: str
    strategy_id: str
    sweep_id: str
    rank_by: MetricName
    scheme: str
    folds: list[WalkForwardFold] = field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def to_summary(self) -> dict:
        return {
            "wf_id": self.wf_id,
            "strategy_id": self.strategy_id,
            "sweep_id": self.sweep_id,
            "rank_by": self.rank_by,
            "scheme": self.scheme,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "folds": [
                {
                    "fold": f.fold,
                    "in_sample": [f.in_sample_start.isoformat(), f.in_sample_end.isoformat()],
                    "out_of_sample": [
                        f.out_of_sample_start.isoformat(),
                        f.out_of_sample_end.isoformat(),
                    ],
                    "best_params": f.best_params,
                    "in_sample_metric": f.in_sample_metric,
                    "out_of_sample_metrics": f.out_of_sample_metrics,
                }
                for f in self.folds
            ],
        }


def slice_folds(
    start: datetime,
    end: datetime,
    folds: int,
    *,
    scheme: str = "expanding",
    in_sample_ratio: float = 0.7,
) -> list[tuple[datetime, datetime, datetime, datetime]]:
    """Returns one (in_start, in_end, oos_start, oos_end) tuple per fold.

    For `expanding`: split [start, end] into `folds + 1` equal segments.
    Fold k (1..N) has in_sample = [start, seg_k_end] and out-of-sample =
    [seg_k_end, seg_{k+1}_end].

    For `rolling`: each fold has a fixed-size in-sample window of length
    `in_sample_ratio * total / folds` immediately preceding the OOS window.
    """
    if folds < 2:
        raise ValueError("folds must be >= 2")
    if end <= start:
        raise ValueError("end must be > start")
    total = end - start
    out: list[tuple[datetime, datetime, datetime, datetime]] = []
    if scheme == "expanding":
        seg_count = folds + 1
        seg = total / seg_count
        for k in range(1, folds + 1):
            in_end = start + seg * k
            oos_end = start + seg * (k + 1)
            out.append((start, in_end, in_end, oos_end))
    elif scheme == "rolling":
        # Walk OOS windows across folds (equal-sized) and require each in-sample
        # window to be `in_sample_ratio * fold_duration_in_sample`-equivalent
        # immediately before it.
        oos_total = total * (1.0 - in_sample_ratio)
        oos_seg = oos_total / folds
        in_seg = total - oos_total  # the in-sample lookback length (fixed)
        # Pin the first OOS window so it starts after the first in-sample lookback
        cursor = start + in_seg
        for k in range(1, folds + 1):
            in_start = cursor - in_seg
            in_end = cursor
            oos_start = cursor
            oos_end = cursor + oos_seg
            out.append((in_start, in_end, oos_start, oos_end))
            cursor = oos_end
    else:
        raise ValueError(f"unknown scheme: {scheme!r}")
    return out


class WalkForward:
    """Walk-forward runner — drives ParameterSweep + FileBacktester per fold."""

    def __init__(
        self,
        *,
        base_strategy: StrategyEntry,
        sqlite_store: SqliteStore | None = None,
        report_dir: Path | None = None,
    ) -> None:
        self._base_strategy = base_strategy
        self._sqlite = sqlite_store
        self._report_dir = report_dir or Path("./data/walk_forwards")

    async def run(
        self,
        wf_cfg: WalkForwardConfig,
        sweep_cfg: SweepRunConfig,
    ) -> WalkForwardReport:
        rank_by: MetricName = wf_cfg.rank_by or sweep_cfg.rank_by
        report = WalkForwardReport(
            wf_id=wf_cfg.id,
            strategy_id=self._base_strategy.id,
            sweep_id=sweep_cfg.id,
            rank_by=rank_by,
            scheme=wf_cfg.scheme,
            started_at=datetime.now(UTC),
        )
        slices = slice_folds(
            sweep_cfg.start,
            sweep_cfg.end,
            wf_cfg.folds,
            scheme=wf_cfg.scheme,
            in_sample_ratio=wf_cfg.in_sample_ratio,
        )
        logger.info(
            "walk_forward_started wf_id=%s folds=%d scheme=%s rank_by=%s",
            wf_cfg.id, wf_cfg.folds, wf_cfg.scheme, rank_by,
        )

        for k, (in_start, in_end, oos_start, oos_end) in enumerate(slices, start=1):
            best_params, in_sample_metric = await self._optimize_in_sample(
                k, sweep_cfg, in_start, in_end, rank_by
            )
            oos_metrics = await self._evaluate_out_of_sample(
                k, sweep_cfg, oos_start, oos_end, best_params
            )
            report.folds.append(
                WalkForwardFold(
                    fold=k,
                    in_sample_start=in_start,
                    in_sample_end=in_end,
                    out_of_sample_start=oos_start,
                    out_of_sample_end=oos_end,
                    best_params=best_params,
                    in_sample_metric=in_sample_metric,
                    out_of_sample_metrics=oos_metrics,
                )
            )
            logger.info(
                "walk_forward_fold_done fold=%d/%d best_params=%s in_sample_%s=%s oos_%s=%s",
                k, wf_cfg.folds, best_params, rank_by, in_sample_metric,
                rank_by, oos_metrics.get(rank_by),
            )

        report.finished_at = datetime.now(UTC)
        await self._persist(wf_cfg, report)
        return report

    # --- Internals ----------------------------------------------------------

    async def _optimize_in_sample(
        self,
        fold: int,
        sweep_cfg: SweepRunConfig,
        in_start: datetime,
        in_end: datetime,
        rank_by: MetricName,
    ) -> tuple[dict[str, Any], float | None]:
        """Run ParameterSweep on the in-sample window, return best params + metric."""
        fold_sweep = sweep_cfg.model_copy(
            update={
                "id": f"{sweep_cfg.id}__wf_fold{fold}_is",
                "start": in_start,
                "end": in_end,
                "rank_by": rank_by,
            }
        )
        sweep = ParameterSweep(
            base_strategy=self._base_strategy,
            sqlite_store=self._sqlite,
            report_dir=self._report_dir / fold_sweep.id,
        )
        report = await sweep.run(fold_sweep)
        best = report.best()
        if best is None:
            return ({}, None)
        return (best.params, best.metrics.get(rank_by))

    async def _evaluate_out_of_sample(
        self,
        fold: int,
        sweep_cfg: SweepRunConfig,
        oos_start: datetime,
        oos_end: datetime,
        best_params: dict[str, Any],
    ) -> dict[str, float]:
        """Backtest a single param combo on the out-of-sample window."""
        merged_params = {**self._base_strategy.params, **best_params}
        cell_strategy = self._base_strategy.model_copy(
            update={
                "id": f"{self._base_strategy.id}__wf_fold{fold}_oos",
                "params": merged_params,
            }
        )
        run_cfg = BacktestRunConfig(
            id=f"{sweep_cfg.id}__wf_fold{fold}_oos",
            mode="file",
            strategy_id=cell_strategy.id,
            symbol=sweep_cfg.symbol,
            timeframe=sweep_cfg.timeframe,
            start=oos_start,
            end=oos_end,
            initial_balance=sweep_cfg.initial_balance,
            slippage_pips=sweep_cfg.slippage_pips,
            data_source=sweep_cfg.data_source,
        )
        bt = FileBacktester(
            strategy=cell_strategy,
            parquet_root=sweep_cfg.data_source,
            sqlite_store=self._sqlite,
            report_dir=self._report_dir / f"{sweep_cfg.id}__wf_fold{fold}_oos",
        )
        result = await bt.run(run_cfg)
        return result.to_metrics_dict()

    async def _persist(self, wf_cfg: WalkForwardConfig, report: WalkForwardReport) -> None:
        self._report_dir.mkdir(parents=True, exist_ok=True)
        path = self._report_dir / f"{wf_cfg.id}_summary.json"
        path.write_text(json.dumps(report.to_summary(), indent=2, default=str))
        # NB: dedicated SQLite table for walk-forwards is Phase 3 follow-up;
        # for now the per-fold backtest_runs + sweep_runs rows already give
        # full auditability through their ids.
