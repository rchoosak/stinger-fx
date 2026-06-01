"""Live-backtest controller — owns the single asyncio Task running a
``FileBacktester`` against the web UI's shared bus.

When the user hits POST `/backtest-live/{run_id}/start?speed=N`, the
endpoint calls ``controller.start(run_id, speed)`` which:

  1. Looks up the run + strategy in the loaded config (must already exist
     in ``config/backtest.yaml``).
  2. Clones ``BacktestRunConfig`` with the requested speed (CLI parity).
  3. Instantiates ``FileBacktester`` with the **shared bus** from the
     ``EngineHandle`` so all subsequent TickEvent/BarEvent/OrderFilledEvent/
     BacktestEquitySampleEvent emissions land on the same bus the SSE
     handler is subscribed to.
  4. Spawns the run as ``asyncio.create_task`` and stores the reference on
     ``self._task`` so ``stop()`` can cancel it and POST `/start` can refuse
     concurrent runs with HTTP 409.

Single-instance by design (Out of Scope: multi-concurrent live backtests).
Cooperative cancellation: the replay loops yield at every event, so the
task winds down within one event when cancelled.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from stinger_fx.backtest.file_backtester import FileBacktester
from stinger_fx.backtest.reports import BacktestReport
from stinger_fx.config.models import BacktestRunConfig, FullConfig
from stinger_fx.data import SqliteStore
from stinger_fx.ui.handle import EngineHandle

logger = logging.getLogger("stinger.ui.web.live_backtest")


class LiveBacktestError(Exception):
    """Raised by the controller for caller-facing failures (HTTP layer maps
    these to 4xx). Internal asyncio errors propagate as-is."""


class LiveBacktestController:
    """Owns the one running live-backtest task, if any.

    Attributes
    ----------
    handle
        Shared ``EngineHandle`` — ``handle.bus`` is the bus the backtester
        publishes into and the SSE endpoint subscribes from.
    cfg
        Parsed ``FullConfig`` so the controller can look up runs and
        strategies by id without re-reading YAML.
    sqlite_store
        Optional — if supplied the backtester writes its summary row into
        SQLite the same way the CLI path does.
    """

    def __init__(
        self,
        handle: EngineHandle,
        cfg: FullConfig,
        sqlite_store: SqliteStore | None = None,
        report_dir: Path | None = None,
    ) -> None:
        self._handle = handle
        self._cfg = cfg
        self._sqlite = sqlite_store
        self._report_dir = report_dir or Path("./data/backtests")
        self._task: asyncio.Task[BacktestReport] | None = None
        self._run_id: str | None = None
        self._speed: float | None = None

    # --- public API -------------------------------------------------------

    def is_running(self) -> bool:
        """True while a backtest task is alive (not done / not cancelled)."""
        return self._task is not None and not self._task.done()

    @property
    def status(self) -> dict[str, object]:
        """Cheap snapshot for the UI poller / page footer."""
        return {
            "running": self.is_running(),
            "run_id": self._run_id if self.is_running() else None,
            "speed": self._speed if self.is_running() else None,
        }

    async def start(self, run_id: str, speed: float) -> None:
        """Spawn the backtester task. Raises ``LiveBacktestError`` on
        concurrent-run conflict or missing config; lets unexpected errors
        propagate (caught + logged in the asyncio task)."""
        if self.is_running():
            raise LiveBacktestError(
                f"a live backtest is already running (run_id={self._run_id!r})"
            )
        if speed <= 0:
            # Live mode requires throttling — speed=0 (max speed) would
            # flood the SSE queue and starve the browser.
            raise LiveBacktestError(
                "live backtest requires speed > 0 (max speed is unviewable)"
            )

        run = next((r for r in self._cfg.backtest.runs if r.id == run_id), None)
        if run is None:
            raise LiveBacktestError(f"no backtest run with id {run_id!r}")
        if run.mode != "file":
            raise LiveBacktestError(
                f"live backtest only supports mode='file'; run {run_id!r} is "
                f"mode={run.mode!r}"
            )
        strategy = next(
            (s for s in self._cfg.strategies.strategies if s.id == run.strategy_id),
            None,
        )
        if strategy is None:
            raise LiveBacktestError(
                f"backtest run {run_id!r} references unknown strategy "
                f"{run.strategy_id!r}"
            )
        if run.data_source is None:
            raise LiveBacktestError(
                f"backtest run {run_id!r} has no data_source — required for "
                f"file mode"
            )

        # CLI parity: --speed overrides the YAML field.
        run_cfg: BacktestRunConfig = run.model_copy(update={"speed": speed})

        backtester = FileBacktester(
            strategy=strategy,
            parquet_root=run_cfg.data_source,  # type: ignore[arg-type]
            sqlite_store=self._sqlite,
            report_dir=self._report_dir,
            bus=self._handle.bus,   # the critical wire — shared with SSE
        )

        async def _wrapped() -> BacktestReport:
            logger.info("live backtest starting run_id=%s speed=%s", run_id, speed)
            try:
                report = await backtester.run(run_cfg)
                logger.info(
                    "live backtest finished run_id=%s trades=%d",
                    run_id, len(report.trades),
                )
                return report
            except asyncio.CancelledError:
                logger.info("live backtest cancelled run_id=%s", run_id)
                raise
            except BaseException as e:
                # Use BaseException so we also surface things like SystemExit
                # or unexpected pydantic ValidationError that propagate before
                # the normal Exception path. logger.exception keeps the traceback.
                logger.exception("live backtest failed run_id=%s err=%r", run_id, e)
                raise

        self._task = asyncio.create_task(_wrapped(), name=f"live-bt:{run_id}")
        self._run_id = run_id
        self._speed = speed

    async def stop(self) -> None:
        """Cancel the running task and wait for clean teardown. No-op if
        nothing is running."""
        if self._task is None:
            return
        if not self._task.done():
            self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        except Exception:
            # Already logged in _wrapped(); swallow here so stop() never raises.
            pass
        self._task = None
        self._run_id = None
        self._speed = None

    async def shutdown(self) -> None:
        """Server-shutdown hook — alias for ``stop()`` to make wiring obvious."""
        await self.stop()
