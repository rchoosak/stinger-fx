"""DriftMonitor — alert when a strategy's recent *live* performance falls
materially below its *backtest* baseline.

On every live close it recomputes the owning strategy's recent win-rate and
expectancy from the persisted `trades` table and compares them to the latest
backtest run's metrics (`backtest_runs`). When live degrades past the configured
floor it publishes a `StrategyDriftEvent` (logged, and routable to notification
sinks) — alert only, it never stops the strategy.

Hysteresis: one alert per degradation episode; it re-arms once the strategy
recovers, so a sustained drift doesn't fire on every close. Best-effort — a DB
or parse error is logged and never breaks trading.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from stinger_fx.config.models import DriftMonitorConfig
from stinger_fx.core.event_bus import AsyncEventBus, Subscription
from stinger_fx.core.events import PositionClosedEvent, StrategyDriftEvent
from stinger_fx.data.repositories import BacktestRepo, TradeRepo
from stinger_fx.data.sqlite_store import SqliteStore

logger = logging.getLogger("stinger.observability.drift")


class DriftMonitor:
    def __init__(
        self,
        bus: AsyncEventBus,
        store: SqliteStore,
        *,
        strategy_for_magic: Callable[[int], str | None],
        cfg: DriftMonitorConfig,
    ) -> None:
        self._bus = bus
        self._trades = TradeRepo(store)
        self._backtests = BacktestRepo(store)
        self._strategy_for_magic = strategy_for_magic
        self._cfg = cfg
        self._sub: Subscription | None = None
        # strategy_id → parsed backtest metrics dict. Only successful loads are
        # cached; a missing baseline is re-queried so a backtest run mid-session
        # is picked up.
        self._baseline_cache: dict[str, dict] = {}
        # strategy_id → currently-alerted (hysteresis re-arm flag).
        self._alerted: dict[str, bool] = {}

    async def start(self) -> None:
        self._sub = self._bus.subscribe(
            PositionClosedEvent, self._on_closed, name="drift_monitor.close"
        )
        logger.info("drift_monitor_started")

    async def stop(self) -> None:
        if self._sub is not None:
            await self._sub.unsubscribe()
            self._sub = None

    def _baseline(self, strategy_id: str) -> dict | None:
        cached = self._baseline_cache.get(strategy_id)
        if cached is not None:
            return cached
        metrics = self._backtests.latest_metrics_for(strategy_id)
        if metrics is None:
            return None
        self._baseline_cache[strategy_id] = metrics
        return metrics

    async def _on_closed(self, evt: PositionClosedEvent) -> None:
        try:
            sid = self._strategy_for_magic(evt.position.magic)
            if sid is None:
                return
            trades = self._trades.recent_trades_for(sid, self._cfg.window)
            n = len(trades)
            if n < self._cfg.min_trades:
                return
            base = self._baseline(sid)
            if base is None:
                return
            base_wr = float(base.get("win_rate", 0.0))
            # Per-lot so the comparison is size-invariant (live position size
            # often differs from the backtest baseline, e.g. risk-based sizing).
            base_exp = base.get("expectancy_per_lot")

            live_wr = sum(1 for pnl, _ in trades if pnl > 0) / n
            per_lot = [pnl / vol for pnl, vol in trades if vol > 0]
            live_exp = sum(per_lot) / len(per_lot) if per_lot else 0.0

            reasons: list[str] = []
            if base_wr > 0 and live_wr < base_wr * self._cfg.min_win_rate_ratio:
                reasons.append(
                    f"win-rate {live_wr:.2f} < floor "
                    f"{base_wr * self._cfg.min_win_rate_ratio:.2f} (baseline {base_wr:.2f})"
                )
            # expectancy_per_lot is absent from baselines produced before this
            # metric existed — skip the expectancy check rather than guess.
            if (
                base_exp is not None
                and float(base_exp) > 0
                and per_lot
                and live_exp < float(base_exp) * self._cfg.min_expectancy_ratio
            ):
                reasons.append(
                    f"expectancy/lot {live_exp:.2f} < floor "
                    f"{float(base_exp) * self._cfg.min_expectancy_ratio:.2f} "
                    f"(baseline {float(base_exp):.2f})"
                )
            degraded = bool(reasons)

            if degraded and not self._alerted.get(sid, False):
                reason = "; ".join(reasons)
                logger.warning("strategy_drift strategy=%s n=%d %s", sid, n, reason)
                await self._bus.publish(
                    StrategyDriftEvent(
                        strategy_id=sid,
                        sample_size=n,
                        live_win_rate=live_wr,
                        live_expectancy_per_lot=live_exp,
                        baseline_win_rate=base_wr,
                        baseline_expectancy_per_lot=(
                            float(base_exp) if base_exp is not None else 0.0
                        ),
                        reason=reason,
                    )
                )
                self._alerted[sid] = True
            elif not degraded:
                self._alerted[sid] = False
        except Exception:
            logger.exception("drift_monitor_check_failed")
