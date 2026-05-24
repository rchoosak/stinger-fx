"""FileBacktester — replays historical bars through the same engine path as live.

The strategy class, runner, bus, and order router are identical to live mode.
Only the broker (`SimBroker`) and clock (`SimClock`) are swapped.
"""

from __future__ import annotations

import asyncio
import heapq
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from stinger_fx.backtest.base import BaseBacktester
from stinger_fx.backtest.order_router import OrderRouter
from stinger_fx.backtest.replay_broker import SimBroker
from stinger_fx.backtest.reports import BacktestReport
from stinger_fx.config.models import BacktestRunConfig, StrategyEntry
from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.core.errors import BacktestError
from stinger_fx.core.events import BarEvent, SignalEvent
from stinger_fx.data import BacktestRepo, SqliteStore, iter_bars
from stinger_fx.strategies import (
    StrategyRunner,
    derive_magic,
    load_strategy_class,
    validate_params,
)

logger = logging.getLogger("stinger.backtest.file")


class FileBacktester(BaseBacktester):
    name = "file"

    def __init__(
        self,
        *,
        strategy: StrategyEntry,
        parquet_root: Path,
        sqlite_store: SqliteStore | None = None,
        report_dir: Path | None = None,
    ) -> None:
        self._strategy_entry = strategy
        self._parquet_root = parquet_root
        self._sqlite = sqlite_store
        self._report_dir = report_dir or Path("./data/backtests")

    async def run(self, cfg: BacktestRunConfig) -> BacktestReport:
        if cfg.strategy_id != self._strategy_entry.id:
            raise BacktestError(
                f"strategy mismatch: run targets {cfg.strategy_id!r} but configured "
                f"strategy is {self._strategy_entry.id!r}"
            )

        bus = AsyncEventBus()
        sim_clock = SimClock(cfg.start)
        broker = SimBroker(bus, initial_balance=cfg.initial_balance, slippage_pips=cfg.slippage_pips)

        strategy_cls = load_strategy_class(self._strategy_entry.class_path)
        # Allow the run config to override symbol/timeframe for sweep cells —
        # but only when the strategy's Params model actually declares those
        # fields. Multi-feed strategies (Phase 4) often don't, since the
        # backtester drives multiple symbols from cfg.feed_list instead.
        params_dict = dict(self._strategy_entry.params)
        param_fields = set(strategy_cls.Params.model_fields.keys())
        if "symbol" in param_fields and cfg.symbol is not None:
            params_dict.setdefault("symbol", cfg.symbol)
        if "timeframe" in param_fields and cfg.timeframe is not None:
            params_dict.setdefault("timeframe", cfg.timeframe.value)
        params = validate_params(strategy_cls, params_dict)

        strategy_id = self._strategy_entry.id
        magic = derive_magic(strategy_id)

        router = OrderRouter(bus, broker, strategy_magic={strategy_id: magic})
        await router.attach()

        async def signal_sink(sig):
            await bus.publish(SignalEvent(signal=sig))

        runner = StrategyRunner(
            strategy_id=strategy_id,
            strategy=strategy_cls(),
            params=params,
            bus=bus,
            clock=sim_clock,
            reload_lock=asyncio.Lock(),
            signal_sink=signal_sink,
        )
        await runner.start()

        repo_row_id: int | None = None
        if self._sqlite is not None:
            self._sqlite.create_all()
            repo_row_id = BacktestRepo(self._sqlite).start_run(
                run_id=cfg.id,
                strategy_id=strategy_id,
                params=params.model_dump(mode="json"),
            )

        started_at = datetime.now(UTC)
        equity_curve: list[tuple[datetime, float]] = []
        bar_count = 0

        # Multi-feed: merge per-feed bar iterators in chronological order.
        # Each iter_bars() already yields ascending by time → heapq.merge is
        # O(rows · log N), one bar at a time, with O(N) memory. For ties the
        # feed_list ordering ((symbol, tf.value)) breaks them deterministically.
        feed_iters = [
            iter_bars(self._parquet_root, sub.symbol, sub.timeframe, cfg.start, cfg.end)
            for sub in cfg.feed_list
        ]
        merged = heapq.merge(*feed_iters, key=lambda b: b.time)

        # Track last-known close per symbol for the MTM mark across all
        # open positions on the current event.
        last_close: dict[str, float] = {}

        for bar in merged:
            sim_clock.advance(bar.time)
            broker.advance_clock(bar.time)
            broker.set_market(bar.symbol, bar.close)
            last_close[bar.symbol] = bar.close

            # Stop-loss / take-profit check using this bar's high/low
            for pos in broker.check_sl_tp(bar.symbol, bar.high, bar.low):
                await broker.close_position(pos.ticket)

            await bus.publish(BarEvent(bar=bar))
            # Let queues drain so the strategy reacts before we move to the next bar.
            for _ in range(3):
                await asyncio.sleep(0)

            # Mark-to-market equity across ALL open positions, valuing each
            # position at the last known close for its symbol. Positions whose
            # symbol hasn't received a bar yet contribute 0 (they would be
            # opened at the first bar so this only matters at warm-up).
            mtm = 0.0
            for p in await broker.get_positions():
                ref = last_close.get(p.symbol)
                if ref is None:
                    continue
                mtm += (ref - p.open_price) * p.side.sign * p.volume * 100_000.0
            equity_curve.append((bar.time, broker.balance + mtm))
            bar_count += 1

        # Close any remaining positions at the last bar's close
        for pos in list(await broker.get_positions()):
            await broker.close_position(pos.ticket)

        await runner.stop()
        await router.detach()
        await bus.close()

        finished_at = datetime.now(UTC)
        report = BacktestReport(
            run_id=cfg.id,
            strategy_id=strategy_id,
            started_at=started_at,
            finished_at=finished_at,
            trades=broker.trades,
            equity_curve=equity_curve,
            initial_balance=cfg.initial_balance,
            final_balance=broker.balance,
        )

        # Persist
        self._report_dir.mkdir(parents=True, exist_ok=True)
        equity_path = self._report_dir / f"{cfg.id}_equity.parquet"
        metrics_path = self._report_dir / f"{cfg.id}_metrics.json"
        trades_path = self._report_dir / f"{cfg.id}_trades.json"
        report.write_equity_curve(equity_path)
        metrics_path.write_text(json.dumps(report.to_metrics_dict(), indent=2))
        # Trade-replay sidecar — used by the Web UI's /backtest/{run_id} view
        # to overlay entry/exit markers on the equity curve. Also embeds the
        # run config so the view doesn't need to re-derive it.
        trades_path.write_text(
            json.dumps(
                {
                    "run_id": cfg.id,
                    "strategy_id": strategy_id,
                    "symbol": cfg.symbol,
                    "timeframe": cfg.timeframe.value,
                    "start": cfg.start.isoformat(),
                    "end": cfg.end.isoformat(),
                    "initial_balance": cfg.initial_balance,
                    "final_balance": report.final_balance,
                    "trades": report.trades_to_jsonable(),
                },
                indent=2,
                default=str,
            )
        )

        if self._sqlite is not None and repo_row_id is not None:
            BacktestRepo(self._sqlite).finish_run(
                repo_row_id, report.to_metrics_dict(), str(equity_path)
            )

        logger.info("backtest done bars=%s trades=%s", bar_count, len(report.trades))
        return report
