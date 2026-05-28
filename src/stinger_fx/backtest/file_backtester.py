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
from stinger_fx.backtest.slippage import build_slippage_model
from stinger_fx.brokers.bar_aggregator import BarAggregator
from stinger_fx.config.models import BacktestRunConfig, StrategyEntry
from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.core.errors import BacktestError
from stinger_fx.core.events import BarEvent, SignalEvent, TickEvent
from stinger_fx.data import BacktestRepo, SqliteStore, iter_bars, iter_ticks
from stinger_fx.domain import Tick
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
        slippage_fn = build_slippage_model(
            cfg.slippage_model,
            pips=cfg.slippage_pips,
            volatility_factor=cfg.slippage_volatility_factor,
        )
        broker = SimBroker(
            bus,
            initial_balance=cfg.initial_balance,
            slippage_pips=cfg.slippage_pips,
            slippage_fn=slippage_fn,
        )

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

        if cfg.granularity == "tick":
            event_count = await self._replay_ticks(cfg, bus, broker, sim_clock, equity_curve)
        else:
            event_count = await self._replay_bars(cfg, bus, broker, sim_clock, equity_curve)
        bar_count = event_count

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

        logger.info(
            "backtest done granularity=%s events=%s trades=%s",
            cfg.granularity, bar_count, len(report.trades),
        )
        return report

    # --- Replay paths --------------------------------------------------------

    async def _replay_bars(
        self,
        cfg: BacktestRunConfig,
        bus: AsyncEventBus,
        broker: SimBroker,
        sim_clock: SimClock,
        equity_curve: list[tuple[datetime, float]],
    ) -> int:
        """Bar-mode replay (legacy). Same logic as before Batch B; merges
        per-feed iter_bars() in chronological order."""
        feed_iters = [
            iter_bars(self._parquet_root, sub.symbol, sub.timeframe, cfg.start, cfg.end)
            for sub in cfg.feed_list
        ]
        merged = heapq.merge(*feed_iters, key=lambda b: b.time)
        last_close: dict[str, float] = {}
        count = 0
        for bar in merged:
            sim_clock.advance(bar.time)
            broker.advance_clock(bar.time)
            broker.set_market(bar.symbol, bar.close)
            last_close[bar.symbol] = bar.close
            for pos in broker.check_sl_tp(bar.symbol, bar.high, bar.low):
                await broker.close_position(pos.ticket)
            # Phase 6.2.A — pending STOP / LIMIT orders triggered by the
            # bar's price range. Pre-fix bar-mode skipped this entirely
            # (only tick-mode called check_pending), so pendings sat in
            # the broker forever and breakout / pullback strategies
            # showed zero trades in bar backtests.
            await broker.check_pending_bar(bar.symbol, bar.high, bar.low)
            await bus.publish(BarEvent(bar=bar))
            for _ in range(3):
                await asyncio.sleep(0)
            mtm = 0.0
            for p in await broker.get_positions():
                ref = last_close.get(p.symbol)
                if ref is None:
                    continue
                mtm += (ref - p.open_price) * p.side.sign * p.volume * 100_000.0
            equity_curve.append((bar.time, broker.balance + mtm))
            count += 1
        return count

    async def _replay_ticks(
        self,
        cfg: BacktestRunConfig,
        bus: AsyncEventBus,
        broker: SimBroker,
        sim_clock: SimClock,
        equity_curve: list[tuple[datetime, float]],
    ) -> int:
        """Tick-mode replay (Phase 4 A.1).

        Reads ticks per-symbol (timeframe is ignored for the data source,
        since ticks aren't keyed by tf), merges chronologically, publishes
        TickEvent. BarAggregators per (symbol, tf) listen on the bus and
        synthesise BarEvent so strategies keep getting on_bar exactly as
        in bar mode. Equity is sampled once per minute-boundary to keep
        the curve compact.
        """
        # Build per-feed aggregators so the strategy gets BarEvent for each
        # (symbol, tf) it subscribed to. Each one subscribes to TickEvent on
        # the bus and self-filters by symbol.
        aggregators: list[BarAggregator] = []
        agg_subs = []
        for sub in cfg.feed_list:
            agg = BarAggregator(sub.symbol, sub.timeframe, bus)
            aggregators.append(agg)
            agg_subs.append(
                bus.subscribe(TickEvent, agg.on_tick, name=f"bt.agg.{sub.symbol}.{sub.timeframe.value}")
            )

        # Unique symbols from feed_list — ticks aren't tf-keyed
        symbols = sorted({sub.symbol for sub in cfg.feed_list})
        tick_iters = [
            iter_ticks(self._parquet_root, sym, cfg.start, cfg.end)
            for sym in symbols
        ]
        merged = heapq.merge(*tick_iters, key=lambda t: t.time)

        last_mid: dict[str, float] = {}
        count = 0
        last_equity_minute: int | None = None
        last_tick: Tick | None = None
        for tick in merged:
            sim_clock.advance(tick.time)
            broker.advance_clock(tick.time)
            # Store both bid AND ask so spread/volatility slippage models have
            # accurate data; long P&L marks at bid (consistent with bar mode).
            broker.set_market_tick(tick.symbol, tick.bid, tick.ask)
            last_mid[tick.symbol] = (tick.bid + tick.ask) / 2.0

            # Tick-precise SL/TP — fires before strategy sees the event so
            # the strategy can't try to act on a position that's about to close.
            for pos in broker.check_sl_tp_tick(tick.symbol, tick.bid, tick.ask):
                await broker.close_position(pos.ticket)

            # Phase 6.2.A — pending orders (BUY/SELL STOP & LIMIT). Triggered
            # orders are promoted to positions and emit OrderFilledEvent —
            # the strategy sees them via its on_order_filled hook.
            await broker.check_pending(tick.symbol, tick.bid, tick.ask)

            await bus.publish(TickEvent(tick=tick))
            # Drain — fewer than bar mode because per-tick strategy work is
            # usually a no-op until the aggregator publishes a closed bar.
            for _ in range(2):
                await asyncio.sleep(0)

            # Sample equity once per minute boundary to keep the curve compact
            tick_minute = int(tick.time.timestamp()) // 60
            if last_equity_minute is None or tick_minute > last_equity_minute:
                mtm = 0.0
                for p in await broker.get_positions():
                    ref = last_mid.get(p.symbol)
                    if ref is None:
                        continue
                    mtm += (ref - p.open_price) * p.side.sign * p.volume * 100_000.0
                equity_curve.append((tick.time, broker.balance + mtm))
                last_equity_minute = tick_minute
            last_tick = tick
            count += 1

        # Ensure at least one equity point on a non-empty replay
        if not equity_curve and last_tick is not None:
            equity_curve.append((last_tick.time, broker.balance))

        for sub in agg_subs:
            await sub.unsubscribe()
        del aggregators  # keep references alive until subs unsub
        return count
