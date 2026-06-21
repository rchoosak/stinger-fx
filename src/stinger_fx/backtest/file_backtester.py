"""FileBacktester — replays historical bars through the same engine path as live.

The strategy class, runner, bus, and order router are identical to live mode.
Only the broker (`SimBroker`) and clock (`SimClock`) are swapped.
"""

from __future__ import annotations

import asyncio
import heapq
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from stinger_fx.backtest.base import BaseBacktester
from stinger_fx.backtest.playback import PlaybackThrottle
from stinger_fx.backtest.replay_broker import SimBroker
from stinger_fx.backtest.reports import BacktestReport
from stinger_fx.backtest.slippage import build_slippage_model
from stinger_fx.brokers.bar_aggregator import BarAggregator
from stinger_fx.config.models import BacktestRunConfig, RiskConfig, StrategyEntry
from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.core.errors import BacktestError
from stinger_fx.core.event_bus import Subscription as BusSubscription
from stinger_fx.core.events import (
    AccountSnapshotEvent,
    BacktestEquitySampleEvent,
    BarEvent,
    SignalEvent,
    TickEvent,
)
from stinger_fx.data import BacktestRepo, SqliteStore, iter_bars
from stinger_fx.domain import Tick
from stinger_fx.execution import OrderRouter
from stinger_fx.risk import RiskMonitor
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
        bus: AsyncEventBus | None = None,
        risk_config: RiskConfig | None = None,
    ) -> None:
        self._strategy_entry = strategy
        self._parquet_root = parquet_root
        self._sqlite = sqlite_store
        self._report_dir = report_dir or Path("./data/backtests")
        # Optional external bus — when supplied the backtester publishes
        # tick/bar/signal/order/equity events to this shared bus instead of
        # creating an isolated one. Live-backtest mode (web UI) uses this to
        # subscribe SSE listeners to the running replay. CLI standalone path
        # leaves it None and gets the legacy private-bus behavior.
        self._external_bus = bus
        # Optional risk config — when supplied the backtest wires a
        # RiskMonitor (driven by the sim clock) into the OrderRouter so the
        # replay enforces the SAME pre-trade risk gates a live engine would
        # (max positions, daily-loss, kill-switch). None = no risk layer
        # (legacy behavior, used by tests that exercise raw strategy edge).
        self._risk_config = risk_config

    async def run(self, cfg: BacktestRunConfig) -> BacktestReport:
        if cfg.strategy_id != self._strategy_entry.id:
            raise BacktestError(
                f"strategy mismatch: run targets {cfg.strategy_id!r} but configured "
                f"strategy is {self._strategy_entry.id!r}"
            )

        bus = self._external_bus or AsyncEventBus()
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
            symbol_contract_sizes=cfg.symbol_contract_sizes,
            commission_per_lot=cfg.commission_per_lot,
            swap_long_per_lot=cfg.swap_long_per_lot,
            swap_short_per_lot=cfg.swap_short_per_lot,
            swap_rollover_hour_utc=cfg.swap_rollover_hour_utc,
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

        # Optional risk layer — same RiskMonitor the live engine uses, but
        # driven by the sim clock so its daily-loss window rolls on simulated
        # UTC days. Wired into the OrderRouter so rejected signals never
        # become orders, exactly as in live.
        risk: RiskMonitor | None = None
        if self._risk_config is not None:
            risk = RiskMonitor(bus, self._risk_config, clock=sim_clock)
            await risk.start()

        router = OrderRouter(
            bus, broker, strategy_magic={strategy_id: magic}, risk=risk,
        )
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
        if risk is not None:
            await risk.stop()
        # Only close the bus we created. When the caller supplied an external
        # bus (live-backtest mode), it owns the lifecycle — closing it here
        # would tear down the web UI's other subscribers.
        if self._external_bus is None:
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
                    "timeframe": cfg.timeframe.value if cfg.timeframe else None,
                    "start": cfg.start.isoformat(),
                    "end": cfg.end.isoformat(),
                    "initial_balance": cfg.initial_balance,
                    "final_balance": report.final_balance,
                    "symbol_contract_sizes": cfg.symbol_contract_sizes,
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
        # Playback throttle — no-op when cfg.speed == 0 (default).
        throttle = PlaybackThrottle(cfg.speed)
        for bar in merged:
            await throttle.wait_for(bar.time)
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
                mtm += (
                    (ref - p.open_price)
                    * p.side.sign
                    * p.volume
                    * broker.contract_size_for(p.symbol)
                )
            equity_value = broker.balance + mtm
            equity_curve.append((bar.time, equity_value))
            # Live-backtest UIs subscribe to BacktestEquitySampleEvent for the
            # equity curve. Internal-only when no external bus (cheap no-op
            # subscribers).
            await bus.publish(
                BacktestEquitySampleEvent(
                    time=bar.time, balance=broker.balance, equity=equity_value,
                )
            )
            # Feed the RiskMonitor (when wired) so its peak-equity /
            # kill-switch / daily-loss tracking sees the same equity curve a
            # live engine would. Without this the monitor's snapshot-driven
            # rules never fire in a backtest.
            await bus.publish(
                AccountSnapshotEvent(snapshot=await broker.get_account_snapshot())
            )
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
        agg_subs: list[BusSubscription[TickEvent]] = []
        for sub in cfg.feed_list:
            agg = BarAggregator(sub.symbol, sub.timeframe, bus)
            aggregators.append(agg)
            agg_subs.append(
                bus.subscribe(TickEvent, agg.on_tick, name=f"bt.agg.{sub.symbol}.{sub.timeframe.value}")
            )

        # Unique symbols from feed_list — ticks aren't tf-keyed
        symbols = sorted({sub.symbol for sub in cfg.feed_list})

        # Read tick data in UTC-day chunks so live-backtest pages receive
        # the first ticks after loading day one, instead of waiting for the
        # whole date range (a month of XAUUSD can be millions of ticks).
        from stinger_fx.data.parquet_store import ParquetStore as _PS

        store = _PS(self._parquet_root)

        def _gen(sym: str, table):
            for batch in table.to_batches():
                for row in batch.to_pylist():
                    yield Tick(
                        symbol=sym,
                        time=row["time_ns"],
                        bid=row["bid"],
                        ask=row["ask"],
                        last=row.get("last") or 0.0,
                        volume=row.get("volume") or 0,
                        flags=row.get("flags") or 0,
                    )

        last_mid: dict[str, float] = {}
        count = 0
        last_equity_minute: int | None = None
        last_tick: Tick | None = None
        # Playback throttle — no-op when cfg.speed == 0 (default).
        throttle = PlaybackThrottle(cfg.speed)

        cursor = cfg.start.astimezone(UTC)
        end = cfg.end.astimezone(UTC)
        while cursor < end:
            next_midnight = (
                datetime(cursor.year, cursor.month, cursor.day, tzinfo=UTC)
                + timedelta(days=1)
            )
            chunk_end = min(next_midnight, end)

            def _load_table(sym: str, start: datetime, stop: datetime):
                return store.read_ticks(sym, start, stop)

            per_symbol_tables = await asyncio.gather(*[
                asyncio.to_thread(_load_table, sym, cursor, chunk_end)
                for sym in symbols
            ])

            merged = heapq.merge(
                *(
                    _gen(sym, tbl)
                    for sym, tbl in zip(symbols, per_symbol_tables, strict=True)
                ),
                key=lambda t: t.time,
            )

            for tick in merged:
                await throttle.wait_for(tick.time)
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
                        mtm += (
                            (ref - p.open_price)
                            * p.side.sign
                            * p.volume
                            * broker.contract_size_for(p.symbol)
                        )
                    equity_value = broker.balance + mtm
                    equity_curve.append((tick.time, equity_value))
                    # Publish for live-backtest UIs (chart equity line). Rate is
                    # 1/minute of sim time, so it doesn't flood the bus even at
                    # max replay speed.
                    await bus.publish(
                        BacktestEquitySampleEvent(
                            time=tick.time, balance=broker.balance, equity=equity_value,
                        )
                    )
                    # Feed the RiskMonitor (when wired) — see bar-mode note.
                    await bus.publish(
                        AccountSnapshotEvent(
                            snapshot=await broker.get_account_snapshot()
                        )
                    )
                    last_equity_minute = tick_minute
                last_tick = tick
                count += 1

            cursor = chunk_end

        # Ensure at least one equity point on a non-empty replay
        if not equity_curve and last_tick is not None:
            equity_curve.append((last_tick.time, broker.balance))

        for bus_sub in agg_subs:
            await bus_sub.unsubscribe()
        del aggregators  # keep references alive until subs unsub
        return count
