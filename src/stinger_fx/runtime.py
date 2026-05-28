"""Engine assembler — wires the engine and all its subcomponents from config.

The CLI calls `assemble_and_run(...)` to start the platform. Backtest commands
build their own narrower stacks and don't go through here.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path

from stinger_fx.backtest.order_router import OrderRouter
from stinger_fx.brokers import BrokerPool, build_broker
from stinger_fx.brokers.bar_aggregator import BarAggregator
from stinger_fx.config import (
    ConfigReloader,
    ConfigWatcher,
    FullConfig,
    ReloadActions,
    StrategyEntry,
    load_all,
)
from stinger_fx.core import AsyncEventBus, LiveClock, TradingEngine
from stinger_fx.core.errors import ConfigError
from stinger_fx.core.events import (
    AccountSnapshotEvent,
    ConfigReloadedEvent,
    ConfigReloadFailedEvent,
    SignalEvent,
    TickEvent,
)
from stinger_fx.data import SqliteStore
from stinger_fx.domain.timeframes import Timeframe
from stinger_fx.log import configure as configure_logs
from stinger_fx.log import set_level
from stinger_fx.risk import RiskMonitor
from stinger_fx.strategies import (
    StrategyRunner,
    derive_magic,
    load_strategy_class,
    validate_params,
)
from stinger_fx.ui.handle import EngineHandle
from stinger_fx.ui.normal import NormalUI

logger = logging.getLogger("stinger.runtime")


class StingerApp:
    """Owns all live-mode singletons. Lifetime = one CLI `run` invocation."""

    def __init__(self, config_dir: Path, *, mode_override: str | None = None) -> None:
        self.config_dir = config_dir
        self._mode_override = mode_override
        self.full_cfg: FullConfig | None = None
        self.engine: TradingEngine | None = None
        self.bus: AsyncEventBus | None = None
        self.handle: EngineHandle | None = None
        self.sqlite: SqliteStore | None = None
        self.runners: dict[str, StrategyRunner] = {}
        self.aggregators: dict[tuple[str, Timeframe], BarAggregator] = {}
        self._watcher: ConfigWatcher | None = None
        self._reloader: ConfigReloader | None = None
        self._router: OrderRouter | None = None
        self._ui: NormalUI | None = None
        self._pool: BrokerPool = BrokerPool()
        self._strategy_accounts: dict[str, str] = {}
        self._risk: RiskMonitor | None = None
        self._notifications: object | None = None
        self._mode: str = "normal"
        self._web_host: str = "127.0.0.1"
        self._web_port: int = 8765
        self._metrics: object | None = None  # MetricsCollector when metrics.enabled

    async def setup(self) -> None:
        self.full_cfg = load_all(self.config_dir)
        app_cfg = self.full_cfg.app
        if self._mode_override is not None:
            app_cfg = app_cfg.model_copy(update={"mode": self._mode_override})

        configure_logs(level=app_cfg.log_level, log_dir=app_cfg.data_dir / "logs")
        logger.info(
            "config loaded mode=%s broker=%s",
            app_cfg.mode,
            app_cfg.primary_broker_config.type,
        )

        self.sqlite = SqliteStore(app_cfg.data_dir / "stinger.db")
        self.sqlite.create_all()

        self.engine = TradingEngine(clock=LiveClock())
        self.bus = self.engine.bus

        # Multi-account broker pool. broker_list yields one BrokerConfig per
        # configured account; backward-compat configs with a singular
        # `broker:` collapse to a one-element list with id="default".
        self._pool = BrokerPool()
        for bcfg in app_cfg.broker_list:
            broker = build_broker(bcfg, self.bus)
            self._pool.add(bcfg.id, broker)
            self.engine.register(broker)

        # Map strategy_id → account_id for the router + UI labelling.
        strategy_accounts: dict[str, str] = {
            s.id: s.account for s in self.full_cfg.strategies.strategies
        }
        # Reject configs that point at unknown brokers up-front.
        for sid, account_id in strategy_accounts.items():
            if not self._pool.has(account_id):
                raise ConfigError(
                    f"strategy {sid!r} targets account {account_id!r} but no "
                    f"broker with that id is configured (known: {sorted(b for b, _ in self._pool.items())})"
                )

        self.handle = EngineHandle(
            bus=self.bus,
            brokers=self._pool,
            runners=self.runners,
            strategy_accounts=strategy_accounts,
        )
        self._strategy_accounts = strategy_accounts
        magic_by_id: dict[str, int] = {}

        # Risk monitor — starts before strategies so it sees their first events.
        self._risk = RiskMonitor(self.bus, app_cfg.risk)
        await self._risk.start()

        # Build strategies
        for entry in self.full_cfg.strategies.strategies:
            if not entry.enabled:
                continue
            await self._add_strategy_internal(entry, magic_by_id)

        # Order router — multi-account aware. handle_signal picks the broker
        # from the pool based on the strategy → account mapping; unknown
        # strategies fall back to the primary broker.
        self._router = OrderRouter(
            self.bus,
            pool=self._pool,
            strategy_magic=magic_by_id,
            strategy_accounts=strategy_accounts,
            risk=self._risk,
        )
        await self._router.attach()

        # Broker subscriptions are deferred until run_until_signal(), because
        # the brokers aren't connected until engine.start() runs through the
        # registered components — calling subscribe_bars() on a not-yet-
        # connected broker raises BrokerNotConnectedError.

        # Notification dispatcher — Telegram / Discord webhooks fired off the
        # bus. Registered as an engine lifecycle component so it starts +
        # stops with the engine and cleans up the httpx client.
        if app_cfg.notifications:
            from stinger_fx.observability import NotificationDispatcher

            self._notifications = NotificationDispatcher(self.bus, app_cfg.notifications)
            self.engine.register(self._notifications)

        # Hot-reload plumbing — uses the primary broker for legacy actions.
        self._reloader = ConfigReloader(self._build_reload_actions(self._pool.primary()))
        self._watcher = ConfigWatcher(self.config_dir, self._on_config_change)

        # Optional Prometheus metrics — collector subscribes to the bus and a
        # standalone HTTP server exposes /metrics on a configurable port. The
        # collector runs as a lifecycle component so it stops with the engine.
        if app_cfg.metrics.enabled:
            from stinger_fx.observability import MetricsCollector, start_metrics_server

            self._metrics = MetricsCollector(self.bus)
            self.engine.register(self._metrics)
            try:
                start_metrics_server(
                    port=app_cfg.metrics.port,
                    addr=app_cfg.metrics.host,
                )
            except OSError as e:
                # Don't kill the engine just because the metrics port is taken.
                logger.error(
                    "metrics_server_bind_failed host=%s port=%d error=%s",
                    app_cfg.metrics.host, app_cfg.metrics.port, e,
                )

        # UI — TUI takes the loop, web mounts uvicorn alongside, normal owns
        # stdout. tui+web are launched in run_until_signal so the engine is
        # already running by the time they mount.
        self._mode = app_cfg.mode
        self._web_host = app_cfg.web.host
        self._web_port = app_cfg.web.port
        if app_cfg.mode == "normal":
            self._ui = NormalUI(self.handle)
            self.engine.register(self._ui)
        elif app_cfg.mode in ("tui", "web"):
            logger.info("UI mode=%s — will mount after engine start", app_cfg.mode)

    async def run_until_signal(self) -> None:
        assert (
            self.engine is not None
        )
        assert (
            self._watcher is not None
        )
        assert (
            len(self._pool) > 0
        )
        # Schedule periodic account snapshots BEFORE engine.start so they're
        # registered on the engine's scheduler. The scheduler starts inside
        # engine.start() and will begin firing the job immediately after.
        self.engine.scheduler.every(
            5.0, self._publish_account_snapshot, name="account_snapshot"
        )
        await self.engine.start()
        # Now that every broker is connected (via engine.start), subscribe
        # each one to the (symbol, timeframe) pairs its strategies declared.
        await self._wire_broker_subscriptions_multi()
        await self._watcher.start()

        if self._mode == "tui":
            # Textual takes over the loop until the user quits with 'q'.
            # Lazy-import so non-TUI runs don't pay the import cost.
            from stinger_fx.ui.tui import StingerTUI

            assert self.handle is not None
            # Pass the risk monitor so the TUI's kill-switch field + 'r' binding
            # work end-to-end; the TUI accepts it duck-typed.
            tui = StingerTUI(self.handle, risk=self._risk)
            try:
                await tui.run_async()
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
        elif self._mode == "web":
            # uvicorn runs in the same asyncio loop as the engine. SIGINT
            # propagates through the awaited serve() and we fall through to
            # the cleanup sequence below.
            import uvicorn

            from stinger_fx.ui.web import create_app

            assert self.handle is not None
            assert self.full_cfg is not None
            web_app = create_app(self.handle, data_dir=self.full_cfg.app.data_dir)
            config = uvicorn.Config(
                web_app,
                host=self._web_host,
                port=self._web_port,
                log_level="warning",
                access_log=False,
                loop="asyncio",
            )
            server = uvicorn.Server(config)
            logger.info("web ui listening on http://%s:%d", self._web_host, self._web_port)
            try:
                await server.serve()
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
        else:
            loop = asyncio.get_running_loop()
            stop = asyncio.Event()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, stop.set)
                except NotImplementedError:
                    pass  # not supported on Windows for some signals
            await stop.wait()

        await self._watcher.stop()
        if self._router is not None:
            await self._router.detach()
        if self._risk is not None:
            await self._risk.stop()
        await self.engine.stop()

    async def _publish_account_snapshot(self) -> None:
        """Scheduler job: pull account state from every broker, fan out on bus.

        Each broker publishes its own AccountSnapshotEvent with its own
        `snapshot.account_id`, so per-account observers (risk, UI, metrics)
        can disambiguate without extra plumbing.
        """
        if self.bus is None or len(self._pool) == 0:
            return
        for account_id, broker in self._pool.items():
            try:
                snapshot = await broker.get_account_snapshot()
            except Exception:
                logger.exception("account snapshot failed account_id=%s", account_id)
                continue
            await self.bus.publish(AccountSnapshotEvent(snapshot=snapshot))

    # --- Strategy lifecycle (also used by hot reload) ---------------------

    async def _add_strategy_internal(
        self, entry: StrategyEntry, magic_by_id: dict[str, int]
    ) -> None:
        assert self.bus is not None
        assert self.engine is not None
        strategy_cls = load_strategy_class(entry.class_path)
        params = validate_params(strategy_cls, entry.params)
        magic = derive_magic(entry.id)
        magic_by_id[entry.id] = magic

        async def sink(sig):
            assert self.bus is not None
            await self.bus.publish(SignalEvent(signal=sig))

        runner = StrategyRunner(
            strategy_id=entry.id,
            strategy=strategy_cls(),
            params=params,
            bus=self.bus,
            clock=self.engine.clock,
            reload_lock=self.engine.reload_lock,
            signal_sink=sink,
        )
        await runner.start()
        self.runners[entry.id] = runner
        logger.info("strategy_started id=%s class=%s", entry.id, entry.class_path)

    async def _wire_broker_subscriptions(self, broker) -> None:
        """Subscribe a single broker to every strategy's declared feeds.

        Kept for backward compatibility / the hot-reload action; the live-mode
        startup uses `_wire_broker_subscriptions_multi` instead so it can fan
        out per-broker.
        """
        assert self.bus is not None
        seen: set[tuple[str, Timeframe]] = set()
        for runner in self.runners.values():
            for sub in runner.strategy.subscriptions(runner._params):
                key = (sub.symbol, sub.timeframe)
                if key in seen:
                    continue
                seen.add(key)
                await self._subscribe_one(broker, sub.symbol, sub.timeframe)

    async def _wire_broker_subscriptions_multi(self) -> None:
        """Multi-account variant: for each strategy, subscribe only the broker
        that owns its account. Brokers without any active strategies stay
        idle (no symbol subscriptions, no tick pump cost).
        """
        assert self.bus is not None
        # Track per-broker dedupe so the same symbol isn't subscribed twice
        # for the same broker even when multiple strategies share it.
        per_broker_seen: dict[str, set[tuple[str, Timeframe]]] = {
            acc: set() for acc, _ in self._pool.items()
        }
        for sid, runner in self.runners.items():
            account_id = self._strategy_accounts.get(sid, self._pool.primary_id())
            broker = self._pool.get(account_id)
            for sub in runner.strategy.subscriptions(runner._params):
                key = (sub.symbol, sub.timeframe)
                if key in per_broker_seen[account_id]:
                    continue
                per_broker_seen[account_id].add(key)
                await self._subscribe_one(broker, sub.symbol, sub.timeframe)

    async def _subscribe_one(self, broker, symbol: str, tf: Timeframe) -> None:
        """Subscribe broker to (symbol, tf), wiring a BarAggregator if the
        timeframe isn't native to MT5."""
        assert self.bus is not None
        await broker.subscribe_bars(symbol, tf)
        if not tf.is_native_mt5 and tf.value != "TICK":
            agg = BarAggregator(symbol, tf, self.bus)
            self.bus.subscribe(
                TickEvent,
                agg.on_tick,  # type: ignore[arg-type]
                name=f"agg.{symbol}.{tf.value}",
            )
            self.aggregators[(symbol, tf)] = agg

    def _build_reload_actions(self, broker) -> ReloadActions:
        magic_by_id: dict[str, int] = {sid: derive_magic(sid) for sid in self.runners}

        async def _subscribe_for_strategy(sid: str, account_id: str) -> None:
            """Wire the *correct* broker for `account_id` to this strategy's
            declared feeds. Pre-fix the add-callback used a captured primary
            broker, so new strategies routed to demo_b got their ticks from
            demo_a (or whatever was primary at startup)."""
            assert self.bus is not None
            target_broker = (
                self._pool.get(account_id)
                if self._pool.has(account_id)
                else self._pool.primary()
            )
            runner = self.runners.get(sid)
            if runner is None:
                return
            seen: set[tuple[str, Timeframe]] = set()
            for sub in runner.strategy.subscriptions(runner._params):
                key = (sub.symbol, sub.timeframe)
                if key in seen:
                    continue
                seen.add(key)
                await self._subscribe_one(target_broker, sub.symbol, sub.timeframe)

        async def add(entry: StrategyEntry) -> None:
            await self._add_strategy_internal(entry, magic_by_id)
            # Account routing — pre-fix this was missing, so new strategies
            # silently fell back to the primary broker regardless of their
            # `account:` field in YAML.
            self._strategy_accounts[entry.id] = entry.account
            if self._router is not None:
                self._router.strategy_magic.update(magic_by_id)
                self._router.strategy_accounts[entry.id] = entry.account
            await _subscribe_for_strategy(entry.id, entry.account)

        async def remove(sid: str) -> None:
            runner = self.runners.pop(sid, None)
            if runner is None:
                return
            await runner.stop()
            # Drop routing entries so a later config-add with the same id
            # starts from a clean slate.
            self._strategy_accounts.pop(sid, None)
            if self._router is not None:
                self._router.strategy_accounts.pop(sid, None)

        async def replace(entry: StrategyEntry) -> None:
            await remove(entry.id)
            await add(entry)

        async def update_params(sid: str, raw: dict) -> None:
            runner = self.runners.get(sid)
            if runner is None:
                return
            params = validate_params(type(runner.strategy), raw)
            await runner.update_params(params)

        async def set_enabled(sid: str, enabled: bool) -> None:
            runner = self.runners.get(sid)
            if runner is None:
                return
            if enabled:
                await runner.resume()
            else:
                await runner.pause()

        async def change_account(sid: str, account_id: str) -> None:
            """Hot re-route a running strategy to a different broker.

            The strategy task keeps running on the bus; only the order-routing
            map flips, plus we subscribe the new broker to the strategy's
            feeds so its tick pump is awake. The previous broker is left
            subscribed — other strategies may still need those symbols, and
            an idle subscription is cheap.
            """
            if not self._pool.has(account_id):
                logger.error(
                    "reload_change_account_unknown sid=%s account=%s known=%s",
                    sid, account_id, sorted(b for b, _ in self._pool.items()),
                )
                raise ValueError(
                    f"unknown account {account_id!r} for strategy {sid!r} — "
                    f"add the broker to app.yaml first or revert the change"
                )
            if sid not in self.runners:
                logger.warning(
                    "reload_change_account_unknown_strategy sid=%s — ignored",
                    sid,
                )
                return
            self._strategy_accounts[sid] = account_id
            if self._router is not None:
                self._router.strategy_accounts[sid] = account_id
            await _subscribe_for_strategy(sid, account_id)

        async def log_level(level: str) -> None:
            set_level(level)

        async def risk(cfg) -> None:
            if self._risk is not None:
                self._risk.update_config(cfg)

        return ReloadActions(
            add_strategy=add,
            remove_strategy=remove,
            replace_strategy=replace,
            update_params=update_params,
            set_enabled=set_enabled,
            change_account=change_account,
            update_log_level=log_level,
            update_risk=risk,
        )

    async def _on_config_change(self) -> None:
        assert self.full_cfg is not None
        assert self._reloader is not None
        assert self.bus is not None
        try:
            new_cfg = load_all(self.config_dir)
        except Exception as e:
            logger.exception("config reload failed (load)")
            await self.bus.publish(ConfigReloadFailedEvent(file=str(self.config_dir), error=str(e)))
            return
        result = await self._reloader.diff_and_apply(self.full_cfg, new_cfg)
        if result.ok and (result.applied or result.needs_restart):
            self.full_cfg = new_cfg
            await self.bus.publish(
                ConfigReloadedEvent(
                    changes={
                        "applied": result.applied,
                        "needs_restart": result.needs_restart,
                    }
                )
            )
        if result.errors:
            await self.bus.publish(
                ConfigReloadFailedEvent(file=str(self.config_dir), error="; ".join(result.errors))
            )


async def assemble_and_run(config_dir: Path, *, mode_override: str | None = None) -> None:
    app = StingerApp(config_dir, mode_override=mode_override)
    await app.setup()
    await app.run_until_signal()
