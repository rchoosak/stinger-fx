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
from stinger_fx.brokers import build_broker
from stinger_fx.brokers.bar_aggregator import BarAggregator
from stinger_fx.config import (
    ConfigReloader,
    ConfigWatcher,
    FullConfig,
    ReloadActions,
    StrategyEntry,
    load_all,
)
from stinger_fx.brokers.base import BaseBroker
from stinger_fx.core import AsyncEventBus, LiveClock, TradingEngine
from stinger_fx.core.events import (
    ConfigReloadedEvent,
    ConfigReloadFailedEvent,
    SignalEvent,
    TickEvent,
)
from stinger_fx.data import SqliteStore
from stinger_fx.domain.timeframes import Timeframe
from stinger_fx.log import configure as configure_logs
from stinger_fx.log import set_level
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

    def __init__(self, config_dir: Path) -> None:
        self.config_dir = config_dir
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
        self._broker: BaseBroker | None = None

    async def setup(self) -> None:
        self.full_cfg = load_all(self.config_dir)
        app_cfg = self.full_cfg.app

        configure_logs(level=app_cfg.log_level, log_dir=app_cfg.data_dir / "logs")
        logger.info("config loaded mode=%s broker=%s", app_cfg.mode, app_cfg.broker.type)

        self.sqlite = SqliteStore(app_cfg.data_dir / "stinger.db")
        self.sqlite.create_all()

        self.engine = TradingEngine(clock=LiveClock())
        self.bus = self.engine.bus

        broker = build_broker(app_cfg.broker, self.bus)
        self.engine.register(broker)

        self.handle = EngineHandle(bus=self.bus, broker=broker, runners=self.runners)
        magic_by_id: dict[str, int] = {}

        # Build strategies
        for entry in self.full_cfg.strategies.strategies:
            if not entry.enabled:
                continue
            await self._add_strategy_internal(entry, magic_by_id)

        # Order router
        self._router = OrderRouter(self.bus, broker, strategy_magic=magic_by_id)
        await self._router.attach()

        # Broker subscriptions are deferred until run_until_signal(), because
        # the broker is not connected until engine.start() runs through the
        # registered components — calling subscribe_bars() on a not-yet-connected
        # broker raises BrokerNotConnectedError.
        self._broker = broker

        # Hot-reload plumbing
        self._reloader = ConfigReloader(self._build_reload_actions(broker))
        self._watcher = ConfigWatcher(self.config_dir, self._on_config_change)

        # UI
        if app_cfg.mode == "normal":
            self._ui = NormalUI(self.handle)
            self.engine.register(self._ui)
        elif app_cfg.mode in ("tui", "web"):
            logger.warning("UI mode %s is a Phase-2 feature; falling back to normal", app_cfg.mode)
            self._ui = NormalUI(self.handle)
            self.engine.register(self._ui)

    async def run_until_signal(self) -> None:
        assert (
            self.engine is not None
            and self._watcher is not None
            and self._broker is not None
        )
        await self.engine.start()
        # Now that the broker is connected (via engine.start), subscribe it to
        # every (symbol, timeframe) the active strategies declared.
        await self._wire_broker_subscriptions(self._broker)
        await self._watcher.start()

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
        await self.engine.stop()

    # --- Strategy lifecycle (also used by hot reload) ---------------------

    async def _add_strategy_internal(
        self, entry: StrategyEntry, magic_by_id: dict[str, int]
    ) -> None:
        assert self.bus is not None and self.engine is not None
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
        # Aggregate the union of all strategy subscriptions
        assert self.bus is not None
        seen: set[tuple[str, Timeframe]] = set()
        for runner in self.runners.values():
            subs = runner.strategy.subscriptions(runner._params)
            for sub in subs:
                key = (sub.symbol, sub.timeframe)
                if key in seen:
                    continue
                seen.add(key)
                await broker.subscribe_bars(sub.symbol, sub.timeframe)
                if not sub.timeframe.is_native_mt5 and sub.timeframe.value != "TICK":
                    agg = BarAggregator(sub.symbol, sub.timeframe, self.bus)
                    # Subscribe the aggregator to TickEvent (the bus dispatches
                    # by isinstance, so `type(agg).__mro__[0]` was a no-op bug
                    # subscribing to the wrong type and dropping every tick).
                    self.bus.subscribe(
                        TickEvent,
                        agg.on_tick,  # type: ignore[arg-type]
                        name=f"agg.{sub.symbol}.{sub.timeframe.value}",
                    )
                    self.aggregators[key] = agg

    def _build_reload_actions(self, broker) -> ReloadActions:
        magic_by_id: dict[str, int] = {sid: derive_magic(sid) for sid in self.runners}

        async def add(entry: StrategyEntry) -> None:
            await self._add_strategy_internal(entry, magic_by_id)
            if self._router is not None:
                self._router.strategy_magic.update(magic_by_id)
            await self._wire_broker_subscriptions(broker)

        async def remove(sid: str) -> None:
            runner = self.runners.pop(sid, None)
            if runner is None:
                return
            await runner.stop()

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

        async def log_level(level: str) -> None:
            set_level(level)

        async def risk(_cfg) -> None:
            # OrderRouter risk hookup is Phase-2; just acknowledge for now.
            logger.info("risk config noted (no live application yet)")

        return ReloadActions(
            add_strategy=add,
            remove_strategy=remove,
            replace_strategy=replace,
            update_params=update_params,
            set_enabled=set_enabled,
            update_log_level=log_level,
            update_risk=risk,
        )

    async def _on_config_change(self) -> None:
        assert self.full_cfg is not None and self._reloader is not None and self.bus is not None
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


async def assemble_and_run(config_dir: Path) -> None:
    app = StingerApp(config_dir)
    await app.setup()
    await app.run_until_signal()
