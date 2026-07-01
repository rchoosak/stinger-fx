"""MT5 Strategy Tester driver.

Owns three responsibilities:
  • Spin up a ZeroMQ REP socket that talks to `stinger_fx_shim.mq5` running
    inside MT5 Strategy Tester.
  • Spawn `terminal64.exe /config:tester.ini /portable` so the tester runs
    headless.
  • Forward incoming ticks through the same engine path used in live mode
    (event bus + StrategyRunner + OrderRouter + SimBroker) and translate
    each `OrderRequestEvent` into the tiny JSON action the shim expects.

Phase 1 deliverable. End-to-end runs require Windows + MT5 + the compiled
shim EA — see `mt5_shim/README.md`. The Python side is unit-testable on any
OS by stubbing the ZMQ socket.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import subprocess
import sys
import textwrap
import time
from datetime import UTC, datetime
from pathlib import Path

import zmq
import zmq.asyncio

from stinger_fx.backtest.base import BaseBacktester
from stinger_fx.backtest.mt5_report_parser import parse_report
from stinger_fx.backtest.reports import BacktestReport
from stinger_fx.brokers.bar_aggregator import BarAggregator
from stinger_fx.config.models import BacktestRunConfig, StrategyEntry
from stinger_fx.core import AsyncEventBus, SimClock
from stinger_fx.core.event_bus import Subscription as BusSubscription
from stinger_fx.core.events import (
    TickEvent,
)
from stinger_fx.domain import (
    OrderRequest,
    OrderType,
    Position,
    Side,
    Tick,
)
from stinger_fx.strategies import (
    StrategyRunner,
    derive_magic,
    load_strategy_class,
    validate_params,
)

logger = logging.getLogger("stinger.backtest.mt5_tester")

DEFAULT_ENDPOINT = "tcp://127.0.0.1:5555"

# The terminal64.exe launcher detaches (relaunches itself and exits), so the
# spawned process handle goes dead immediately and can't gate the serve loop.
# Instead we drive completion off the wire + the report file:
#   • STARTUP_TIMEOUT — give up if the shim never connects (DLL imports off,
#     no tick history for the requested range, EA failed to compile, …).
#   • IDLE_TIMEOUT    — once ticks have flowed, this much silence means the
#     tester finished (and is writing/whas written the report on shutdown).
STARTUP_TIMEOUT_S = 300.0
IDLE_TIMEOUT_S = 60.0


class MT5StrategyTester(BaseBacktester):
    name = "mt5_tester"

    def __init__(
        self,
        *,
        strategy: StrategyEntry,
        terminal_path: Path,
        tester_workdir: Path,
        report_dir: Path,
        endpoint: str = DEFAULT_ENDPOINT,
    ) -> None:
        self._strategy_entry = strategy
        self._terminal_path = terminal_path
        self._tester_workdir = tester_workdir
        self._report_dir = report_dir
        self._endpoint = endpoint
        self._pending_orders: list[OrderRequest] = []
        self._next_ticket = 1
        self._positions: dict[int, Position] = {}

    # --- BaseBacktester ------------------------------------------------------

    async def run(self, cfg: BacktestRunConfig) -> BacktestReport:
        self._report_dir.mkdir(parents=True, exist_ok=True)
        self._tester_workdir.mkdir(parents=True, exist_ok=True)
        # Absolute paths — the terminal runs with its own cwd, so a relative
        # Report= line would land somewhere we never look.
        report_xml = (self._report_dir / f"{cfg.id}_report.xml").resolve()
        tester_ini = (self._tester_workdir / f"{cfg.id}_tester.ini").resolve()
        # A stale report from a prior run would make us "complete" instantly.
        if report_xml.exists():
            report_xml.unlink()
        self._write_tester_ini(cfg, tester_ini, report_xml)

        # Wire bus + strategy + router so we can intercept order requests
        bus = AsyncEventBus()
        sim_clock = SimClock(cfg.start)
        strategy_cls = load_strategy_class(self._strategy_entry.class_path)
        params_dict = dict(self._strategy_entry.params)
        param_fields = set(strategy_cls.Params.model_fields.keys())
        if "symbol" in param_fields and cfg.symbol is not None:
            params_dict.setdefault("symbol", cfg.symbol)
        if "timeframe" in param_fields and cfg.timeframe is not None:
            params_dict.setdefault("timeframe", cfg.timeframe.value)
        params = validate_params(strategy_cls, params_dict)
        magic = derive_magic(self._strategy_entry.id)

        async def signal_sink(sig):
            # Queue an order request for the next tick exchange.
            self._pending_orders.append(
                OrderRequest(
                    strategy_id=self._strategy_entry.id,
                    symbol=sig.symbol,
                    side=sig.side,
                    type=OrderType.MARKET,
                    volume=sig.suggested_volume or 0.01,
                    sl=sig.suggested_sl,
                    tp=sig.suggested_tp,
                    comment=sig.comment,
                    magic=magic,
                    client_order_id="",
                )
            )

        runner = StrategyRunner(
            strategy_id=self._strategy_entry.id,
            strategy=strategy_cls(),
            params=params,
            bus=bus,
            clock=sim_clock,
            reload_lock=asyncio.Lock(),
            signal_sink=signal_sink,
        )
        await runner.start()

        agg_subs: list[BusSubscription[TickEvent]] = []
        for sub in strategy_cls.subscriptions(params):
            if sub.timeframe.is_tick:
                continue
            agg = BarAggregator(sub.symbol, sub.timeframe, bus)
            agg_subs.append(
                bus.subscribe(
                    TickEvent,
                    agg.on_tick,
                    name=f"mt5_tester.agg.{sub.symbol}.{sub.timeframe.value}",
                )
            )

        # --- ZMQ + subprocess --------------------------------------------------
        ctx = zmq.asyncio.Context.instance()
        sock = ctx.socket(zmq.REP)
        sock.bind(self._endpoint)

        proc = self._spawn_terminal(tester_ini)
        logger.info("MT5 tester spawned pid=%s", proc.pid)

        started_at = datetime.now(UTC)
        try:
            await self._serve(sock, bus, sim_clock, proc, runner, report_xml)
        finally:
            sock.close(linger=0)
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

        for sub in agg_subs:
            await sub.unsubscribe()
        await runner.stop()
        await bus.close()
        finished_at = datetime.now(UTC)

        # --- Report ------------------------------------------------------------
        if report_xml.exists():
            report = parse_report(report_xml, run_id=cfg.id, strategy_id=self._strategy_entry.id)
            report.started_at = started_at
            report.finished_at = finished_at
            if not report.initial_balance:
                report.initial_balance = cfg.initial_balance
            return report

        # No report — return whatever we tracked from the wire protocol
        return BacktestReport(
            run_id=cfg.id,
            strategy_id=self._strategy_entry.id,
            started_at=started_at,
            finished_at=finished_at,
            initial_balance=cfg.initial_balance,
            final_balance=cfg.initial_balance,
        )

    # --- Internals ----------------------------------------------------------

    async def _serve(
        self,
        sock,
        bus: AsyncEventBus,
        sim_clock: SimClock,
        proc: subprocess.Popen,
        runner: StrategyRunner,
        report_xml: Path,
    ) -> None:
        del runner, proc  # the launcher detaches; never gate on its handle
        started = time.monotonic()
        last_activity: float | None = None
        while True:
            # The terminal writes the report on shutdown (ShutdownTerminal=1).
            # Its appearance is the authoritative "test finished" signal.
            if await asyncio.to_thread(report_xml.exists):
                logger.info("MT5 tester report detected — finishing")
                break
            try:
                raw = await asyncio.wait_for(sock.recv(), timeout=1.0)
            except TimeoutError:
                now = time.monotonic()
                if last_activity is None:
                    if now - started > STARTUP_TIMEOUT_S:
                        logger.warning(
                            "MT5 tester: shim never connected within %.0fs — "
                            "check 'Allow DLL imports' and that tick history "
                            "exists for the requested range",
                            STARTUP_TIMEOUT_S,
                        )
                        break
                elif now - last_activity > IDLE_TIMEOUT_S:
                    logger.info(
                        "MT5 tester: idle %.0fs after activity — finishing",
                        IDLE_TIMEOUT_S,
                    )
                    break
                continue
            last_activity = time.monotonic()
            payload = json.loads(raw.decode("utf-8"))
            if payload.get("type") == "tick":
                tick_time = datetime.fromtimestamp(int(payload["time"]), tz=UTC)
                sim_clock.advance(tick_time)
                tick = Tick(
                    symbol=payload["symbol"],
                    time=tick_time,
                    bid=float(payload["bid"]),
                    ask=float(payload["ask"]),
                )
                await bus.publish(TickEvent(tick=tick))
                # Let the strategy queue process the tick and possibly emit a signal
                for _ in range(3):
                    await asyncio.sleep(0)

                reply = self._dequeue_action(tick.symbol, tick.bid, tick.ask)
                await sock.send_string(json.dumps(reply))
            else:
                await sock.send_string(json.dumps({"action": "NONE"}))

    def _dequeue_action(self, symbol: str, bid: float, ask: float) -> dict:
        # Take one pending order at a time — keep it simple.
        if not self._pending_orders:
            return {"action": "NONE"}
        req = self._pending_orders.pop(0)
        action = {"action": "BUY" if req.side is Side.BUY else "SELL", "volume": req.volume}
        if req.sl is not None:
            action["sl"] = req.sl
        if req.tp is not None:
            action["tp"] = req.tp
        # We don't try to track fills here — the MT5 report is the source of truth.
        del symbol, bid, ask
        return action

    def _spawn_terminal(self, tester_ini: Path) -> subprocess.Popen:
        config_path = _windows_short_path(tester_ini) if sys.platform == "win32" else str(tester_ini)
        return subprocess.Popen(
            [str(self._terminal_path), f"/config:{config_path}", "/portable"],
            cwd=str(self._tester_workdir),
        )

    @staticmethod
    def _write_tester_ini(cfg: BacktestRunConfig, ini: Path, report_xml: Path) -> None:
        ini.parent.mkdir(parents=True, exist_ok=True)
        report_path = _windows_short_path(report_xml) if sys.platform == "win32" else str(report_xml)
        # mt5_tester is single-symbol/single-timeframe; both fields are
        # mandatory (the multi-feed path is for file mode only).
        assert cfg.symbol is not None
        assert cfg.timeframe is not None
        content = textwrap.dedent(
            f"""
            [Tester]
            Expert=StingerFx\\stinger_fx_shim
            Symbol={cfg.symbol}
            Period={cfg.timeframe.value}
            Optimization=0
            Model=4
            FromDate={cfg.start.strftime("%Y.%m.%d")}
            ToDate={cfg.end.strftime("%Y.%m.%d")}
            ForwardMode=0
            Deposit={int(cfg.initial_balance)}
            Currency=USD
            Leverage=1:100
            ExecutionMode=0
            Report={report_path}
            ReplaceReport=1
            ShutdownTerminal=1
            """
        ).strip()
        ini.write_text(content, encoding="utf-16")


def _windows_short_path(path: Path) -> str:
    """Return an 8.3 path so MT5 can parse /config: under Program Files."""
    raw = str(path)
    get_short_path_name = ctypes.windll.kernel32.GetShortPathNameW
    needed = get_short_path_name(raw, None, 0)
    if needed == 0 and not path.exists() and path.parent.exists():
        return str(Path(_windows_short_path(path.parent)) / path.name)
    if needed == 0:
        return raw
    buffer = ctypes.create_unicode_buffer(needed)
    written = get_short_path_name(raw, buffer, needed)
    return buffer.value if written else raw
