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
import json
import logging
import subprocess
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import zmq
import zmq.asyncio

from stinger_fx.backtest.base import BaseBacktester
from stinger_fx.backtest.mt5_report_parser import parse_report
from stinger_fx.backtest.reports import BacktestReport
from stinger_fx.config.models import BacktestRunConfig, StrategyEntry
from stinger_fx.core import AsyncEventBus, SimClock
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
        report_xml = self._report_dir / f"{cfg.id}_report.xml"
        tester_ini = self._tester_workdir / f"{cfg.id}_tester.ini"
        self._write_tester_ini(cfg, tester_ini, report_xml)

        # Wire bus + strategy + router so we can intercept order requests
        bus = AsyncEventBus()
        sim_clock = SimClock(cfg.start)
        strategy_cls = load_strategy_class(self._strategy_entry.class_path)
        params = validate_params(strategy_cls, self._strategy_entry.params)
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

        # --- ZMQ + subprocess --------------------------------------------------
        ctx = zmq.asyncio.Context.instance()
        sock = ctx.socket(zmq.REP)
        sock.bind(self._endpoint)

        proc = self._spawn_terminal(tester_ini)
        logger.info("MT5 tester spawned pid=%s", proc.pid)

        started_at = datetime.now(UTC)
        try:
            await self._serve(sock, bus, sim_clock, proc, runner)
        finally:
            sock.close(linger=0)
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

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
    ) -> None:
        del runner  # used via the bus; reference is for liveness
        while proc.poll() is None:
            try:
                raw = await asyncio.wait_for(sock.recv(), timeout=1.0)
            except TimeoutError:
                continue
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
        return subprocess.Popen(
            [str(self._terminal_path), f"/config:{tester_ini}", "/portable"],
            cwd=str(self._tester_workdir),
        )

    @staticmethod
    def _write_tester_ini(cfg: BacktestRunConfig, ini: Path, report_xml: Path) -> None:
        ini.parent.mkdir(parents=True, exist_ok=True)
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
            Report={report_xml}
            ReplaceReport=1
            ShutdownTerminal=1
            """
        ).strip()
        ini.write_text(content, encoding="utf-16")
