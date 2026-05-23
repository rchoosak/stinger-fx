"""Textual app — the `--mode tui` dashboard.

Subscribes to the same EngineHandle bus the normal-mode UI uses; each panel
is fed by a different event class. The TUI runs on the same asyncio loop as
the engine, so no thread crossings are needed.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header

from stinger_fx.core.event_bus import Subscription
from stinger_fx.core.events import (
    AccountSnapshotEvent,
    BarEvent,
    ConfigReloadedEvent,
    ConfigReloadFailedEvent,
    DecisionEvent,
    OrderFilledEvent,
    OrderRejectedEvent,
    StrategyStateChangedEvent,
    TickEvent,
)
from stinger_fx.ui.handle import EngineHandle
from stinger_fx.ui.tui.widgets.account_panel import AccountPanel
from stinger_fx.ui.tui.widgets.log_panel import LogPanel
from stinger_fx.ui.tui.widgets.market_panel import MarketPanel
from stinger_fx.ui.tui.widgets.positions_panel import PositionsPanel
from stinger_fx.ui.tui.widgets.strategies_panel import StrategiesPanel


class StingerTUI(App[None]):
    """The Textual dashboard. One per `--mode tui` invocation."""

    TITLE = "Stinger-Fx"
    SUB_TITLE = "EA Bot Platform"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("p", "toggle_pause", "Pause/Resume selected", show=True),
        Binding("r", "reset_kill", "Reset kill switch", show=True),
    ]

    CSS = """
    Screen {
        layout: vertical;
    }
    #top {
        height: auto;
    }
    #panels {
        height: auto;
    }
    """

    def __init__(self, handle: EngineHandle, risk: object | None = None) -> None:
        """`risk` is forward-compatible with the RiskMonitor type from the
        risk-module PR. Typed loosely here so this TUI PR stays independent."""
        super().__init__()
        self._handle = handle
        self._risk = risk
        self._subs: list[Subscription] = []
        self._poll_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="top"):
            yield AccountPanel(id="account")
            yield MarketPanel(id="market")
        with Horizontal(id="panels"):
            yield StrategiesPanel(id="strategies")
            yield PositionsPanel(id="positions")
        yield LogPanel(id="logs")
        yield Footer()

    async def on_mount(self) -> None:
        bus = self._handle.bus
        # Event fan-in
        self._subs.append(bus.subscribe(TickEvent, self._on_tick, name="tui.tick"))
        self._subs.append(bus.subscribe(BarEvent, self._on_bar, name="tui.bar"))
        self._subs.append(
            bus.subscribe(AccountSnapshotEvent, self._on_snapshot, name="tui.snapshot")
        )
        self._subs.append(bus.subscribe(OrderFilledEvent, self._on_filled, name="tui.fill"))
        self._subs.append(bus.subscribe(OrderRejectedEvent, self._on_rejected, name="tui.reject"))
        self._subs.append(bus.subscribe(DecisionEvent, self._on_decision, name="tui.decision"))
        self._subs.append(
            bus.subscribe(StrategyStateChangedEvent, self._on_strategy_state, name="tui.strategy")
        )
        self._subs.append(
            bus.subscribe(ConfigReloadedEvent, self._on_reload, name="tui.reload")
        )
        self._subs.append(
            bus.subscribe(ConfigReloadFailedEvent, self._on_reload_failed, name="tui.reload_fail")
        )

        # Pull-driven refresh: positions and strategies aren't necessarily
        # updated by every event, so poll every 2s.
        self._poll_task = asyncio.create_task(self._poll_loop())

        # Greet
        self.query_one(LogPanel).push(level="info", message="TUI mounted, watching engine bus")

    async def on_unmount(self) -> None:
        for sub in self._subs:
            await sub.unsubscribe()
        self._subs.clear()
        if self._poll_task is not None:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task

    # --- Polling -----------------------------------------------------------

    async def _poll_loop(self) -> None:
        while True:
            try:
                strategies = await self._handle.list_strategies()
                self.query_one(StrategiesPanel).refresh_rows(strategies)
                positions = await self._handle.get_positions()
                self.query_one(PositionsPanel).refresh_rows(positions)
                # Mirror risk state into the account panel if available.
                # Typed via duck-typing so this TUI PR doesn't depend on the
                # risk-module branch landing first.
                if self._risk is not None and hasattr(self._risk, "snapshot"):
                    snap = self._risk.snapshot()
                    panel = self.query_one(AccountPanel)
                    panel.peak_equity = snap.get("peak_equity")  # type: ignore[assignment]
                    panel.drawdown_pct = float(snap.get("drawdown_pct", 0.0))  # type: ignore[arg-type]
                    panel.kill_switch = bool(snap.get("kill_switch_tripped", False))
            except Exception as e:  # noqa: BLE001
                self.query_one(LogPanel).push(level="warning", message=f"poll: {e}")
            await asyncio.sleep(2)

    # --- Event handlers ----------------------------------------------------

    async def _on_tick(self, evt: TickEvent) -> None:
        panel = self.query_one(MarketPanel)
        panel.symbol = evt.tick.symbol
        panel.bid = evt.tick.bid
        panel.ask = evt.tick.ask
        panel.total_ticks = panel.total_ticks + 1

    async def _on_bar(self, evt: BarEvent) -> None:
        if not evt.bar.is_closed:
            return
        panel = self.query_one(MarketPanel)
        panel.last_bar_close = evt.bar.close
        panel.last_bar_time = evt.bar.time.strftime("%Y-%m-%d %H:%M")
        panel.last_bar_tf = evt.bar.timeframe.value
        panel.total_bars = panel.total_bars + 1
        self.query_one(LogPanel).push(
            level="info",
            message=f"bar_closed {evt.bar.symbol}@{evt.bar.timeframe.value} close={evt.bar.close:.5f}",
            ts=_localize(evt.ts),
        )

    async def _on_snapshot(self, evt: AccountSnapshotEvent) -> None:
        snap = evt.snapshot
        panel = self.query_one(AccountPanel)
        panel.balance = snap.balance
        panel.equity = snap.equity
        panel.profit = snap.profit
        panel.broker = (await self._handle.get_account()).broker
        panel.currency = (await self._handle.get_account()).currency

    async def _on_filled(self, evt: OrderFilledEvent) -> None:
        o = evt.order
        self.query_one(LogPanel).push(
            level="info",
            message=(
                f"order_filled strategy={o.strategy_id} symbol={o.symbol} "
                f"side={o.side.value} vol={o.volume} price={o.fill_price}"
            ),
            ts=_localize(evt.ts),
        )

    async def _on_rejected(self, evt: OrderRejectedEvent) -> None:
        o = evt.order
        self.query_one(LogPanel).push(
            level="warning",
            message=(
                f"order_rejected strategy={o.strategy_id} symbol={o.symbol} "
                f"reason={evt.reason}"
            ),
            ts=_localize(evt.ts),
        )

    async def _on_decision(self, evt: DecisionEvent) -> None:
        d = evt.decision
        if d.action == "rejected":
            self.query_one(LogPanel).push(
                level="warning",
                message=f"signal_rejected_by_risk strategy={d.signal.strategy_id} reason={d.reason}",
                ts=_localize(evt.ts),
            )

    async def _on_strategy_state(self, evt: StrategyStateChangedEvent) -> None:
        level = "info" if evt.state == "started" else "warning"
        if evt.state == "quarantined":
            level = "error"
        self.query_one(LogPanel).push(
            level=level,
            message=f"strategy_state id={evt.strategy_id} state={evt.state}",
            ts=_localize(evt.ts),
        )

    async def _on_reload(self, evt: ConfigReloadedEvent) -> None:
        self.query_one(LogPanel).push(
            level="info",
            message=f"config_reloaded applied={evt.changes.get('applied', [])} "
                    f"needs_restart={evt.changes.get('needs_restart', [])}",
            ts=_localize(evt.ts),
        )

    async def _on_reload_failed(self, evt: ConfigReloadFailedEvent) -> None:
        self.query_one(LogPanel).push(
            level="error",
            message=f"config_reload_failed file={evt.file} error={evt.error}",
            ts=_localize(evt.ts),
        )

    # --- Actions -----------------------------------------------------------

    async def action_toggle_pause(self) -> None:
        table = self.query_one(StrategiesPanel)
        try:
            row = table.get_row_at(table.cursor_row)
        except IndexError:
            return
        if not row:
            return
        sid = str(row[0])
        strategies = await self._handle.list_strategies()
        match = next((s for s in strategies if s.id == sid), None)
        if match is None:
            return
        if match.state == "paused":
            await self._handle.resume_strategy(sid)
            self.query_one(LogPanel).push(level="info", message=f"resumed {sid}")
        else:
            await self._handle.pause_strategy(sid)
            self.query_one(LogPanel).push(level="info", message=f"paused {sid}")

    async def action_reset_kill(self) -> None:
        if self._risk is None or not hasattr(self._risk, "reset_kill_switch"):
            self.query_one(LogPanel).push(level="warning", message="no risk monitor attached")
            return
        self._risk.reset_kill_switch()
        self.query_one(LogPanel).push(level="info", message="kill switch reset")


def _localize(ts: datetime) -> datetime:
    """Strip tz so RichLog timestamp formatting looks clean."""
    return ts.replace(tzinfo=None) if ts.tzinfo else ts
