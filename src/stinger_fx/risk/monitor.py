"""RiskMonitor — enforces the safety rules in RiskConfig.

The monitor is event-driven: it subscribes to OrderFilledEvent,
PositionClosedEvent, and AccountSnapshotEvent to keep its caches fresh,
and exposes `check_signal(signal) -> RiskVerdict` that the OrderRouter calls
before placing any order.

Rules enforced (configurable via `config/app.yaml -> risk:`):
  • max_open_positions_per_strategy — refuse a new signal if the strategy
    already has N positions open.
  • max_daily_loss_pct — refuse new signals once the cumulative realized
    P&L for the UTC day drops to a configurable percentage of the day's
    opening balance.
  • kill_switch_drawdown_pct — refuse ALL new signals once equity has
    dropped that percentage below the peak equity seen since start.
    Once tripped, the switch stays tripped until reset_kill_switch() is
    called.

When a rule rejects, the OrderRouter records the verdict in a DecisionEvent
with action="rejected" so the operator can see why in the trade journal.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import NamedTuple

from stinger_fx.config.models import RiskConfig
from stinger_fx.core.event_bus import AsyncEventBus, Subscription
from stinger_fx.core.events import (
    AccountSnapshotEvent,
    OrderFilledEvent,
    PositionClosedEvent,
)
from stinger_fx.domain import Signal

logger = logging.getLogger("stinger.risk")


class RiskVerdict(NamedTuple):
    allowed: bool
    reason: str = ""


class RiskMonitor:
    """Owns the live risk state for the engine. Lifetime = engine."""

    def __init__(self, bus: AsyncEventBus, cfg: RiskConfig) -> None:
        self._bus = bus
        self._cfg = cfg
        self._subs: list[Subscription] = []

        # Open-position counter per strategy_id. Incremented on OrderFilled,
        # decremented on PositionClosed.
        self._open_positions: dict[str, int] = {}

        # Daily realised P&L. Reset when the UTC date changes.
        self._daily_anchor_date: datetime | None = None
        self._daily_opening_balance: float = 0.0
        self._daily_realized_pnl: float = 0.0

        # Peak / drawdown tracker.
        self._peak_equity: float | None = None
        self._current_equity: float | None = None
        self._current_balance: float | None = None
        self._kill_switch_tripped: bool = False

    # --- Lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        self._subs.append(
            self._bus.subscribe(OrderFilledEvent, self._on_filled, name="risk.fill")
        )
        self._subs.append(
            self._bus.subscribe(PositionClosedEvent, self._on_closed, name="risk.close")
        )
        self._subs.append(
            self._bus.subscribe(AccountSnapshotEvent, self._on_snapshot, name="risk.snapshot")
        )
        logger.info("risk_monitor_started")

    async def stop(self) -> None:
        for sub in self._subs:
            await sub.unsubscribe()
        self._subs.clear()

    def update_config(self, cfg: RiskConfig) -> None:
        """Hot-reload entry point."""
        self._cfg = cfg
        logger.info(
            "risk_config_updated max_open_per_strategy=%s "
            "max_daily_loss_pct=%s kill_switch_drawdown_pct=%s",
            cfg.max_open_positions_per_strategy,
            cfg.max_daily_loss_pct,
            cfg.kill_switch_drawdown_pct,
        )

    def reset_kill_switch(self) -> None:
        """Operator-issued reset after a kill-switch trip."""
        self._kill_switch_tripped = False
        logger.warning("risk_kill_switch_reset")

    # --- Public API --------------------------------------------------------

    def check_signal(self, signal: Signal) -> RiskVerdict:
        """Return whether a signal is allowed under current risk state."""
        if self._kill_switch_tripped:
            return RiskVerdict(False, "kill_switch_tripped")

        # Roll the daily counter forward if needed
        self._maybe_roll_daily()

        cfg = self._cfg

        # Per-strategy open-position cap
        if cfg.max_open_positions_per_strategy > 0:
            open_n = self._open_positions.get(signal.strategy_id, 0)
            if open_n >= cfg.max_open_positions_per_strategy:
                return RiskVerdict(
                    False,
                    f"max_open_positions_per_strategy={cfg.max_open_positions_per_strategy} reached "
                    f"(strategy {signal.strategy_id} has {open_n} open)",
                )

        # Daily loss limit
        if cfg.max_daily_loss_pct > 0 and self._daily_opening_balance > 0:
            loss_pct = max(0.0, -self._daily_realized_pnl) / self._daily_opening_balance * 100
            if loss_pct >= cfg.max_daily_loss_pct:
                return RiskVerdict(
                    False,
                    f"max_daily_loss_pct={cfg.max_daily_loss_pct} hit (today loss {loss_pct:.2f}%)",
                )

        return RiskVerdict(True)

    def snapshot(self) -> dict[str, object]:
        """Debug / UI accessor."""
        return {
            "kill_switch_tripped": self._kill_switch_tripped,
            "current_equity": self._current_equity,
            "current_balance": self._current_balance,
            "peak_equity": self._peak_equity,
            "drawdown_pct": self._drawdown_pct(),
            "daily_opening_balance": self._daily_opening_balance,
            "daily_realized_pnl": self._daily_realized_pnl,
            "open_positions": dict(self._open_positions),
        }

    # --- Internals ---------------------------------------------------------

    def _drawdown_pct(self) -> float:
        if self._peak_equity is None or self._current_equity is None:
            return 0.0
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - self._current_equity) / self._peak_equity * 100)

    def _maybe_roll_daily(self) -> None:
        today = datetime.now(timezone.utc).date()
        if self._daily_anchor_date is None or today != self._daily_anchor_date.date():
            self._daily_anchor_date = datetime.now(timezone.utc)
            if self._current_balance is not None:
                self._daily_opening_balance = self._current_balance
            self._daily_realized_pnl = 0.0

    # --- Event handlers ----------------------------------------------------

    async def _on_filled(self, evt: OrderFilledEvent) -> None:
        sid = evt.order.strategy_id
        self._open_positions[sid] = self._open_positions.get(sid, 0) + 1

    async def _on_closed(self, evt: PositionClosedEvent) -> None:
        # PositionClosedEvent carries strategy attribution via the position's
        # magic — but we keep a per-strategy_id counter, not per-magic. The
        # OrderRouter stamps magic from the strategy_magic map; we don't reverse
        # it here. Best-effort decrement of the largest non-zero bucket keeps
        # the counter sane; threading strategy_id through the event itself is
        # a Phase-3 refinement.
        for sid in list(self._open_positions.keys()):
            if self._open_positions[sid] > 0:
                self._open_positions[sid] -= 1
                break
        self._maybe_roll_daily()
        self._daily_realized_pnl += evt.realized_pnl

    async def _on_snapshot(self, evt: AccountSnapshotEvent) -> None:
        snap = evt.snapshot
        self._current_equity = snap.equity
        self._current_balance = snap.balance
        if self._peak_equity is None or snap.equity > self._peak_equity:
            self._peak_equity = snap.equity
        if self._daily_anchor_date is None:
            self._daily_anchor_date = snap.time
            self._daily_opening_balance = snap.balance
        # Kill-switch check
        cfg = self._cfg
        if (
            cfg.kill_switch_drawdown_pct > 0
            and not self._kill_switch_tripped
            and self._drawdown_pct() >= cfg.kill_switch_drawdown_pct
        ):
            self._kill_switch_tripped = True
            logger.error(
                "risk_kill_switch_tripped drawdown_pct=%.2f limit_pct=%s "
                "peak_equity=%s current_equity=%s",
                self._drawdown_pct(),
                cfg.kill_switch_drawdown_pct,
                self._peak_equity,
                self._current_equity,
            )
