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
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, NamedTuple

from stinger_fx.config.models import (
    PositionSizingConfig,
    RiskConfig,
    TradingFilterConfig,
)
from stinger_fx.core.clock import Clock, LiveClock
from stinger_fx.core.event_bus import AsyncEventBus, Subscription
from stinger_fx.core.events import (
    AccountSnapshotEvent,
    OrderFilledEvent,
    PartialClosedEvent,
    PositionClosedEvent,
)
from stinger_fx.domain import Signal

if TYPE_CHECKING:
    from stinger_fx.data.repositories import RiskStateRepo

logger = logging.getLogger("stinger.risk")


class RiskVerdict(NamedTuple):
    allowed: bool
    reason: str = ""


class RiskMonitor:
    """Owns the live risk state for the engine. Lifetime = engine."""

    def __init__(
        self,
        bus: AsyncEventBus,
        cfg: RiskConfig,
        *,
        clock: Clock | None = None,
        state_repo: RiskStateRepo | None = None,
    ) -> None:
        self._bus = bus
        self._cfg = cfg
        self._subs: list[Subscription] = []
        # Persists peak_equity + kill_switch_tripped so they survive an
        # engine restart (see rehydrate()). None in backtests / unit tests —
        # persistence is a live-engine concern only.
        self._state_repo = state_repo
        # Set by rehydrate(): the next AccountSnapshot reconstructs the day's
        # opening balance from (current balance − rehydrated realized P&L)
        # rather than treating the restart-time balance as the day's open.
        self._needs_opening_balance = False
        # Injectable clock so the daily-loss window rolls on *simulated*
        # time in backtests, not wall-clock. With the default LiveClock the
        # behavior is unchanged for the live engine. Without this, a backtest
        # that wired the monitor would roll the daily counter on the wall
        # clock — i.e. never, since a multi-day replay finishes in seconds —
        # so the daily-loss limit would accumulate the whole run instead of
        # resetting each simulated UTC day.
        self._clock: Clock = clock or LiveClock()

        # Open-position counter per strategy_id. Incremented on OrderFilled,
        # decremented on PositionClosed — *for the strategy that actually
        # owns the closed position*, looked up via _ticket_to_strategy.
        self._open_positions: dict[str, int] = {}

        # (account_id, ticket) → strategy_id, populated on OrderFilledEvent so that
        # PositionClosedEvent (which doesn't carry strategy_id) can be
        # attributed back to the right per-strategy bucket. Without this
        # account scoping, identical broker tickets can collide in a
        # multi-account process.
        self._ticket_to_strategy: dict[tuple[str, int], str] = {}

        # Per-symbol open-position counter (for per_symbol.max_open_positions).
        self._open_positions_by_symbol: dict[str, int] = {}

        # Daily realised P&L. Reset when the UTC date changes.
        self._daily_anchor_date: datetime | None = None
        self._daily_opening_balance: float = 0.0
        self._daily_realized_pnl: float = 0.0

        # Per-symbol daily realized P&L (for per_symbol.max_daily_loss_usd).
        self._daily_pnl_by_symbol: dict[str, float] = {}

        # Peak / drawdown tracker.
        self._peak_equity: float | None = None
        self._current_equity: float | None = None
        self._current_balance: float | None = None
        self._kill_switch_tripped: bool = False

        # Profit-lock — session-open equity + high-watermark drive a "lock in
        # gains" trip. Baseline is per engine run (resets on restart); a tripped
        # lock is persisted so it survives a restart like the kill switch.
        self._pl_open_equity: float | None = None
        self._pl_high_equity: float | None = None
        self._profit_lock_tripped: bool = False

    # --- Lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        self._subs.append(
            self._bus.subscribe(OrderFilledEvent, self._on_filled, name="risk.fill")
        )
        self._subs.append(
            self._bus.subscribe(PositionClosedEvent, self._on_closed, name="risk.close")
        )
        self._subs.append(
            self._bus.subscribe(
                PartialClosedEvent, self._on_partial_closed, name="risk.partial_close"
            )
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
        """Operator-issued reset after a kill-switch trip.

        Rebases the drawdown peak to the current equity so the reset actually
        lets trading resume. Without this, the next AccountSnapshot would
        re-measure drawdown against the old (pre-trip) peak and immediately
        re-trip while still inside the drawdown the operator just acknowledged
        — making reset a no-op exactly when it's needed. If no snapshot has
        arrived yet (`_current_equity is None`) the peak is left untouched.
        """
        self._kill_switch_tripped = False
        if self._current_equity is not None:
            self._peak_equity = self._current_equity
        logger.warning(
            "risk_kill_switch_reset peak_rebased_to=%s", self._current_equity
        )
        self._persist_state()

    def reset_profit_lock(self) -> None:
        """Operator-issued reset after a profit-lock trip. Rebases the
        session-open + high-watermark to current equity so the lock re-arms
        from here instead of immediately re-tripping."""
        self._profit_lock_tripped = False
        if self._current_equity is not None:
            self._pl_open_equity = self._current_equity
            self._pl_high_equity = self._current_equity
        logger.warning(
            "risk_profit_lock_reset rebased_to=%s", self._current_equity
        )
        self._persist_state()

    async def rehydrate(
        self,
        *,
        open_positions: Sequence[
            tuple[str | None, str, int] | tuple[str | None, str, int, str]
        ],
        daily_realized_pnl: float,
        daily_pnl_by_symbol: dict[str, float],
    ) -> None:
        """Rebuild in-memory risk state after an engine restart.

        `open_positions` is
        ``(strategy_id | None, symbol, ticket, account_id)`` for every
        position currently open at the broker — the broker is the source of
        truth for what's actually live. ``strategy_id`` is None for positions
        whose magic doesn't map to a known strategy (manual trades, other EAs):
        those count toward per-symbol caps but not per-strategy ones, matching
        how `_on_closed` handles an unknown ticket.

        Daily realized P&L comes from the persisted trade log; peak equity and
        a tripped kill switch come from the persisted state row. Call once at
        startup after brokers connect — empty inputs leave a fresh start at
        defaults.
        """
        self._open_positions.clear()
        self._open_positions_by_symbol.clear()
        self._ticket_to_strategy.clear()
        for item in open_positions:
            if len(item) == 3:
                sid, symbol, ticket = item
                account_id = "default"
            else:
                sid, symbol, ticket, account_id = item
            self._open_positions_by_symbol[symbol] = (
                self._open_positions_by_symbol.get(symbol, 0) + 1
            )
            if sid is not None:
                self._open_positions[sid] = self._open_positions.get(sid, 0) + 1
                self._ticket_to_strategy[(account_id, ticket)] = sid

        self._daily_realized_pnl = daily_realized_pnl
        self._daily_pnl_by_symbol = dict(daily_pnl_by_symbol)
        self._daily_anchor_date = self._clock.now()
        self._needs_opening_balance = True

        if self._state_repo is not None:
            try:
                st = self._state_repo.load()
            except Exception:
                logger.exception("risk_state_load_failed")
                st = None
            if st is not None:
                self._peak_equity = st.peak_equity
                self._kill_switch_tripped = st.kill_switch_tripped
                self._profit_lock_tripped = st.profit_lock_tripped

        logger.info(
            "risk_rehydrated open_positions=%d strategies=%d daily_pnl=%.2f "
            "peak_equity=%s kill_switch_tripped=%s",
            len(open_positions),
            len(self._open_positions),
            self._daily_realized_pnl,
            self._peak_equity,
            self._kill_switch_tripped,
        )

    def _persist_state(self) -> None:
        """Save peak equity + kill-switch flag. No-op without a state repo;
        a DB hiccup must never break trading, so failures are logged only."""
        if self._state_repo is None:
            return
        try:
            self._state_repo.save(
                peak_equity=self._peak_equity,
                kill_switch_tripped=self._kill_switch_tripped,
                profit_lock_tripped=self._profit_lock_tripped,
            )
        except Exception:
            logger.exception("risk_state_persist_failed")

    # --- Public API --------------------------------------------------------

    def check_signal(self, signal: Signal) -> RiskVerdict:
        """Return whether a signal is allowed under current risk state."""
        if self._kill_switch_tripped:
            return RiskVerdict(False, "kill_switch_tripped")
        if self._profit_lock_tripped:
            return RiskVerdict(False, "profit_lock_tripped")

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

        # Daily loss limit (account-wide)
        if cfg.max_daily_loss_pct > 0 and self._daily_opening_balance > 0:
            loss_pct = max(0.0, -self._daily_realized_pnl) / self._daily_opening_balance * 100
            if loss_pct >= cfg.max_daily_loss_pct:
                return RiskVerdict(
                    False,
                    f"max_daily_loss_pct={cfg.max_daily_loss_pct} hit (today loss {loss_pct:.2f}%)",
                )

        # Per-symbol checks
        sym_cfg = cfg.per_symbol.get(signal.symbol)
        if sym_cfg is not None:
            if sym_cfg.max_open_positions > 0:
                sym_open = self._open_positions_by_symbol.get(signal.symbol, 0)
                if sym_open >= sym_cfg.max_open_positions:
                    return RiskVerdict(
                        False,
                        f"per_symbol.max_open_positions={sym_cfg.max_open_positions} reached "
                        f"({signal.symbol} has {sym_open} open)",
                    )
            if sym_cfg.max_daily_loss_usd > 0:
                sym_pnl = self._daily_pnl_by_symbol.get(signal.symbol, 0.0)
                sym_loss = max(0.0, -sym_pnl)
                if sym_loss >= sym_cfg.max_daily_loss_usd:
                    return RiskVerdict(
                        False,
                        f"per_symbol.max_daily_loss_usd={sym_cfg.max_daily_loss_usd} hit "
                        f"({signal.symbol} today loss ${sym_loss:.2f})",
                    )

        return RiskVerdict(True)

    def current_equity(self) -> float | None:
        """Latest equity from AccountSnapshotEvent, or None before the first
        snapshot. Used by the OrderRouter for risk-based position sizing."""
        return self._current_equity

    def position_sizing(self) -> PositionSizingConfig:
        """Account-level sizing config (stays current across update_config)."""
        return self._cfg.position_sizing

    def trading_filter(self) -> TradingFilterConfig:
        """Account-level pre-trade filter config (current across update_config)."""
        return self._cfg.trading_filter

    def snapshot(self) -> dict[str, object]:
        """Debug / UI accessor."""
        return {
            "kill_switch_tripped": self._kill_switch_tripped,
            "profit_lock_tripped": self._profit_lock_tripped,
            "current_equity": self._current_equity,
            "current_balance": self._current_balance,
            "peak_equity": self._peak_equity,
            "drawdown_pct": self._drawdown_pct(),
            "daily_opening_balance": self._daily_opening_balance,
            "daily_realized_pnl": self._daily_realized_pnl,
            "open_positions": dict(self._open_positions),
            "open_positions_by_symbol": dict(self._open_positions_by_symbol),
            "daily_pnl_by_symbol": dict(self._daily_pnl_by_symbol),
        }

    # --- Internals ---------------------------------------------------------

    def _drawdown_pct(self) -> float:
        if self._peak_equity is None or self._current_equity is None:
            return 0.0
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - self._current_equity) / self._peak_equity * 100)

    def _maybe_roll_daily(self) -> None:
        now = self._clock.now()
        if self._daily_anchor_date is None or now.date() != self._daily_anchor_date.date():
            self._daily_anchor_date = now
            if self._current_balance is not None:
                self._daily_opening_balance = self._current_balance
            self._daily_realized_pnl = 0.0
            self._daily_pnl_by_symbol.clear()

    # --- Event handlers ----------------------------------------------------

    async def _on_filled(self, evt: OrderFilledEvent) -> None:
        sid = evt.order.strategy_id
        self._open_positions[sid] = self._open_positions.get(sid, 0) + 1
        sym = evt.order.symbol
        self._open_positions_by_symbol[sym] = self._open_positions_by_symbol.get(sym, 0) + 1
        # Remember which strategy owns this ticket so the matching
        # PositionClosedEvent can decrement the right per-strategy bucket
        # (PositionClosedEvent carries only `magic`, not strategy_id).
        if evt.order.ticket:
            self._ticket_to_strategy[(evt.account_id, evt.order.ticket)] = sid

    async def _on_closed(self, evt: PositionClosedEvent) -> None:
        # Per-strategy decrement: look up the ticket recorded at fill-time
        # so we touch the strategy that actually owns this position, not
        # the first non-zero bucket. Pre-fix code decremented "first
        # non-zero bucket" which silently moved counts between strategies
        # in any multi-strategy run.
        ticket = evt.position.ticket
        sid = self._ticket_to_strategy.pop((evt.account_id, ticket), None)
        if sid is not None:
            if self._open_positions.get(sid, 0) > 0:
                self._open_positions[sid] -= 1
        else:
            # Position closed but we never saw its fill (e.g. opened
            # before RiskMonitor started, or filled on a different engine
            # instance). Skip the per-strategy decrement rather than
            # corrupting another strategy's bucket — logging makes the
            # gap visible without breaking trading.
            logger.warning(
                "risk_close_ticket_unknown ticket=%s symbol=%s magic=%s — "
                "per-strategy bucket not decremented",
                ticket, evt.position.symbol, evt.position.magic,
            )
        sym = evt.position.symbol
        if self._open_positions_by_symbol.get(sym, 0) > 0:
            self._open_positions_by_symbol[sym] -= 1
        self._maybe_roll_daily()
        self._daily_realized_pnl += evt.realized_pnl
        self._daily_pnl_by_symbol[sym] = self._daily_pnl_by_symbol.get(sym, 0.0) + evt.realized_pnl

    async def _on_partial_closed(self, evt: PartialClosedEvent) -> None:
        """Count realized P&L without decrementing open-position buckets."""
        self._maybe_roll_daily()
        sym = evt.position.symbol
        self._daily_realized_pnl += evt.realized_pnl
        self._daily_pnl_by_symbol[sym] = (
            self._daily_pnl_by_symbol.get(sym, 0.0) + evt.realized_pnl
        )

    async def _on_snapshot(self, evt: AccountSnapshotEvent) -> None:
        snap = evt.snapshot
        self._current_equity = snap.equity
        self._current_balance = snap.balance
        peak_changed = False
        if self._peak_equity is None or snap.equity > self._peak_equity:
            self._peak_equity = snap.equity
            peak_changed = True
        if self._daily_anchor_date is None:
            self._daily_anchor_date = snap.time
            self._daily_opening_balance = snap.balance
        elif self._needs_opening_balance:
            # First snapshot after a mid-day rehydrate: the current balance
            # already includes today's realized P&L, so back it out to recover
            # the day's opening balance for the loss-% gate.
            self._daily_opening_balance = snap.balance - self._daily_realized_pnl
            self._needs_opening_balance = False
        # Kill-switch check
        cfg = self._cfg
        tripped_now = False
        if (
            cfg.kill_switch_drawdown_pct > 0
            and not self._kill_switch_tripped
            and self._drawdown_pct() >= cfg.kill_switch_drawdown_pct
        ):
            self._kill_switch_tripped = True
            tripped_now = True
            logger.error(
                "risk_kill_switch_tripped drawdown_pct=%.2f limit_pct=%s "
                "peak_equity=%s current_equity=%s",
                self._drawdown_pct(),
                cfg.kill_switch_drawdown_pct,
                self._peak_equity,
                self._current_equity,
            )
        # Profit-lock check — lock in gains once up `activate_pct`, trip if the
        # account gives back more than `giveback_pct` of the gain.
        pl = cfg.profit_lock
        pl_tripped_now = False
        if pl.enabled:
            if self._pl_open_equity is None:
                self._pl_open_equity = snap.equity
                self._pl_high_equity = snap.equity
            elif self._pl_high_equity is None or snap.equity > self._pl_high_equity:
                self._pl_high_equity = snap.equity
            if (
                not self._profit_lock_tripped
                and self._pl_open_equity is not None
                and self._pl_high_equity is not None
            ):
                arm_level = self._pl_open_equity * (1.0 + pl.activate_pct / 100.0)
                if self._pl_high_equity >= arm_level:
                    gain = self._pl_high_equity - self._pl_open_equity
                    floor = self._pl_high_equity - gain * pl.giveback_pct / 100.0
                    if snap.equity < floor:
                        self._profit_lock_tripped = True
                        pl_tripped_now = True
                        logger.error(
                            "risk_profit_lock_tripped equity=%.2f floor=%.2f "
                            "open=%.2f high=%.2f",
                            snap.equity, floor, self._pl_open_equity,
                            self._pl_high_equity,
                        )
        # Persist only when something durable changed — peak ratchets up
        # rarely and a trip happens once, so write volume stays low.
        if peak_changed or tripped_now or pl_tripped_now:
            self._persist_state()
