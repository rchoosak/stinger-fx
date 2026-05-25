"""Position-management helpers — trailing stops, break-even movers, etc.

Managers are attached to a `StrategyContext` via `ctx.attach_manager(...)`.
The runner forwards `on_tick` to every attached manager before invoking the
strategy's own `on_tick`, so managers can preempt the strategy's view of
the market (e.g. by moving an SL closer in before the strategy ever sees
the breaching tick)."""

from stinger_fx.strategies.managers.base import PositionManager
from stinger_fx.strategies.managers.break_even import BreakEvenMover
from stinger_fx.strategies.managers.trailing import TrailingStopManager

__all__ = ["BreakEvenMover", "PositionManager", "TrailingStopManager"]
