"""Position-management helpers — trailing stops, break-even movers, etc.

Managers are attached to a `StrategyContext` via `ctx.attach_manager(...)`.
The runner forwards `on_tick` (and optionally `on_bar`) to every attached
manager before invoking the strategy's own hooks, so managers can preempt
the strategy's view of the market."""

from stinger_fx.strategies.managers.base import PositionManager
from stinger_fx.strategies.managers.break_even import BreakEvenMover
from stinger_fx.strategies.managers.ladder import LadderManager
from stinger_fx.strategies.managers.time_exit import TimeExitManager
from stinger_fx.strategies.managers.trailing import TrailingStopManager

__all__ = [
    "BreakEvenMover",
    "LadderManager",
    "PositionManager",
    "TimeExitManager",
    "TrailingStopManager",
]
