"""Strategy framework — BaseStrategy, runner, registry, indicators."""

from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.context import HistoryView, PositionView, StrategyContext
from stinger_fx.strategies.parameters import StrategyParams
from stinger_fx.strategies.registry import load_strategy_class, validate_params
from stinger_fx.strategies.runner import StrategyRunner, derive_magic

__all__ = [
    "BaseStrategy",
    "HistoryView",
    "PositionView",
    "StrategyContext",
    "StrategyParams",
    "StrategyRunner",
    "derive_magic",
    "load_strategy_class",
    "validate_params",
]
