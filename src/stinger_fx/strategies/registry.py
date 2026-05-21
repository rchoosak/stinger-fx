"""Strategy class loader.

A `class_path` is the import-form string `"pkg.module:ClassName"`. The registry
resolves it to a `BaseStrategy` subclass and validates the params dict against
that subclass's `Params` model.
"""

from __future__ import annotations

import importlib

from pydantic import ValidationError

from stinger_fx.core.errors import StrategyError
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.parameters import StrategyParams


def load_strategy_class(class_path: str) -> type[BaseStrategy]:
    if ":" not in class_path:
        raise StrategyError(f"invalid class_path {class_path!r}; expected 'module:Class'")
    module_name, class_name = class_path.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise StrategyError(f"cannot import {module_name}: {e}") from e
    cls = getattr(module, class_name, None)
    if cls is None:
        raise StrategyError(f"{module_name!r} has no attribute {class_name!r}")
    if not isinstance(cls, type) or not issubclass(cls, BaseStrategy):
        raise StrategyError(f"{class_path} is not a BaseStrategy subclass")
    return cls


def validate_params(
    strategy_cls: type[BaseStrategy], raw: dict
) -> StrategyParams:
    try:
        return strategy_cls.Params.model_validate(raw)
    except ValidationError as e:
        raise StrategyError(
            f"invalid params for {strategy_cls.__name__}:\n{e}"
        ) from e
