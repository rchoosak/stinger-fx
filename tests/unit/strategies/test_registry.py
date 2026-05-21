from __future__ import annotations

import pytest

from stinger_fx.core.errors import StrategyError
from stinger_fx.strategies import load_strategy_class, validate_params
from stinger_fx.strategies.examples.ma_crossover import MACrossover, MACrossoverParams


def test_load_strategy_class_resolves_dotted_path() -> None:
    cls = load_strategy_class("stinger_fx.strategies.examples.ma_crossover:MACrossover")
    assert cls is MACrossover


def test_load_strategy_class_rejects_path_without_colon() -> None:
    with pytest.raises(StrategyError):
        load_strategy_class("no_colon_here")


def test_load_strategy_class_rejects_non_strategy_target() -> None:
    with pytest.raises(StrategyError):
        load_strategy_class("stinger_fx.domain:Side")  # not a BaseStrategy


def test_validate_params_round_trips_defaults() -> None:
    params = validate_params(MACrossover, {})
    assert isinstance(params, MACrossoverParams)
    assert params.fast == 10
    assert params.slow == 30


def test_validate_params_rejects_invalid_values() -> None:
    with pytest.raises(StrategyError):
        validate_params(MACrossover, {"fast": -1})
