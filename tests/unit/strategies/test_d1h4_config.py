"""D1H4TrendParams validation — every parameter is constrained, frozen, and
rejects extras. Mirrors how the loader validates YAML params at startup."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stinger_fx.core.errors import StrategyError
from stinger_fx.domain import Subscription, Timeframe
from stinger_fx.strategies.examples.d1h4_trend import (
    D1H4TrendParams,
    D1H4TrendStrategy,
)
from stinger_fx.strategies.registry import validate_params


def test_defaults_are_valid() -> None:
    p = D1H4TrendParams()
    assert p.symbol == "XAUUSD"
    assert p.d1_slow_ema == 150 and p.chandelier_lookback == 22
    assert p.allow_short is True


def test_subscriptions_is_h1_only() -> None:
    subs = D1H4TrendStrategy.subscriptions(D1H4TrendParams())
    assert subs == [Subscription(symbol="XAUUSD", timeframe=Timeframe.H1)]


def test_warmup_declares_enough_h1_for_d1_slow_ema() -> None:
    p = D1H4TrendParams()
    warmup = D1H4TrendStrategy.warmup_bars(p)
    assert warmup is not None
    h1 = warmup[Subscription(symbol="XAUUSD", timeframe=Timeframe.H1)]
    assert h1 >= (p.d1_slow_ema + p.d1_slope_lookback) * 24


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("daily_anchor_hour", 24),
        ("daily_anchor_hour", -1),
        ("d1_fast_ema", 0),
        ("d1_slow_ema", 1),
        ("d1_slope_lookback", 0),
        ("d1_adx_length", 1),
        ("d1_long_adx_min", -1.0),
        ("d1_short_adx_min", -1.0),
        ("d1_exit_ema", 0),
        ("h4_fast_ema", 0),
        ("h4_slow_ema", 1),
        ("breakout_lookback", 0),
        ("atr_length", 1),
        ("initial_stop_atr", 0.0),
        ("max_breakout_atr", 0.0),
        ("max_channel_breakout_atr", 0.0),
        ("chandelier_lookback", 0),
        ("chandelier_atr", 0.0),
        ("volume", 0.0),
    ],
)
def test_rejects_out_of_range(field: str, bad: object) -> None:
    with pytest.raises(ValidationError):
        D1H4TrendParams(**{field: bad})  # type: ignore[arg-type]


def test_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        D1H4TrendParams(unknown_param=1)  # type: ignore[call-arg]


def test_is_frozen() -> None:
    p = D1H4TrendParams()
    with pytest.raises(ValidationError):
        p.d1_slow_ema = 200  # type: ignore[misc]


def test_validate_params_via_registry() -> None:
    p = validate_params(D1H4TrendStrategy, {"symbol": "XAUUSD", "allow_short": False})
    assert isinstance(p, D1H4TrendParams)
    assert p.allow_short is False
    with pytest.raises(StrategyError):
        validate_params(D1H4TrendStrategy, {"chandelier_atr": -1.0})
