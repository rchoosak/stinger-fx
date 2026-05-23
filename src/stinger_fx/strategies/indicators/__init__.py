"""Built-in indicator helpers for strategy authors.

Single-output helpers return `float | None` (None means "not enough data yet").
Multi-output helpers (Bollinger, MACD, Stochastic, Donchian) return a typed
NamedTuple so call sites stay self-documenting.
"""

from stinger_fx.strategies.indicators.atr import atr
from stinger_fx.strategies.indicators.bollinger import BollingerBands, bollinger
from stinger_fx.strategies.indicators.donchian import DonchianChannels, donchian
from stinger_fx.strategies.indicators.macd import MACDResult, macd
from stinger_fx.strategies.indicators.moving_average import ema, sma
from stinger_fx.strategies.indicators.rsi import rsi
from stinger_fx.strategies.indicators.stochastic import StochasticResult, stochastic

__all__ = [
    "BollingerBands",
    "DonchianChannels",
    "MACDResult",
    "StochasticResult",
    "atr",
    "bollinger",
    "donchian",
    "ema",
    "macd",
    "rsi",
    "sma",
    "stochastic",
]
