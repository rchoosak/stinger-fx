"""Built-in indicator helpers for strategy authors.

Single-output helpers return `float | None` (None means "not enough data yet").
Multi-output helpers (Bollinger, MACD, Stochastic, Donchian, Ichimoku,
ADX, Keltner, PivotLevels) return a typed NamedTuple so call sites stay
self-documenting.
"""

from stinger_fx.strategies.indicators.adx import ADXResult, adx
from stinger_fx.strategies.indicators.atr import atr
from stinger_fx.strategies.indicators.bollinger import BollingerBands, bollinger
from stinger_fx.strategies.indicators.cci import cci
from stinger_fx.strategies.indicators.correlation import correlation
from stinger_fx.strategies.indicators.donchian import DonchianChannels, donchian
from stinger_fx.strategies.indicators.ichimoku import IchimokuResult, ichimoku
from stinger_fx.strategies.indicators.keltner import KeltnerChannels, keltner
from stinger_fx.strategies.indicators.macd import MACDResult, macd
from stinger_fx.strategies.indicators.moving_average import ema, sma
from stinger_fx.strategies.indicators.pivot_points import PivotLevels, pivot_points
from stinger_fx.strategies.indicators.rsi import rsi
from stinger_fx.strategies.indicators.stochastic import StochasticResult, stochastic
from stinger_fx.strategies.indicators.vwap import vwap_rolling, vwap_session

__all__ = [
    "ADXResult",
    "BollingerBands",
    "DonchianChannels",
    "IchimokuResult",
    "KeltnerChannels",
    "MACDResult",
    "PivotLevels",
    "StochasticResult",
    "adx",
    "atr",
    "bollinger",
    "cci",
    "correlation",
    "donchian",
    "ema",
    "ichimoku",
    "keltner",
    "macd",
    "pivot_points",
    "rsi",
    "sma",
    "stochastic",
    "vwap_rolling",
    "vwap_session",
]
