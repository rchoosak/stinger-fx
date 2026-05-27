"""Built-in indicator helpers for strategy authors.

Single-output helpers return `float | None` (None means "not enough data yet").
Multi-output helpers return a typed NamedTuple so call sites stay
self-documenting.
"""

from stinger_fx.strategies.indicators.adx import ADXResult, adx
from stinger_fx.strategies.indicators.aroon import AroonResult, aroon
from stinger_fx.strategies.indicators.atr import atr
from stinger_fx.strategies.indicators.bollinger import BollingerBands, bollinger
from stinger_fx.strategies.indicators.cci import cci
from stinger_fx.strategies.indicators.correlation import correlation
from stinger_fx.strategies.indicators.donchian import DonchianChannels, donchian
from stinger_fx.strategies.indicators.ehlers import (
    ehlers_decycler,
    ehlers_super_smoother,
)
from stinger_fx.strategies.indicators.heikin_ashi import (
    HeikinAshiCandle,
    heikin_ashi,
    heikin_ashi_series,
)
from stinger_fx.strategies.indicators.ichimoku import IchimokuResult, ichimoku
from stinger_fx.strategies.indicators.keltner import KeltnerChannels, keltner
from stinger_fx.strategies.indicators.macd import MACDResult, macd
from stinger_fx.strategies.indicators.moving_average import ema, sma
from stinger_fx.strategies.indicators.obv import obv
from stinger_fx.strategies.indicators.pivot_points import PivotLevels, pivot_points
from stinger_fx.strategies.indicators.psar import PSARResult, psar
from stinger_fx.strategies.indicators.rsi import rsi
from stinger_fx.strategies.indicators.stoch_rsi import StochRSIResult, stoch_rsi
from stinger_fx.strategies.indicators.stochastic import StochasticResult, stochastic
from stinger_fx.strategies.indicators.vwap import vwap_rolling, vwap_session
from stinger_fx.strategies.indicators.williams_r import williams_r

__all__ = [
    "ADXResult",
    "AroonResult",
    "BollingerBands",
    "DonchianChannels",
    "HeikinAshiCandle",
    "IchimokuResult",
    "KeltnerChannels",
    "MACDResult",
    "PSARResult",
    "PivotLevels",
    "StochRSIResult",
    "StochasticResult",
    "adx",
    "aroon",
    "atr",
    "bollinger",
    "cci",
    "correlation",
    "donchian",
    "ehlers_decycler",
    "ehlers_super_smoother",
    "ema",
    "heikin_ashi",
    "heikin_ashi_series",
    "ichimoku",
    "keltner",
    "macd",
    "obv",
    "pivot_points",
    "psar",
    "rsi",
    "sma",
    "stoch_rsi",
    "stochastic",
    "vwap_rolling",
    "vwap_session",
    "williams_r",
]
