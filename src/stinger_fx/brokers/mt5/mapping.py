"""Map between MT5 SDK constants and Stinger-Fx domain types."""

from __future__ import annotations

from stinger_fx.domain import Timeframe

# `MetaTrader5` constants are integers; we materialise the mapping at import time
# from the SDK so we don't hard-code numeric values that may differ across versions.
_TF_TO_MT5: dict[Timeframe, int] = {}
_MT5_TO_TF: dict[int, Timeframe] = {}


def _ensure_loaded() -> None:
    """Populate the maps lazily so the module imports cleanly off-Windows."""
    if _TF_TO_MT5:
        return
    try:
        import MetaTrader5 as mt5
    except ImportError as e:
        raise RuntimeError(
            "MetaTrader5 SDK not available — install with `uv sync --extra mt5` on Windows"
        ) from e

    native = {
        Timeframe.M1: mt5.TIMEFRAME_M1,
        Timeframe.M5: mt5.TIMEFRAME_M5,
        Timeframe.M15: mt5.TIMEFRAME_M15,
        Timeframe.M30: mt5.TIMEFRAME_M30,
        Timeframe.H1: mt5.TIMEFRAME_H1,
        Timeframe.H2: mt5.TIMEFRAME_H2,
        Timeframe.H4: mt5.TIMEFRAME_H4,
        Timeframe.D1: mt5.TIMEFRAME_D1,
        Timeframe.W1: mt5.TIMEFRAME_W1,
        Timeframe.MN1: mt5.TIMEFRAME_MN1,
    }
    _TF_TO_MT5.update(native)
    _MT5_TO_TF.update({v: k for k, v in native.items()})


def to_mt5(tf: Timeframe) -> int:
    _ensure_loaded()
    if tf not in _TF_TO_MT5:
        raise ValueError(
            f"timeframe {tf} is not native to MT5; use the BarAggregator to synthesize"
        )
    return _TF_TO_MT5[tf]


def from_mt5(value: int) -> Timeframe:
    _ensure_loaded()
    if value not in _MT5_TO_TF:
        raise ValueError(f"unknown MT5 timeframe value: {value}")
    return _MT5_TO_TF[value]
