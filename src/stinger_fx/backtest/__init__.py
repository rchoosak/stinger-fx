"""Backtest engines — file replay (Phase 1) + MT5 Strategy Tester (Phase 1)."""

from stinger_fx.backtest.base import BaseBacktester, Granularity
from stinger_fx.backtest.file_backtester import FileBacktester
from stinger_fx.backtest.replay_broker import SimBroker
from stinger_fx.backtest.reports import BacktestReport, TradeRecord
from stinger_fx.backtest.sweep import (
    ParameterSweep,
    SweepCellResult,
    SweepReport,
    enumerate_grid,
)
from stinger_fx.execution import OrderRouter

__all__ = [
    "BacktestReport",
    "BaseBacktester",
    "FileBacktester",
    "Granularity",
    "OrderRouter",
    "ParameterSweep",
    "SimBroker",
    "SweepCellResult",
    "SweepReport",
    "TradeRecord",
    "enumerate_grid",
]
