"""Backtest engines — file replay (Phase 1) + MT5 Strategy Tester (Phase 1)."""

from stinger_fx.backtest.base import BaseBacktester
from stinger_fx.backtest.file_backtester import FileBacktester
from stinger_fx.backtest.order_router import OrderRouter
from stinger_fx.backtest.replay_broker import SimBroker
from stinger_fx.backtest.reports import BacktestReport, TradeRecord

__all__ = [
    "BacktestReport",
    "BaseBacktester",
    "FileBacktester",
    "OrderRouter",
    "SimBroker",
    "TradeRecord",
]
