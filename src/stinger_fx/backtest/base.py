"""BaseBacktester — common contract for file / MT5-tester / MT4-tester."""

from __future__ import annotations

from abc import ABC, abstractmethod

from stinger_fx.backtest.reports import BacktestReport
from stinger_fx.config.models import BacktestRunConfig


class BaseBacktester(ABC):
    name: str = "base"

    @abstractmethod
    async def run(self, cfg: BacktestRunConfig) -> BacktestReport: ...
