"""BaseBacktester — common contract for file / MT5-tester / MT4-tester."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

from stinger_fx.backtest.reports import BacktestReport
from stinger_fx.config.models import BacktestRunConfig


class Granularity(StrEnum):
    """Backtest replay granularity.

    BAR — iterate `iter_bars()` per feed, publish BarEvent (Phase 1–3 default).
    TICK — iterate `iter_ticks()` per feed, publish TickEvent; the
    BarAggregator synthesises BarEvent for the strategy as before.
    """

    BAR = "bar"
    TICK = "tick"


class BaseBacktester(ABC):
    name: str = "base"

    @abstractmethod
    async def run(self, cfg: BacktestRunConfig) -> BacktestReport: ...
