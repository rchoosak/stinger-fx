"""End-to-end: trailing stop locks in profit during a tick-mode backtest.

Scenario: a strategy buys on the first bar, then attaches a
`TrailingStopManager(distance_pips=15)`. The tick stream pushes price up
20 pips, then crashes back to the original price. Expected outcome:

  • Without trailing → the SL was set ~10 pips below entry, the crash
    blows through it and the trade exits at a small loss.
  • With trailing    → the SL has advanced as price rose. The same crash
    triggers SL at a price *above* entry, locking in profit.

We assert: the final balance is **higher** than the initial balance.
That's the whole point of trailing — turn a wick reversal into a winning
trade. We also assert the SL fired at a price above the entry to prove
the trail moved it (rather than the trade closing at the initial SL).
"""

from __future__ import annotations

import sys
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stinger_fx.backtest import FileBacktester
from stinger_fx.config.models import BacktestRunConfig, StrategyEntry
from stinger_fx.data import in_memory_store
from stinger_fx.data.parquet_store import ParquetStore
from stinger_fx.domain import Tick, Timeframe


def _seed_ticks(
    root: Path,
    symbol: str,
    base: datetime,
    bid_series: list[float],
    *,
    spread: float = 2e-5,
    step_ms: int = 1_000,
) -> None:
    store = ParquetStore(root)
    ticks = [
        Tick(
            symbol=symbol,
            time=base + timedelta(milliseconds=step_ms * i),
            bid=b,
            ask=b + spread,
        )
        for i, b in enumerate(bid_series)
    ]
    store.append_ticks(symbol, ticks)


@pytest.mark.asyncio
async def test_trailing_locks_in_profit_on_wick_reversal(tmp_path: Path) -> None:
    # Tick path: open near 1.10, climb 20 pips to 1.1020, then crash back
    # to 1.10. With a 15-pip trail, SL moves to 1.1005 at the peak — the
    # crash trips that, exiting at ~+5 pips × 0.1 lot × 100k = +$5.
    rise = [1.1000 + 0.0001 * i for i in range(21)]   # 1.1000 → 1.1020
    fall = [rise[-1] - 0.0001 * (i + 1) for i in range(25)]  # 1.1019 → 1.0995
    bid_series = rise + fall

    root = tmp_path / "parquet"
    base = datetime(2024, 1, 1, tzinfo=UTC)
    _seed_ticks(root, "EURUSD", base, bid_series, step_ms=1_000)

    # In-test strategy: buy once on first tick (no SL/TP), attach a
    # TrailingStopManager with distance=15 pips.
    strat_dir = tmp_path / "trail_strats"
    strat_dir.mkdir()
    (strat_dir / "__init__.py").write_text("")
    (strat_dir / "trail_buyer.py").write_text(textwrap.dedent("""
        from stinger_fx.domain import Subscription, Timeframe
        from stinger_fx.strategies.base import BaseStrategy
        from stinger_fx.strategies.managers.trailing import TrailingStopManager
        from stinger_fx.strategies.parameters import StrategyParams

        class TrailBuyerParams(StrategyParams):
            symbol: str = "EURUSD"

        class TrailBuyer(BaseStrategy):
            name = "trail_buyer"
            Params = TrailBuyerParams
            opened: bool = False

            @classmethod
            def subscriptions(cls, params):
                return [Subscription(symbol=params.symbol, timeframe=Timeframe.M1)]

            async def on_start(self, ctx):
                ctx.attach_manager(TrailingStopManager(ctx, distance_pips=15))

            async def on_tick(self, ctx, tick):
                if not type(self).opened:
                    type(self).opened = True
                    # No SL — the trailing manager will install one as the
                    # market moves in our favour.
                    await ctx.buy(0.1)
    """))

    sys.path.insert(0, str(tmp_path))
    try:
        from importlib import import_module

        mod = import_module("trail_strats.trail_buyer")
        mod.TrailBuyer.opened = False

        entry = StrategyEntry(
            id="trail_test",
            class_path="trail_strats.trail_buyer:TrailBuyer",
            enabled=True,
            params={"symbol": "EURUSD"},
        )
        cfg = BacktestRunConfig(
            id="trail_run",
            mode="file",
            strategy_id="trail_test",
            symbol="EURUSD",
            timeframe=Timeframe.M1,
            start=base,
            end=base + timedelta(minutes=10),
            initial_balance=10_000.0,
            granularity="tick",
        )
        bt = FileBacktester(
            strategy=entry,
            parquet_root=root,
            sqlite_store=in_memory_store(),
            report_dir=tmp_path / "reports",
        )
        report = await bt.run(cfg)

        # The trailing stop should have triggered an SL-based close
        # during the crash, locking in a positive P&L for the trade.
        # (Without trailing, the trade would end with the broker forced-
        # close at the last tick — close ~= open, so ~0 P&L.)
        assert len(report.trades) >= 1
        # Find a trade that closed at a price above the open price (BUY)
        winning_closes = [t for t in report.trades if t.close_price > t.open_price]
        assert winning_closes, (
            "expected at least one BUY trade closed above entry by the "
            "trailing stop, got: "
            + ", ".join(f"{t.open_price}->{t.close_price}" for t in report.trades)
        )
        # And final balance > initial — trailing turned a wick into a win
        assert report.final_balance > 10_000.0, (
            f"trailing stop didn't lock in profit: final={report.final_balance}"
        )
    finally:
        sys.path.remove(str(tmp_path))
