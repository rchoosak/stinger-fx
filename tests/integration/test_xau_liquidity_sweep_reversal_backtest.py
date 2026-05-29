"""End-to-end multi-feed backtest for XauLiquiditySweepReversal.

The strategy needs three timeframes wired through the backtester:

  * M15 for the ADX regime gate
  * M5 for the Donchian range + ATR scale
  * M1 for the sweep-and-revert entry trigger

This file seeds synthetic XAU/USD bars at all three timeframes into a
Parquet root, hands them to ``FileBacktester`` via the multi-feed
``symbols=["XAUUSD"], timeframes=[M1, M5, M15]`` shape, and asserts:

  * In a *ranging* regime the strategy takes at least one reversal trade
    when an M1 sweep candle re-enters the range.
  * In a *trending* regime the ADX gate blocks all entries; report has
    zero trades.

The exact P&L isn't asserted — we're verifying the wiring (multi-TF
dispatch + regime gate + entry trigger + close path), not the strategy's
profitability under synthetic data.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stinger_fx.backtest import FileBacktester
from stinger_fx.config.models import BacktestRunConfig, StrategyEntry
from stinger_fx.data import in_memory_store
from stinger_fx.data.parquet_store import ParquetStore
from stinger_fx.domain import Bar, Timeframe

SYMBOL = "XAUUSD"
RANGE_HIGH = 2005.0
RANGE_LOW = 1995.0
RANGE_MID = (RANGE_HIGH + RANGE_LOW) / 2
WARMUP_HOURS = 8  # enough M15 bars (32) for ADX(14)
SWEEP_AT_MINUTE = 8 * 60 + 5  # M1 sweep at 08:05


def _bar(
    *, tf: Timeframe, t: datetime, o: float, h: float, lo: float, c: float,
    vol: int = 100,
) -> Bar:
    return Bar(
        symbol=SYMBOL, timeframe=tf, time=t,
        open=o, high=h, low=lo, close=c,
        tick_volume=vol, is_closed=True,
    )


def _ranging_m15(base: datetime, hours: int = WARMUP_HOURS + 1) -> list[Bar]:
    """Tiny-oscillation M15 — ADX stays below 22 throughout."""
    bars: list[Bar] = []
    for i in range(hours * 4):  # 4 M15 bars per hour
        c = RANGE_MID + (0.05 if i % 2 == 0 else -0.05)
        bars.append(_bar(
            tf=Timeframe.M15,
            t=base + timedelta(minutes=15 * i),
            o=RANGE_MID, h=RANGE_MID + 0.5, lo=RANGE_MID - 0.5, c=c,
        ))
    return bars


def _trending_m15(base: datetime, hours: int = WARMUP_HOURS + 1) -> list[Bar]:
    """Strong monotonic uptrend → ADX climbs above 25 → entries blocked."""
    bars: list[Bar] = []
    price = RANGE_MID
    for i in range(hours * 4):
        bars.append(_bar(
            tf=Timeframe.M15,
            t=base + timedelta(minutes=15 * i),
            o=price, h=price + 1.0, lo=price - 0.2, c=price + 0.9,
        ))
        price += 1.0
    return bars


def _ranging_m5(base: datetime, hours: int = WARMUP_HOURS + 1) -> list[Bar]:
    """Sideway M5: oscillates inside [RANGE_LOW, RANGE_HIGH] with a few
    bars touching each extreme so Donchian returns the full range."""
    bars: list[Bar] = []
    n = hours * 12  # 12 M5 bars per hour
    for i in range(n):
        if i % 12 == 0:
            h, lo, c = RANGE_HIGH, RANGE_MID, RANGE_MID + 1
        elif i % 12 == 6:
            h, lo, c = RANGE_MID, RANGE_LOW, RANGE_MID - 1
        else:
            h, lo, c = RANGE_MID + 1.0, RANGE_MID - 1.0, RANGE_MID + 0.2
        bars.append(_bar(
            tf=Timeframe.M5,
            t=base + timedelta(minutes=5 * i),
            o=RANGE_MID, h=h, lo=lo, c=c,
        ))
    return bars


def _m1_with_sweep(base: datetime, hours: int = WARMUP_HOURS + 1) -> list[Bar]:
    """M1 bars covering ``hours`` of session. At ``SWEEP_AT_MINUTE`` a
    single bar wicks above RANGE_HIGH then closes back inside — the
    canonical short-sweep setup."""
    bars: list[Bar] = []
    n = hours * 60  # 60 M1 bars per hour
    for i in range(n):
        t = base + timedelta(minutes=i)
        if i == SWEEP_AT_MINUTE:
            # Sweep candle: high 3 USD above range_high, close 3 USD
            # below range_high → both conditions of the short setup met.
            bars.append(_bar(
                tf=Timeframe.M1, t=t,
                o=RANGE_HIGH - 0.5, h=RANGE_HIGH + 3.0,
                lo=RANGE_HIGH - 4.0, c=RANGE_HIGH - 3.0,
            ))
        else:
            # Boring inside-the-range bars.
            c = RANGE_MID + 0.1 if i % 2 == 0 else RANGE_MID - 0.1
            bars.append(_bar(
                tf=Timeframe.M1, t=t,
                o=RANGE_MID, h=RANGE_MID + 0.5, lo=RANGE_MID - 0.5, c=c,
            ))
    return bars


def _seed(root: Path, m1: list[Bar], m5: list[Bar], m15: list[Bar]) -> None:
    store = ParquetStore(root)
    store.append_bars(SYMBOL, Timeframe.M1, m1)
    store.append_bars(SYMBOL, Timeframe.M5, m5)
    store.append_bars(SYMBOL, Timeframe.M15, m15)


def _build_cfg(parquet_root: Path, base: datetime) -> BacktestRunConfig:
    return BacktestRunConfig(
        id="xau_sweep_smoke",
        mode="file",
        strategy_id="xau_sweep",
        symbols=[SYMBOL],
        timeframes=[Timeframe.M1, Timeframe.M5, Timeframe.M15],
        start=base,
        end=base + timedelta(hours=WARMUP_HOURS + 1),
        data_source=parquet_root,
        initial_balance=10_000.0,
    )


def _build_entry() -> StrategyEntry:
    return StrategyEntry(
        id="xau_sweep",
        class_path=(
            "stinger_fx.strategies.examples.xau_liquidity_sweep_reversal"
            ":XauLiquiditySweepReversal"
        ),
        enabled=True,
        params={
            "symbol": SYMBOL,
            "entry_timeframe": "M1",
            "structure_timeframe": "M5",
            "regime_timeframe": "M15",
            # Tight buffers so the synthetic sweep candle triggers.
            "sweep_buffer_atr": 0.0,
            "reentry_buffer_atr": 0.0,
            "stop_buffer_atr": 0.2,
            "min_rr": 0.1,
            # Make sure session filter doesn't trip on our base time.
            "use_session_filter": False,
            "tp_mode": "mid",
        },
    )


# ----------------------------------------------------------------------
# Test cases
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_strategy_takes_reversal_trade_in_sideway(tmp_path: Path) -> None:
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    root = tmp_path / "parquet"
    _seed(root, _m1_with_sweep(base), _ranging_m5(base), _ranging_m15(base))

    bt = FileBacktester(
        strategy=_build_entry(),
        parquet_root=root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "reports",
    )
    report = await bt.run(_build_cfg(root, base))

    assert len(report.trades) >= 1, (
        f"expected at least one reversal trade in sideway regime — "
        f"got {report.trades}"
    )
    # The first trade should be the SHORT setup (sweep over range_high).
    # SimBroker records Side.SELL → trade.side == 'sell'.
    assert report.trades[0].side == "sell"


@pytest.mark.asyncio
async def test_strategy_takes_no_trades_in_trending_regime(tmp_path: Path) -> None:
    """Same M1 sweep candle, but M15 ADX climbs above max_adx → all
    entries blocked → zero trades."""
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    root = tmp_path / "parquet"
    _seed(root, _m1_with_sweep(base), _ranging_m5(base), _trending_m15(base))

    bt = FileBacktester(
        strategy=_build_entry(),
        parquet_root=root,
        sqlite_store=in_memory_store(),
        report_dir=tmp_path / "reports",
    )
    report = await bt.run(_build_cfg(root, base))

    assert report.trades == [], (
        f"trending regime must block all sweep entries; got "
        f"{report.trades}"
    )
