"""SimBroker.check_pending_bar — bar-mode pending order triggers.

Pre-fix bug
===========

``FileBacktester._replay_bars`` called ``broker.check_sl_tp(...)`` and
published ``BarEvent``, but never invoked the equivalent of
``broker.check_pending(...)``. Tick-mode replay (``_replay_ticks``) did
call ``check_pending``. Net effect:

  * Strategies that placed STOP / LIMIT orders in a **bar** backtest
    saw the orders enter ``_pending`` and stay there forever — the
    backtest equity curve never reflected any fills.
  * Tick backtests worked correctly, masking the issue if the user
    only ran tick mode.

Fix
===

Add ``SimBroker.check_pending_bar(symbol, bar_high, bar_low)`` — bar
mode lacks bid/ask, so all four pending types collapse to "did the
bar's [low, high] range include the trigger price?". Fill price:

  * STOP / STOP_LIMIT → trigger price + slippage (conservative).
  * LIMIT → trigger price exactly.

``FileBacktester._replay_bars`` now calls it after the SL/TP check and
before publishing ``BarEvent``, matching the tick-mode ordering.

These unit tests pin the new method's behaviour. End-to-end coverage
(a real backtest with pending fills appearing in the report) is in
``tests/integration/test_file_backtest_pending_bar.py``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from stinger_fx.backtest.replay_broker import SimBroker
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import OrderFilledEvent
from stinger_fx.domain import OrderRequest, OrderStatus, OrderType, Side


def _req(
    *, side: Side, type_: OrderType, price: float,
    volume: float = 0.1, coid: str = "coid-1",
) -> OrderRequest:
    return OrderRequest(
        strategy_id="s1",
        symbol="EURUSD",
        side=side,
        type=type_,
        volume=volume,
        price=price,
        client_order_id=coid,
    )


async def _drain(bus: AsyncEventBus, *, ticks: int = 3) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


# --- 1. Each pending type triggers when the bar's range covers its price -


@pytest.mark.asyncio
async def test_buy_stop_fires_when_bar_range_covers_trigger() -> None:
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.advance_clock(datetime(2024, 1, 1, tzinfo=UTC))
    sb.set_market_tick("EURUSD", 1.0995, 1.0997)  # below the stop

    await sb.place_order(_req(side=Side.BUY, type_=OrderType.STOP, price=1.1010))
    # Pending until check_pending_bar.
    assert len(await sb.get_open_orders()) == 1

    # Bar covers 1.0990–1.1050; trigger at 1.1010 is inside.
    triggered = await sb.check_pending_bar("EURUSD", bar_high=1.1050, bar_low=1.0990)

    assert len(triggered) == 1
    assert triggered[0].status == OrderStatus.FILLED
    assert triggered[0].symbol == "EURUSD"
    # And the position is now open.
    assert len(await sb.get_positions()) == 1
    # Pending is drained.
    assert len(await sb.get_open_orders()) == 0


@pytest.mark.asyncio
async def test_sell_stop_fires_when_bar_drops_to_trigger() -> None:
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.advance_clock(datetime(2024, 1, 1, tzinfo=UTC))
    sb.set_market_tick("EURUSD", 1.1010, 1.1012)  # above the stop

    await sb.place_order(_req(side=Side.SELL, type_=OrderType.STOP, price=1.0990))
    triggered = await sb.check_pending_bar("EURUSD", bar_high=1.1010, bar_low=1.0985)

    assert len(triggered) == 1


@pytest.mark.asyncio
async def test_buy_limit_fires_when_bar_dips_to_trigger() -> None:
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.advance_clock(datetime(2024, 1, 1, tzinfo=UTC))
    sb.set_market_tick("EURUSD", 1.1010, 1.1012)  # above the limit

    await sb.place_order(_req(side=Side.BUY, type_=OrderType.LIMIT, price=1.0995))
    triggered = await sb.check_pending_bar("EURUSD", bar_high=1.1015, bar_low=1.0990)

    assert len(triggered) == 1
    # LIMIT fills exactly at the limit price.
    assert triggered[0].fill_price == pytest.approx(1.0995)


@pytest.mark.asyncio
async def test_sell_limit_fires_when_bar_rises_to_trigger() -> None:
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.advance_clock(datetime(2024, 1, 1, tzinfo=UTC))
    sb.set_market_tick("EURUSD", 1.0995, 1.0997)  # below the limit

    await sb.place_order(_req(side=Side.SELL, type_=OrderType.LIMIT, price=1.1010))
    triggered = await sb.check_pending_bar("EURUSD", bar_high=1.1020, bar_low=1.0990)

    assert len(triggered) == 1
    assert triggered[0].fill_price == pytest.approx(1.1010)


# --- 2. Triggers don't fire when the bar's range misses ------------------


@pytest.mark.asyncio
async def test_no_fire_when_bar_range_below_buy_stop() -> None:
    """BUY_STOP at 1.1050; bar spans 1.0990–1.1020 — no fire."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.advance_clock(datetime(2024, 1, 1, tzinfo=UTC))
    sb.set_market_tick("EURUSD", 1.1000, 1.1002)

    await sb.place_order(_req(side=Side.BUY, type_=OrderType.STOP, price=1.1050))
    triggered = await sb.check_pending_bar("EURUSD", bar_high=1.1020, bar_low=1.0990)

    assert triggered == []
    assert len(await sb.get_open_orders()) == 1  # still pending


@pytest.mark.asyncio
async def test_no_fire_when_symbol_mismatches() -> None:
    """Bar on GBPUSD does not trigger a pending order on EURUSD."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.advance_clock(datetime(2024, 1, 1, tzinfo=UTC))
    sb.set_market_tick("EURUSD", 1.0995, 1.0997)

    await sb.place_order(_req(side=Side.BUY, type_=OrderType.STOP, price=1.1010))
    triggered = await sb.check_pending_bar("GBPUSD", bar_high=1.1050, bar_low=1.0990)

    assert triggered == []
    assert len(await sb.get_open_orders()) == 1


# --- 3. OrderFilledEvent is published on trigger -------------------------


@pytest.mark.asyncio
async def test_trigger_publishes_order_filled_event() -> None:
    """Bar-mode trigger must publish the same OrderFilledEvent that
    tick-mode would, so strategies' on_order_filled hooks fire in both
    backtest modes."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.advance_clock(datetime(2024, 1, 1, tzinfo=UTC))
    sb.set_market_tick("EURUSD", 1.0995, 1.0997)

    seen: list[OrderFilledEvent] = []

    async def _capture(evt: OrderFilledEvent) -> None:
        seen.append(evt)

    bus.subscribe(OrderFilledEvent, _capture, name="capture.fill")
    await sb.place_order(_req(side=Side.BUY, type_=OrderType.STOP, price=1.1010))
    await sb.check_pending_bar("EURUSD", bar_high=1.1050, bar_low=1.0990)
    await _drain(bus)

    assert len(seen) == 1
    assert seen[0].order.status == OrderStatus.FILLED
    assert seen[0].order.symbol == "EURUSD"


# --- 4. Multiple pendings in one bar -------------------------------------


@pytest.mark.asyncio
async def test_multiple_pendings_fire_in_one_bar() -> None:
    """A bar wide enough to cover several pending triggers must fire
    all of them in one call (mirrors tick-mode behaviour when a single
    tick crosses multiple pendings)."""
    bus = AsyncEventBus()
    sb = SimBroker(bus, initial_balance=10_000)
    sb.advance_clock(datetime(2024, 1, 1, tzinfo=UTC))
    sb.set_market_tick("EURUSD", 1.1000, 1.1002)

    await sb.place_order(_req(
        side=Side.BUY, type_=OrderType.STOP, price=1.1010, coid="bs1",
    ))
    await sb.place_order(_req(
        side=Side.SELL, type_=OrderType.STOP, price=1.0995, coid="ss1",
    ))

    # A volatile bar that engulfs both triggers.
    triggered = await sb.check_pending_bar("EURUSD", bar_high=1.1030, bar_low=1.0985)

    assert len(triggered) == 2
    assert len(await sb.get_open_orders()) == 0  # both pendings drained
    assert len(await sb.get_positions()) == 2
