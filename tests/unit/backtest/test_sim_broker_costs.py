"""SimBroker commission + swap costs (Tier-2 backtest fidelity).

Verifies that the fill path charges commission per side and swap per night,
that the reported trade pnl is net of both, and that
``initial_balance + Σ(net pnl) == final_balance`` holds — including partial
closes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from stinger_fx.backtest.replay_broker import SimBroker, _rollover_count
from stinger_fx.core import AsyncEventBus
from stinger_fx.domain import OrderRequest, OrderStatus, OrderType, Side

CONTRACT = {"XAUUSD": 100.0}


def _req(side: Side = Side.BUY, volume: float = 1.0, symbol: str = "XAUUSD") -> OrderRequest:
    return OrderRequest(
        strategy_id="s1",
        symbol=symbol,
        side=side,
        type=OrderType.MARKET,
        volume=volume,
        client_order_id=f"coid-{symbol}-{side.value}-{volume}",
    )


# --- _rollover_count (pure helper) ----------------------------------------


def test_rollover_count_table() -> None:
    d1 = datetime(2024, 1, 1, tzinfo=UTC)
    h = 21
    # same day, never reaches rollover hour → 0
    assert _rollover_count(d1.replace(hour=10), d1.replace(hour=15), h) == 0
    # crosses the 21:00 boundary once
    assert _rollover_count(d1.replace(hour=10), d1.replace(hour=22), h) == 1
    # close exactly at the boundary is included (half-open lower bound)
    assert _rollover_count(d1.replace(hour=10), d1.replace(hour=21), h) == 1
    # open exactly at the boundary is NOT counted at open instant
    nxt = d1 + timedelta(days=1)
    assert _rollover_count(d1.replace(hour=21), nxt.replace(hour=22), h) == 1
    # multi-day: 21:00 on Jan 1, 2, 3 → 3 nights
    d4 = datetime(2024, 1, 4, 2, tzinfo=UTC)
    assert _rollover_count(d1.replace(hour=10), d4, h) == 3
    # reversed / equal → 0
    assert _rollover_count(d1.replace(hour=15), d1.replace(hour=10), h) == 0


# --- commission (per side) -------------------------------------------------


@pytest.mark.asyncio
async def test_commission_charged_both_sides_and_pnl_is_net() -> None:
    bus = AsyncEventBus()
    broker = SimBroker(
        bus,
        initial_balance=10_000,
        symbol_contract_sizes=CONTRACT,
        commission_per_lot=3.0,  # per lot per side
    )
    try:
        broker.advance_clock(datetime(2024, 1, 1, 10, tzinfo=UTC))
        broker.set_market("XAUUSD", 2000.0)
        result = await broker.place_order(_req(volume=1.0))
        # open-side commission (3 × 1) already debited
        assert broker.balance == pytest.approx(10_000 - 3.0)

        broker.set_market("XAUUSD", 2010.0)
        broker.advance_clock(datetime(2024, 1, 1, 11, tzinfo=UTC))  # same day → no swap
        close = await broker.close_position(result.ticket or 0)
        assert close.status == OrderStatus.FILLED

        gross = (2010.0 - 2000.0) * 1.0 * 100.0  # 1000
        # balance: -3 (open) +1000 (gross) -3 (close) = 10_994
        assert broker.balance == pytest.approx(10_000 - 3.0 + gross - 3.0)

        trade = broker.trades[-1]
        assert trade.fees == pytest.approx(6.0)  # round-turn (2 × 3 × 1)
        assert trade.swap == pytest.approx(0.0)
        assert trade.pnl == pytest.approx(gross - 6.0)  # net
        # invariant
        assert broker.balance == pytest.approx(10_000 + trade.pnl)
    finally:
        await bus.close()


# --- swap (per night, signed, by side) ------------------------------------


@pytest.mark.asyncio
async def test_swap_long_charged_per_night_held() -> None:
    bus = AsyncEventBus()
    broker = SimBroker(
        bus,
        initial_balance=10_000,
        symbol_contract_sizes=CONTRACT,
        swap_long_per_lot=-2.0,  # longs pay 2/night/lot
        swap_short_per_lot=1.0,
        swap_rollover_hour_utc=21,
    )
    try:
        broker.advance_clock(datetime(2024, 1, 1, 10, tzinfo=UTC))
        broker.set_market("XAUUSD", 2000.0)
        result = await broker.place_order(_req(side=Side.BUY, volume=2.0))

        broker.set_market("XAUUSD", 2000.0)  # flat price → isolate swap
        # held to next day 10:00 → crosses one 21:00 rollover → 1 night
        broker.advance_clock(datetime(2024, 1, 2, 10, tzinfo=UTC))
        await broker.close_position(result.ticket or 0)

        trade = broker.trades[-1]
        # 1 night × -2 × 2 lots = -4
        assert trade.swap == pytest.approx(-4.0)
        assert trade.pnl == pytest.approx(-4.0)  # gross 0, no commission
        assert broker.balance == pytest.approx(10_000 - 4.0)
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_swap_short_uses_short_rate() -> None:
    bus = AsyncEventBus()
    broker = SimBroker(
        bus,
        initial_balance=10_000,
        symbol_contract_sizes=CONTRACT,
        swap_long_per_lot=-2.0,
        swap_short_per_lot=1.5,  # shorts receive 1.5/night/lot
        swap_rollover_hour_utc=21,
    )
    try:
        broker.advance_clock(datetime(2024, 1, 1, 10, tzinfo=UTC))
        broker.set_market("XAUUSD", 2000.0)
        result = await broker.place_order(_req(side=Side.SELL, volume=1.0))
        broker.set_market("XAUUSD", 2000.0)
        broker.advance_clock(datetime(2024, 1, 2, 10, tzinfo=UTC))  # 1 night
        await broker.close_position(result.ticket or 0)

        trade = broker.trades[-1]
        assert trade.swap == pytest.approx(1.5)  # credit, short rate
        assert trade.pnl == pytest.approx(1.5)
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_no_swap_when_closed_same_session() -> None:
    bus = AsyncEventBus()
    broker = SimBroker(
        bus,
        initial_balance=10_000,
        symbol_contract_sizes=CONTRACT,
        swap_long_per_lot=-99.0,
        swap_rollover_hour_utc=21,
    )
    try:
        broker.advance_clock(datetime(2024, 1, 1, 10, tzinfo=UTC))
        broker.set_market("XAUUSD", 2000.0)
        result = await broker.place_order(_req(volume=1.0))
        broker.set_market("XAUUSD", 2000.0)
        broker.advance_clock(datetime(2024, 1, 1, 15, tzinfo=UTC))  # same day
        await broker.close_position(result.ticket or 0)
        assert broker.trades[-1].swap == pytest.approx(0.0)
    finally:
        await bus.close()


# --- partial close invariant ----------------------------------------------


@pytest.mark.asyncio
async def test_partial_close_cost_invariant() -> None:
    bus = AsyncEventBus()
    broker = SimBroker(
        bus,
        initial_balance=10_000,
        symbol_contract_sizes=CONTRACT,
        commission_per_lot=3.0,
    )
    try:
        broker.advance_clock(datetime(2024, 1, 1, 10, tzinfo=UTC))
        broker.set_market("XAUUSD", 2000.0)
        result = await broker.place_order(_req(volume=2.0))
        ticket = result.ticket or 0

        broker.set_market("XAUUSD", 2010.0)
        broker.advance_clock(datetime(2024, 1, 1, 11, tzinfo=UTC))
        await broker.close_position(ticket, volume=1.0)  # partial

        broker.set_market("XAUUSD", 2020.0)
        broker.advance_clock(datetime(2024, 1, 1, 12, tzinfo=UTC))
        await broker.close_position(ticket)  # remainder

        net_sum = sum(t.pnl for t in broker.trades)
        assert broker.balance == pytest.approx(10_000 + net_sum)
        assert len(broker.trades) == 2
    finally:
        await bus.close()


# --- defaults = backward compatible ---------------------------------------


@pytest.mark.asyncio
async def test_zero_costs_match_gross() -> None:
    bus = AsyncEventBus()
    broker = SimBroker(bus, initial_balance=10_000, symbol_contract_sizes=CONTRACT)
    try:
        broker.advance_clock(datetime(2024, 1, 1, 10, tzinfo=UTC))
        broker.set_market("XAUUSD", 2000.0)
        result = await broker.place_order(_req(volume=1.0))
        assert broker.balance == pytest.approx(10_000)  # no open commission

        broker.set_market("XAUUSD", 2010.0)
        broker.advance_clock(datetime(2024, 1, 5, tzinfo=UTC))  # days later, no swap cfg
        await broker.close_position(result.ticket or 0)

        gross = (2010.0 - 2000.0) * 1.0 * 100.0
        trade = broker.trades[-1]
        assert trade.fees == pytest.approx(0.0)
        assert trade.swap == pytest.approx(0.0)
        assert trade.pnl == pytest.approx(gross)
        assert broker.balance == pytest.approx(10_000 + gross)
    finally:
        await bus.close()
