from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from stinger_fx.domain import (
    Bar,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
    Tick,
    Timeframe,
)


def test_bar_validates_ohlc_bounds() -> None:
    with pytest.raises(ValidationError):
        Bar(
            symbol="EURUSD",
            timeframe=Timeframe.M15,
            time=datetime(2024, 1, 1, tzinfo=UTC),
            open=1.10,
            high=1.05,  # high < open — should fail
            low=1.00,
            close=1.04,
        )


def test_tick_spread_and_mid() -> None:
    t = Tick(symbol="EURUSD", time=datetime.now(UTC), bid=1.10, ask=1.10010)
    assert pytest.approx(t.spread, abs=1e-9) == 0.00010
    assert pytest.approx(t.mid, abs=1e-9) == 1.10005


def test_side_sign() -> None:
    assert Side.BUY.sign == 1
    assert Side.SELL.sign == -1


def test_order_frozen_immutable() -> None:
    o = Order(
        ticket=1,
        strategy_id="s",
        symbol="EURUSD",
        side=Side.BUY,
        type=OrderType.MARKET,
        volume=0.1,
        status=OrderStatus.FILLED,
    )
    with pytest.raises(ValidationError):
        o.ticket = 2  # type: ignore[misc]


def test_position_magic_default() -> None:
    p = Position(
        ticket=1,
        symbol="EURUSD",
        side=Side.BUY,
        volume=0.1,
        open_price=1.10,
        open_time=datetime.now(UTC),
    )
    assert p.magic == 0
