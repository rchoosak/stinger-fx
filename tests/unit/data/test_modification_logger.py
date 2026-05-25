"""ModificationLogger — persists OrderModifiedEvent and PartialClosedEvent to SQLite."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import OrderModifiedEvent, PartialClosedEvent
from stinger_fx.data import ModificationLogger, OrderModificationRepo, in_memory_store
from stinger_fx.domain import Position, Side


def _make_position(ticket: int, magic: int, symbol: str = "EURUSD") -> Position:
    return Position(
        ticket=ticket,
        symbol=symbol,
        side=Side.BUY,
        volume=0.1,
        open_price=1.10,
        open_time=datetime(2024, 1, 1, tzinfo=UTC),
        sl=1.0980,
        tp=1.105,
        magic=magic,
    )


async def _drain(bus: AsyncEventBus, *, ticks: int = 3) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_modification_logger_records_modify() -> None:
    """OrderModifiedEvent must create a 'modify_sl_tp' row in SQLite."""
    bus = AsyncEventBus()
    store = in_memory_store()
    magic = 99
    ml = ModificationLogger(bus, store, strategy_magic={"my_strat": magic})
    await ml.start()

    pos = _make_position(ticket=1, magic=magic)
    evt = OrderModifiedEvent(position=pos, reason="trailing")
    await bus.publish(evt)
    await _drain(bus)

    repo = OrderModificationRepo(store)
    rows = repo.recent(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row.ticket == 1
    assert row.strategy_id == "my_strat"
    assert row.modification_type == "modify_sl_tp"
    assert row.new_sl == pytest.approx(1.0980)
    assert row.new_tp == pytest.approx(1.105)
    assert row.reason == "trailing"

    await ml.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_modification_logger_records_partial_close() -> None:
    """PartialClosedEvent must create a 'partial_close' row in SQLite."""
    bus = AsyncEventBus()
    store = in_memory_store()
    magic = 42
    ml = ModificationLogger(bus, store, strategy_magic={"strat_x": magic})
    await ml.start()

    pos = _make_position(ticket=5, magic=magic)
    evt = PartialClosedEvent(position=pos, closed_volume=0.05, realized_pnl=12.5, reason="manual")
    await bus.publish(evt)
    await _drain(bus)

    repo = OrderModificationRepo(store)
    rows = repo.recent(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row.ticket == 5
    assert row.strategy_id == "strat_x"
    assert row.modification_type == "partial_close"
    assert row.closed_volume == pytest.approx(0.05)
    assert row.realized_pnl == pytest.approx(12.5)
    assert row.reason == "manual"

    await ml.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_modification_logger_noop_without_store() -> None:
    """When store=None the logger must silently do nothing."""
    bus = AsyncEventBus()
    ml = ModificationLogger(bus, None, strategy_magic={"s": 1})
    await ml.start()

    pos = _make_position(ticket=1, magic=1)
    await bus.publish(OrderModifiedEvent(position=pos, reason="test"))
    await _drain(bus)
    # No assertion — just must not raise

    await ml.stop()
    await bus.close()
