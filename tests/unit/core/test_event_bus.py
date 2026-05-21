from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from stinger_fx.core import BLOCK, DROP_OLDEST, AsyncEventBus
from stinger_fx.core.events import OrderRequestEvent, TickEvent
from stinger_fx.domain import OrderRequest, OrderType, Side, Tick


async def _tick(symbol: str = "EURUSD", price: float = 1.1) -> TickEvent:
    return TickEvent(
        tick=Tick(symbol=symbol, time=datetime.now(UTC), bid=price, ask=price + 0.0001)
    )


@pytest.mark.asyncio
async def test_publish_routes_to_typed_subscribers() -> None:
    bus = AsyncEventBus()
    received: list[str] = []

    async def handler(evt: TickEvent) -> None:
        received.append(evt.tick.symbol)

    bus.subscribe(TickEvent, handler)
    await bus.publish(await _tick("EURUSD"))
    await bus.publish(await _tick("GBPUSD"))
    await asyncio.sleep(0.05)
    await bus.close()
    assert received == ["EURUSD", "GBPUSD"]


@pytest.mark.asyncio
async def test_unsubscribed_handler_stops_receiving() -> None:
    bus = AsyncEventBus()
    received: list[str] = []

    async def handler(evt: TickEvent) -> None:
        received.append(evt.tick.symbol)

    sub = bus.subscribe(TickEvent, handler)
    await bus.publish(await _tick("EURUSD"))
    await asyncio.sleep(0.02)
    await sub.unsubscribe()
    await bus.publish(await _tick("GBPUSD"))
    await asyncio.sleep(0.02)
    await bus.close()
    assert received == ["EURUSD"]


@pytest.mark.asyncio
async def test_drop_oldest_does_not_block_publisher() -> None:
    bus = AsyncEventBus()
    blocked = asyncio.Event()
    received: list[str] = []

    async def slow_handler(evt: TickEvent) -> None:
        await blocked.wait()
        received.append(evt.tick.symbol)

    bus.subscribe(TickEvent, slow_handler, maxsize=2, overflow=DROP_OLDEST)
    for i in range(10):
        await bus.publish(await _tick(f"S{i}"))
    blocked.set()
    await asyncio.sleep(0.05)
    await bus.close()
    # Slow handler should not have received all 10 — older ones dropped.
    assert len(received) < 10


@pytest.mark.asyncio
async def test_block_policy_does_not_drop() -> None:
    bus = AsyncEventBus()
    received: list[int] = []
    started = asyncio.Event()

    async def slow_handler(evt: OrderRequestEvent) -> None:
        started.set()
        await asyncio.sleep(0.01)
        received.append(evt.request.volume)

    bus.subscribe(OrderRequestEvent, slow_handler, maxsize=1, overflow=BLOCK)

    async def publish_many() -> None:
        for i in range(5):
            await bus.publish(OrderRequestEvent(request=OrderRequest(
                strategy_id="s",
                symbol="EURUSD",
                side=Side.BUY,
                type=OrderType.MARKET,
                volume=float(i + 1),
                client_order_id=f"c{i}",
            )))

    await publish_many()
    await asyncio.sleep(0.2)
    await bus.close()
    # BLOCK policy means all 5 must arrive
    assert sorted(received) == [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.mark.asyncio
async def test_handler_exception_does_not_break_bus() -> None:
    bus = AsyncEventBus()
    good_received: list[str] = []

    async def bad(evt: TickEvent) -> None:
        raise RuntimeError("boom")

    async def good(evt: TickEvent) -> None:
        good_received.append(evt.tick.symbol)

    bus.subscribe(TickEvent, bad, name="bad")
    bus.subscribe(TickEvent, good, name="good")
    await bus.publish(await _tick("EURUSD"))
    await bus.publish(await _tick("GBPUSD"))
    await asyncio.sleep(0.05)
    await bus.close()
    assert good_received == ["EURUSD", "GBPUSD"]
