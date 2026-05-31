"""Integration test for live-mode BarAggregator wiring (Plan A1).

Pre-fix bug
===========

`StingerApp._subscribe_one` (src/stinger_fx/runtime.py:359) gated
BarAggregator creation on ``if not tf.is_native_mt5 and tf.value != "TICK"``.
Since `is_native_mt5` is True for the common timeframes M1/M5/M15/M30/H1/H4,
no aggregator was ever wired for those.  And because ``BarEvent`` is
published in exactly **one** place — `BarAggregator._emit` — no live
``BarEvent`` ever reached the bus for native timeframes.

Every example strategy that just shipped (LSR / VPC / ORB) subscribes
to M1/M5/M15, so live mode was silently non-functional: tick stream
arrived but ``strategy.on_bar()`` was never called.

What this test pins
===================

After the fix, ``_subscribe_one`` wires a ``BarAggregator`` for **every**
non-TICK timeframe.  Pushing TickEvents on the bus must then produce
``BarEvent`` for every (symbol, tf) the runtime was told to subscribe.

The test instantiates a minimal ``StingerApp`` (no config loading, no
broker pool wiring) and calls ``_subscribe_one`` directly — that's the
unit under test.  A no-op fake broker satisfies the
``subscribe_bars()`` call.  TickEvents are fed through the engine bus;
emitted BarEvents are captured by a subscriber.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import BarEvent, TickEvent
from stinger_fx.domain import Tick, Timeframe
from stinger_fx.runtime import StingerApp

SYMBOL = "XAUUSD"


class _NoopBroker:
    """Minimal stand-in for MT5Broker — `_subscribe_one` only calls
    ``broker.subscribe_bars(symbol, tf)`` on it.  Everything else (tick
    polling, account info, etc.) is irrelevant for the wiring test:
    we feed TickEvents directly onto the bus."""

    def __init__(self) -> None:
        self.subscribed: list[tuple[str, Timeframe]] = []

    async def subscribe_bars(self, symbol: str, tf: Timeframe) -> None:
        self.subscribed.append((symbol, tf))


def _make_app(bus: AsyncEventBus) -> StingerApp:
    """Build a minimum StingerApp with only the fields ``_subscribe_one``
    touches: ``bus`` and ``aggregators``.  Skips ``setup()`` entirely —
    no config files, no broker pool, no risk monitor — because the unit
    under test is just the wiring helper."""
    app = StingerApp(config_dir=Path("/nonexistent"))
    app.bus = bus
    return app


def _tick(t: datetime, bid: float) -> Tick:
    return Tick(symbol=SYMBOL, time=t, bid=bid, ask=bid + 0.2)


# ---------------------------------------------------------------------- #
# Tests                                                                    #
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_native_timeframes_get_bar_aggregator() -> None:
    """The pre-fix bug: M1/M5/M15 (native MT5 timeframes) were silently
    skipped.  After the fix, every non-TICK TF gets an aggregator wired
    into ``app.aggregators`` and subscribed to TickEvent on the bus."""
    bus = AsyncEventBus()
    app = _make_app(bus)
    broker = _NoopBroker()

    for tf in (Timeframe.M1, Timeframe.M5, Timeframe.M15):
        await app._subscribe_one(broker, SYMBOL, tf)

    # All three native TFs must now have aggregators registered.
    assert (SYMBOL, Timeframe.M1) in app.aggregators
    assert (SYMBOL, Timeframe.M5) in app.aggregators
    assert (SYMBOL, Timeframe.M15) in app.aggregators
    # And the broker was asked to subscribe to bars for each.
    assert (SYMBOL, Timeframe.M1) in broker.subscribed
    assert (SYMBOL, Timeframe.M5) in broker.subscribed
    assert (SYMBOL, Timeframe.M15) in broker.subscribed
    await bus.close()


@pytest.mark.asyncio
async def test_non_native_timeframes_still_get_aggregator() -> None:
    """Backward compat: non-native MT5 TFs (M2/M3/M10/M45) still get
    aggregators wired — the pre-fix behaviour that worked for these
    must keep working."""
    bus = AsyncEventBus()
    app = _make_app(bus)
    broker = _NoopBroker()

    for tf in (Timeframe.M2, Timeframe.M3, Timeframe.M10, Timeframe.M45):
        await app._subscribe_one(broker, SYMBOL, tf)

    assert (SYMBOL, Timeframe.M2) in app.aggregators
    assert (SYMBOL, Timeframe.M3) in app.aggregators
    assert (SYMBOL, Timeframe.M10) in app.aggregators
    assert (SYMBOL, Timeframe.M45) in app.aggregators
    await bus.close()


@pytest.mark.asyncio
async def test_tick_timeframe_does_not_get_aggregator() -> None:
    """TICK-subscribed strategies consume TickEvent directly — no
    aggregator (BarAggregator rejects TICK in its constructor)."""
    bus = AsyncEventBus()
    app = _make_app(bus)
    broker = _NoopBroker()

    await app._subscribe_one(broker, SYMBOL, Timeframe.TICK)

    assert (SYMBOL, Timeframe.TICK) not in app.aggregators
    # The broker still gets asked to subscribe (so ticks flow on the bus).
    assert (SYMBOL, Timeframe.TICK) in broker.subscribed
    await bus.close()


@pytest.mark.asyncio
async def test_bar_events_emerge_when_ticks_cross_m1_boundary() -> None:
    """End-to-end wiring: after ``_subscribe_one(M1)``, pushing ticks
    that cross an M1 boundary must produce a closed BarEvent with the
    correct OHLC computed from the bid prices."""
    bus = AsyncEventBus()
    captured: list[BarEvent] = []

    async def collect(evt: BarEvent) -> None:
        captured.append(evt)

    bus.subscribe(BarEvent, collect, name="probe.bar")

    app = _make_app(bus)
    broker = _NoopBroker()
    await app._subscribe_one(broker, SYMBOL, Timeframe.M1)

    # Push three ticks inside the same M1 (12:00:00 → 12:00:59) with
    # known high/low/close, then one tick in the NEXT M1 to force the
    # boundary crossing that closes + emits the previous bar.
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    await bus.publish(TickEvent(tick=_tick(base + timedelta(seconds=10), 1900.0)))
    await bus.publish(TickEvent(tick=_tick(base + timedelta(seconds=20), 1905.0)))  # H
    await bus.publish(TickEvent(tick=_tick(base + timedelta(seconds=40), 1895.0)))  # L
    await bus.publish(TickEvent(tick=_tick(base + timedelta(seconds=55), 1902.0)))  # last in bar = close
    # Crossing tick — enters the next M1 → triggers emit for the 12:00 bar.
    await bus.publish(TickEvent(tick=_tick(base + timedelta(seconds=65), 1903.0)))

    # Drain the bus.
    for _ in range(3):
        import asyncio
        await asyncio.sleep(0)

    assert len(captured) == 1, (
        f"expected exactly one closed BarEvent after crossing the M1 "
        f"boundary; got {captured}"
    )
    bar = captured[0].bar
    assert bar.symbol == SYMBOL
    assert bar.timeframe is Timeframe.M1
    assert bar.is_closed is True
    assert bar.open == pytest.approx(1900.0)
    assert bar.high == pytest.approx(1905.0)
    assert bar.low == pytest.approx(1895.0)
    assert bar.close == pytest.approx(1902.0)
    await bus.close()


@pytest.mark.asyncio
async def test_multi_tf_subscription_produces_one_bar_per_tf_at_each_boundary() -> None:
    """A strategy that subscribes [M1, M5, M15] — after _subscribe_one
    wires three aggregators, a stream of ticks that crosses each
    boundary must produce exactly one BarEvent per TF for the just-
    closed bar."""
    bus = AsyncEventBus()
    captured: list[BarEvent] = []

    async def collect(evt: BarEvent) -> None:
        captured.append(evt)

    bus.subscribe(BarEvent, collect, name="probe.bar")

    app = _make_app(bus)
    broker = _NoopBroker()
    for tf in (Timeframe.M1, Timeframe.M5, Timeframe.M15):
        await app._subscribe_one(broker, SYMBOL, tf)

    # 16 ticks at 1-minute intervals starting at 12:00:30 — crosses an
    # M1 boundary every minute, an M5 boundary at 12:05/12:10/12:15,
    # and an M15 boundary at 12:15.
    base = datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC)
    for i in range(17):  # 0..16 minutes → 17 ticks
        await bus.publish(TickEvent(
            tick=_tick(base + timedelta(minutes=i), 1900.0 + i),
        ))

    for _ in range(5):
        import asyncio
        await asyncio.sleep(0)

    # Bucket emitted bars by TF to make the assertions readable.
    bars_by_tf: dict[Timeframe, list[BarEvent]] = {
        Timeframe.M1: [],
        Timeframe.M5: [],
        Timeframe.M15: [],
    }
    for evt in captured:
        bars_by_tf[evt.bar.timeframe].append(evt)

    # 17 ticks spanning 17 minutes → 16 M1 boundary crossings → 16
    # closed M1 bars.
    assert len(bars_by_tf[Timeframe.M1]) == 16, (
        f"M1 expected 16, got {len(bars_by_tf[Timeframe.M1])}"
    )
    # 17 minutes spans M5 boundaries at 12:05, 12:10, 12:15 → 3 closed
    # M5 bars.
    assert len(bars_by_tf[Timeframe.M5]) == 3
    # 17 minutes spans M15 boundary at 12:15 → 1 closed M15 bar.
    assert len(bars_by_tf[Timeframe.M15]) == 1
    await bus.close()
