"""Regression tests for two MT5Broker.close_position robustness hotfixes.

Pre-fix bugs
============

**P2 #1: AttributeError on missing tick** —
``close_position`` called ``mt5.symbol_info_tick(pos.symbol)`` then
indexed ``tick.ask`` / ``tick.bid`` immediately. When the terminal had
no current tick (just-reconnected, market closed, symbol not in
MarketWatch, broker glitch), the SDK returns ``None`` and the next
attribute access crashed with ``AttributeError: 'NoneType' object has
no attribute 'ask'``.

``place_order`` already handled this case gracefully — it returns
``OrderResult(ok=False, status=REJECTED, message="no current tick ...")``
— so two sibling methods in the same broker had inconsistent error
shapes. In live ops the close path is precisely when an unhandled
exception is most expensive: it tears down the strategy task, leaves
a position the engine "doesn't know how to close", and any close
managed by ``OCOGroupManager`` / ``TimeExitManager`` / ``ctx.close()``
crashes the runner.

**P2 #2: status=FILLED on a partial close** —
After PR #58 the broker correctly accepts ``RETCODE_DONE_PARTIAL`` and
emits ``PartialClosedEvent``, but the returned ``OrderResult.status``
was hard-coded to ``OrderStatus.FILLED`` regardless. Callers / log
sinks / audit tools that inspect ``result.status`` (rather than
subscribing to the bus) couldn't tell a full close apart from a
partial one. SQLite ``order_modifications`` mirror, CLI status output,
operator dashboards all read ``status``.

Fix
===

P2 #1: add the same ``if tick is None: return REJECTED`` guard as
``place_order``.

P2 #2: pick ``OrderStatus.FILLED`` vs ``OrderStatus.PARTIALLY_FILLED``
based on the same ``full_close`` flag the event-emission branch uses.
PartialClosedEvent ↔ PARTIALLY_FILLED, PositionClosedEvent ↔ FILLED.

These tests pin:

  1. ``symbol_info_tick`` returns None → close_position returns
     ``ok=False`` with no exception. (P2 #1 fix.)
  2. The None-path doesn't publish any event.
  3. ``RETCODE_DONE_PARTIAL`` on close → ``OrderResult.status`` is
     ``PARTIALLY_FILLED``, not ``FILLED``. (P2 #2 fix.)
  4. Full close (``RETCODE_DONE``) still surfaces as ``FILLED``.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from stinger_fx.brokers.mt5.broker import MT5Broker
from stinger_fx.config.models import MT5Config
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import (
    PartialClosedEvent,
    PositionClosedEvent,
)
from stinger_fx.domain import OrderStatus, Side
from tests._helpers import collect_into

RETCODE_DONE = 10009
RETCODE_DONE_PARTIAL = 10010


class _FakeResult:
    def __init__(self, *, retcode: int = RETCODE_DONE,
                 price: float = 0.0, volume: float = 0.0) -> None:
        self.retcode = retcode
        self.order = 999
        self.deal = 0
        self.price = price
        self.volume = volume
        self.comment = "ok"


class _FakePosition:
    def __init__(self, *, ticket: int, side: Side = Side.BUY,
                 volume: float = 0.1, open_price: float = 1.1000,
                 magic: int = 42) -> None:
        self.ticket = ticket
        self.symbol = "EURUSD"
        self.type = 0 if side is Side.BUY else 1
        self.volume = volume
        self.price_open = open_price
        self.sl = 0.0
        self.tp = 0.0
        self.magic = magic
        self.comment = "test"
        self.time = 1_700_000_000


class _FakeMT5:
    TRADE_RETCODE_DONE = RETCODE_DONE
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 6
    TRADE_ACTION_MODIFY = 7
    TRADE_ACTION_PENDING = 5
    TRADE_ACTION_REMOVE = 8
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TYPE_BUY_LIMIT = 2
    ORDER_TYPE_SELL_LIMIT = 3
    ORDER_TYPE_BUY_STOP = 4
    ORDER_TYPE_SELL_STOP = 5
    ORDER_TYPE_BUY_STOP_LIMIT = 6
    ORDER_TYPE_SELL_STOP_LIMIT = 7
    ORDER_FILLING_IOC = 1
    ORDER_TIME_GTC = 0

    def __init__(
        self, *, positions: list[_FakePosition] | None = None,
        send_retcode: int = RETCODE_DONE,
        send_price: float = 1.1100,
        send_volume: float = 0.0,
        tick_returns_none: bool = False,
    ) -> None:
        self._positions = positions or []
        self._send_retcode = send_retcode
        self._send_price = send_price
        self._send_volume = send_volume
        self._tick_returns_none = tick_returns_none
        self.send_requests: list[dict] = []
        self.connected = True

    def initialize(self, **_kwargs) -> bool:
        return True

    def shutdown(self) -> None:
        self.connected = False

    def terminal_info(self):
        return types.SimpleNamespace(connected=True)

    def last_error(self):
        return (0, "no error")

    def symbol_select(self, _s, _e):
        return True

    def symbol_info(self, _s: str):
        return types.SimpleNamespace(trade_contract_size=100_000.0)

    def symbol_info_tick(self, _s: str):
        # The exact failure mode P2 #1 is regression-testing: MT5 returns
        # None when no quote is available.
        if self._tick_returns_none:
            return None
        return types.SimpleNamespace(
            bid=1.1099, ask=1.1101, last=1.1100,
            time=1_700_000_500, volume=1, flags=0,
        )

    def positions_get(self, ticket: int = 0, **_kw):
        return tuple(p for p in self._positions if p.ticket == ticket)

    def orders_get(self, ticket: int = 0, **_kw):
        return ()

    def order_send(self, request: dict):
        self.send_requests.append(dict(request))
        echo_volume = self._send_volume or float(request.get("volume", 0.0))
        return _FakeResult(
            retcode=self._send_retcode,
            price=self._send_price,
            volume=echo_volume,
        )


def _install(monkeypatch, fake) -> None:
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)


def _make_broker(bus: AsyncEventBus) -> MT5Broker:
    cfg = MT5Config(terminal_path="", login=0, password="", server="", timeout_ms=1000)
    broker = MT5Broker(bus, cfg)
    broker._health_check_interval = 60.0
    return broker


async def _drain(bus: AsyncEventBus, *, ticks: int = 3) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


# --- P2 #1: None-tick handling --------------------------------------------


@pytest.mark.asyncio
async def test_close_position_with_no_tick_rejects_gracefully(monkeypatch) -> None:
    """Regression: pre-fix close_position crashed with AttributeError when
    symbol_info_tick returned None. Live MT5 returns None on
    just-reconnected / market-closed / symbol-not-in-MarketWatch /
    transient broker glitches.

    The close path is precisely when an unhandled exception is most
    expensive — it tears down the strategy task and orphans the position.
    """
    fake = _FakeMT5(
        positions=[_FakePosition(ticket=100, side=Side.BUY, magic=42)],
        tick_returns_none=True,
    )
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)

    try:
        await broker.connect()
        # The crucial behaviour: this MUST NOT raise.
        result = await broker.close_position(100)
        assert result.ok is False
        assert result.status == OrderStatus.REJECTED
        assert "tick" in result.message.lower(), (
            f"reject message should reference the missing tick — got "
            f"{result.message!r}"
        )
        # And order_send should NOT have been called — we bail out before
        # constructing the request.
        assert fake.send_requests == [], (
            f"close should have short-circuited before order_send; got "
            f"{fake.send_requests}"
        )
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_close_position_with_no_tick_publishes_no_events(monkeypatch) -> None:
    """The None-tick rejection must NOT publish PositionClosed or
    PartialClosed events — nothing happened at the broker."""
    fake = _FakeMT5(
        positions=[_FakePosition(ticket=100, side=Side.BUY, magic=42)],
        tick_returns_none=True,
    )
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)

    closed: list[PositionClosedEvent] = []
    partials: list[PartialClosedEvent] = []
    bus.subscribe(PositionClosedEvent, collect_into(closed))
    bus.subscribe(PartialClosedEvent, collect_into(partials))

    try:
        await broker.connect()
        await broker.close_position(100)
        await _drain(bus)
        assert closed == []
        assert partials == []
    finally:
        await broker.disconnect()
        await bus.close()


# --- P2 #2: Correct status for DONE_PARTIAL --------------------------------


@pytest.mark.asyncio
async def test_close_done_partial_returns_partially_filled_status(monkeypatch) -> None:
    """Regression: pre-fix, MT5 returning DONE_PARTIAL on a close deal
    produced ``OrderResult.status=FILLED`` even though only part of the
    position was closed. Subscribers reading ``result.status`` (CLI,
    audit, SQLite mirror) couldn't distinguish full vs partial."""
    fake = _FakeMT5(
        positions=[_FakePosition(
            ticket=100, side=Side.BUY, volume=0.2, open_price=1.1000, magic=42,
        )],
        send_retcode=RETCODE_DONE_PARTIAL,
        send_price=1.1050,
        send_volume=0.05,  # broker only closed 0.05 of requested 0.1
    )
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)

    partials: list[PartialClosedEvent] = []
    bus.subscribe(PartialClosedEvent, collect_into(partials))

    try:
        await broker.connect()
        result = await broker.close_position(100, volume=0.1)
        await _drain(bus)
        assert result.ok is True
        assert result.status is OrderStatus.PARTIALLY_FILLED, (
            f"DONE_PARTIAL on close must surface as PARTIALLY_FILLED in "
            f"OrderResult.status (not FILLED). Pre-fix this hard-coded "
            f"FILLED. Got {result.status}."
        )
        # And the event still emits with the chunk volume.
        assert len(partials) == 1
        assert partials[0].closed_volume == pytest.approx(0.05)
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_close_done_full_still_returns_filled_status(monkeypatch) -> None:
    """Regression guard: full close (DONE retcode, broker volume matches
    request) must still surface as ``status=FILLED``."""
    fake = _FakeMT5(
        positions=[_FakePosition(
            ticket=100, side=Side.BUY, volume=0.1, open_price=1.1000, magic=42,
        )],
        send_retcode=RETCODE_DONE,
        send_price=1.1050,
    )
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)

    try:
        await broker.connect()
        result = await broker.close_position(100)
        assert result.ok is True
        assert result.status is OrderStatus.FILLED, (
            f"full close (DONE) must remain FILLED; got {result.status}"
        )
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_close_done_with_partial_volume_returns_partially_filled(monkeypatch) -> None:
    """Edge case: MT5 returns DONE retcode but reports a smaller
    ``result.volume`` than requested. The existing logic computes
    full_close = filled_chunk >= pos_volume - epsilon; if the broker
    actually closed only part, the status must reflect that even
    though the retcode says DONE."""
    fake = _FakeMT5(
        positions=[_FakePosition(
            ticket=100, side=Side.BUY, volume=0.2, open_price=1.1000, magic=42,
        )],
        send_retcode=RETCODE_DONE,
        send_price=1.1050,
        send_volume=0.05,  # 0.05 actually closed; pos remains 0.15
    )
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)

    try:
        await broker.connect()
        result = await broker.close_position(100, volume=0.1)
        assert result.ok is True
        assert result.status is OrderStatus.PARTIALLY_FILLED, (
            f"broker said DONE but filled less than the position — status "
            f"must be PARTIALLY_FILLED; got {result.status}"
        )
    finally:
        await broker.disconnect()
        await bus.close()
