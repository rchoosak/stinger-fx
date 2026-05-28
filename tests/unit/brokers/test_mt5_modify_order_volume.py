"""Regression tests for the MT5Broker.modify_order(volume=) hotfix.

Pre-fix bug
===========

``OrderRouter.handle_modify`` (backtest/order_router.py:210) routes pending
order modifications by calling::

    await broker.modify_order(
        ticket, sl=..., tp=..., price=..., volume=...,
    )

``SimBroker.modify_order`` accepts ``volume=``, so the backtest path was
healthy. But ``BaseBroker.modify_order`` (the ABC) and the live
``MT5Broker.modify_order`` had no ``volume`` parameter — mypy actually
flagged it:

    src/stinger_fx/backtest/order_router.py:210: error: Unexpected
    keyword argument "volume" for "modify_order" of "BaseBroker"

That means in **live** trading, the moment a strategy modified a pending
order's volume, the router would crash with::

    TypeError: modify_order() got an unexpected keyword argument 'volume'

Fix: add ``volume: float | None = None`` to ``BaseBroker.modify_order``
and implement it in ``MT5Broker.modify_order`` via ``TRADE_ACTION_MODIFY``
(which supports a ``volume`` field for pending order amendments).

These tests pin three invariants:

  1. The keyword-argument shape: ``modify_order(volume=)`` must not
     raise ``TypeError`` on MT5Broker. (Without the fix, mypy already
     caught it; this test catches it at runtime too.)
  2. The MT5 request payload for a pending order includes ``volume``,
     and that volume is the one the caller supplied.
  3. Position-side modifications cannot change ``volume`` (or ``price``)
     — MT5 doesn't allow it post-fill — so the broker rejects loudly
     rather than silently dropping the request.
"""

from __future__ import annotations

import sys
import types

import pytest

from stinger_fx.brokers.mt5.broker import MT5Broker
from stinger_fx.config.models import MT5Config
from stinger_fx.core import AsyncEventBus
from stinger_fx.domain import OrderStatus


TRADE_RETCODE_DONE = 10009


# --- helpers ---------------------------------------------------------------


class _FakeResult:
    def __init__(self, retcode: int = TRADE_RETCODE_DONE):
        self.retcode = retcode
        self.order = 99
        self.volume = 0.0
        self.price = 0.0
        self.comment = "ok"


class _FakePending:
    """Stand-in for the namedtuple-ish object MT5 returns from orders_get."""

    def __init__(
        self, ticket: int, *, price_open: float = 1.10,
        volume_current: float = 0.1, sl: float = 0.0, tp: float = 0.0,
    ):
        self.ticket = ticket
        self.price_open = price_open
        self.volume_current = volume_current
        self.sl = sl
        self.tp = tp


class _FakePosition:
    """Stand-in for the namedtuple-ish object MT5 returns from positions_get."""

    def __init__(self, ticket: int, symbol: str = "EURUSD"):
        self.ticket = ticket
        self.symbol = symbol


class _FakeMT5:
    """Minimal MT5 stand-in that captures order_send requests for assertion.

    The test populates ``pending_orders`` and ``open_positions`` to control
    which branch of modify_order fires.
    """

    TRADE_RETCODE_DONE = TRADE_RETCODE_DONE
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 6
    TRADE_ACTION_MODIFY = 7
    TRADE_ACTION_PENDING = 5
    TRADE_ACTION_REMOVE = 8
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_FILLING_IOC = 1
    ORDER_TIME_GTC = 0

    def __init__(self) -> None:
        self.pending_orders: list[_FakePending] = []
        self.open_positions: list[_FakePosition] = []
        self.send_requests: list[dict] = []
        self._connected = True

    # -- lifecycle ----------------------------------------------------------

    def initialize(self, **_kwargs) -> bool:
        return True

    def shutdown(self) -> None:
        self._connected = False

    def terminal_info(self):
        return types.SimpleNamespace(connected=True)

    def last_error(self):
        return (0, "no error")

    def symbol_select(self, _s: str, _e: bool) -> bool:
        return True

    def symbol_info_tick(self, _s: str):
        return types.SimpleNamespace(
            ask=1.1002, bid=1.1000, last=1.1001,
            time=1_700_000_000, volume=1, flags=0,
        )

    # -- orders / positions -------------------------------------------------

    def positions_get(self, ticket: int = 0, **_kwargs):
        return tuple(p for p in self.open_positions if p.ticket == ticket)

    def orders_get(self, ticket: int = 0, **_kwargs):
        return tuple(o for o in self.pending_orders if o.ticket == ticket)

    def order_send(self, request: dict):
        self.send_requests.append(dict(request))
        return _FakeResult(TRADE_RETCODE_DONE)


def _install(monkeypatch, fake: _FakeMT5) -> None:
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)


def _make_broker(bus: AsyncEventBus) -> MT5Broker:
    cfg = MT5Config(terminal_path="", login=0, password="", server="", timeout_ms=1000)
    broker = MT5Broker(bus, cfg)
    broker._health_check_interval = 60.0  # don't fire during the test
    return broker


# --- 1. The keyword-arg invariant ------------------------------------------


@pytest.mark.asyncio
async def test_modify_order_accepts_volume_keyword(monkeypatch) -> None:
    """Regression: ``broker.modify_order(ticket, volume=...)`` must not
    raise TypeError on MT5Broker.

    Pre-fix the signature was ``(ticket, *, sl, tp, price)`` — passing
    ``volume=`` raised TypeError instantly. After the fix, the broker
    accepts volume on the pending-order path."""
    fake = _FakeMT5()
    fake.pending_orders.append(_FakePending(ticket=42, price_open=1.10, volume_current=0.1))
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)
    try:
        await broker.connect()
        # The crucial call: kwarg `volume=` must be accepted.
        result = await broker.modify_order(42, volume=0.25)
        assert result.ok is True, f"modify failed: {result.message}"
    finally:
        await broker.disconnect()
        await bus.close()


# --- 2. The MT5 request payload carries volume -----------------------------


@pytest.mark.asyncio
async def test_modify_order_volume_reaches_mt5_request(monkeypatch) -> None:
    """The volume the caller supplied must end up in the MT5 order_send
    request payload — otherwise the volume modification is silent."""
    fake = _FakeMT5()
    fake.pending_orders.append(_FakePending(ticket=42, price_open=1.10, volume_current=0.1))
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)
    try:
        await broker.connect()
        await broker.modify_order(42, volume=0.33)
        # The pending-order branch sends exactly one MODIFY request.
        modify_reqs = [r for r in fake.send_requests if r.get("action") == fake.TRADE_ACTION_MODIFY]
        assert len(modify_reqs) == 1, f"expected one MODIFY request, got {fake.send_requests}"
        req = modify_reqs[0]
        assert "volume" in req, (
            f"MT5 request is missing 'volume' — broker dropped it. req={req}"
        )
        assert req["volume"] == 0.33, (
            f"expected volume=0.33 to reach MT5, got {req['volume']}"
        )
        # And the original ticket is what we wanted to modify.
        assert req["order"] == 42
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_modify_order_none_volume_carries_existing_value(monkeypatch) -> None:
    """When the caller passes ``volume=None`` (i.e. "leave alone"), MT5
    still needs a concrete volume in the request — we carry through the
    existing pending order's volume_current so MT5 doesn't reset it."""
    fake = _FakeMT5()
    fake.pending_orders.append(_FakePending(ticket=42, price_open=1.10, volume_current=0.42))
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)
    try:
        await broker.connect()
        # Modify SL only — volume should carry through unchanged.
        await broker.modify_order(42, sl=1.0950)
        modify_reqs = [r for r in fake.send_requests if r.get("action") == fake.TRADE_ACTION_MODIFY]
        assert len(modify_reqs) == 1
        req = modify_reqs[0]
        assert req["volume"] == 0.42, (
            f"existing volume not preserved when caller passed None — got {req['volume']}"
        )
        assert req["sl"] == 1.0950
    finally:
        await broker.disconnect()
        await bus.close()


# --- 3. Position modifications reject volume changes -----------------------


@pytest.mark.asyncio
async def test_modify_order_rejects_volume_on_open_position(monkeypatch) -> None:
    """An open position's volume is immutable post-fill in MT5. Attempting
    to modify it must be rejected loudly (with ok=False + REJECTED) — not
    silently dropped."""
    fake = _FakeMT5()
    fake.open_positions.append(_FakePosition(ticket=77))
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)
    try:
        await broker.connect()
        result = await broker.modify_order(77, volume=0.5)
        assert result.ok is False
        assert result.status == OrderStatus.REJECTED
        # And no actual MT5 send happened.
        assert fake.send_requests == [], (
            f"broker should not have sent any MT5 request, got {fake.send_requests}"
        )
    finally:
        await broker.disconnect()
        await bus.close()


@pytest.mark.asyncio
async def test_modify_order_sltp_only_on_open_position_still_works(monkeypatch) -> None:
    """Sanity: when only SL/TP is supplied on an open position, the broker
    still routes through TRADE_ACTION_SLTP — the volume guard only fires
    when the caller actually tries to change volume/price."""
    fake = _FakeMT5()
    fake.open_positions.append(_FakePosition(ticket=77))
    _install(monkeypatch, fake)
    bus = AsyncEventBus()
    broker = _make_broker(bus)
    try:
        await broker.connect()
        result = await broker.modify_order(77, sl=1.0900, tp=1.1200)
        assert result.ok is True
        sltp_reqs = [r for r in fake.send_requests if r.get("action") == fake.TRADE_ACTION_SLTP]
        assert len(sltp_reqs) == 1, f"expected one SLTP request, got {fake.send_requests}"
        assert sltp_reqs[0]["sl"] == 1.0900
        assert sltp_reqs[0]["tp"] == 1.1200
        assert sltp_reqs[0]["position"] == 77
    finally:
        await broker.disconnect()
        await bus.close()


# --- 4. BaseBroker ABC accepts volume in its signature ---------------------


def test_basebroker_modify_order_signature_includes_volume() -> None:
    """Structural assertion: the ABC's modify_order signature includes
    ``volume``. This is the type-system half of the fix (the runtime
    half is tested above against MT5Broker). Without this, mypy would
    keep flagging ``order_router.py:210`` and any new BaseBroker
    implementation could silently re-drop the parameter."""
    import inspect

    from stinger_fx.brokers.base import BaseBroker

    sig = inspect.signature(BaseBroker.modify_order)
    assert "volume" in sig.parameters, (
        f"BaseBroker.modify_order is missing 'volume' parameter — "
        f"OrderRouter.handle_modify will crash via this interface. "
        f"params={list(sig.parameters)}"
    )
    # And it must be keyword-only with a None default (matches sl/tp/price).
    p = sig.parameters["volume"]
    assert p.kind == inspect.Parameter.KEYWORD_ONLY
    assert p.default is None
