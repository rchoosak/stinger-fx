"""StingerApp._audit_orphan_positions — flags broker positions not owned by any
configured strategy at startup."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from stinger_fx.brokers.base import BaseBroker
from stinger_fx.brokers.pool import BrokerPool
from stinger_fx.config.models import ReconciliationConfig, RiskConfig
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import ReconciliationMismatchEvent
from stinger_fx.domain import Position, Side
from stinger_fx.runtime import StingerApp

KNOWN_MAGIC = 111
ORPHAN_MAGIC = 999


def _pos(ticket: int, magic: int) -> Position:
    return Position(
        ticket=ticket, symbol="XAUUSD", side=Side.BUY, volume=0.1,
        open_price=2000.0, open_time=datetime(2024, 1, 1, tzinfo=UTC), magic=magic,
    )


class _PosBroker(BaseBroker):
    name = "fake"

    def __init__(self, bus, positions) -> None:
        super().__init__(bus)
        self._positions = positions

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def is_connected(self) -> bool: return True
    async def get_account_info(self): raise NotImplementedError
    async def get_account_snapshot(self): raise NotImplementedError
    async def get_symbol_info(self, symbol): raise NotImplementedError
    async def list_symbols(self): return []
    async def subscribe_ticks(self, symbol): ...
    async def subscribe_bars(self, symbol, tf): ...
    async def unsubscribe(self, symbol, tf=None): ...
    async def get_history_bars(self, *a, **kw): raise NotImplementedError
    async def get_history_ticks(self, *a, **kw): raise NotImplementedError
    async def place_order(self, req): raise NotImplementedError
    async def modify_order(self, ticket, **kw): raise NotImplementedError
    async def close_position(self, ticket, volume=None): raise NotImplementedError
    async def cancel_order(self, ticket): raise NotImplementedError
    async def get_positions(self): return list(self._positions)
    async def get_open_orders(self): return []


async def _run_audit(*, enabled: bool, startup_audit: bool = True):
    bus = AsyncEventBus()
    broker = _PosBroker(bus, [_pos(1, KNOWN_MAGIC), _pos(2, ORPHAN_MAGIC)])
    pool = BrokerPool()
    pool.add("default", broker)

    app = StingerApp(Path("/tmp"))
    app.bus = bus
    app._pool = pool
    app._router = SimpleNamespace(strategy_magic={"s1": KNOWN_MAGIC})  # type: ignore[assignment]
    app.full_cfg = SimpleNamespace(  # type: ignore[assignment]
        app=SimpleNamespace(
            risk=RiskConfig(
                reconciliation=ReconciliationConfig(
                    enabled=enabled, startup_audit=startup_audit
                )
            )
        )
    )
    events: list[ReconciliationMismatchEvent] = []
    bus.subscribe(ReconciliationMismatchEvent, lambda e: events.append(e), name="probe")

    await app._audit_orphan_positions()
    for _ in range(3):
        await asyncio.sleep(0)
    await bus.close()
    return events


@pytest.mark.asyncio
async def test_flags_only_orphan_positions() -> None:
    events = await _run_audit(enabled=True)
    assert len(events) == 1
    e = events[0]
    assert e.ticket == 2  # the ORPHAN_MAGIC position
    assert e.mismatch_type == "orphan_position"
    assert "999" in e.details


@pytest.mark.asyncio
async def test_disabled_audits_nothing() -> None:
    assert await _run_audit(enabled=False) == []
    assert await _run_audit(enabled=True, startup_audit=False) == []
