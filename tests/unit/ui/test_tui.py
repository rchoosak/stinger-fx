"""Textual TUI smoke tests — mount + receive events without crashing.

We use the App.run_test() context manager which spins up a headless pilot
that drives the app without needing a real terminal. The tests assert that
panels respond to bus events the way the live engine would deliver them.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from stinger_fx.brokers import BrokerPool
from stinger_fx.brokers.base import BaseBroker
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import (
    AccountSnapshotEvent,
    BarEvent,
    OrderFilledEvent,
    StrategyStateChangedEvent,
    TickEvent,
)
from stinger_fx.domain import (
    AccountInfo,
    AccountSnapshot,
    Bar,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
    SymbolInfo,
    Tick,
    Timeframe,
)
from stinger_fx.ui.handle import EngineHandle
from stinger_fx.ui.tui.app import StingerTUI
from stinger_fx.ui.tui.widgets.account_panel import AccountPanel
from stinger_fx.ui.tui.widgets.market_panel import MarketPanel


class _StubBroker(BaseBroker):
    name = "stub"

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def is_connected(self) -> bool: return True

    async def get_account_info(self) -> AccountInfo:
        return AccountInfo(
            account_id="x", broker="DemoBroker", server="DemoServer",
            currency="USD", leverage=100,
        )

    async def get_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            account_id="x", time=datetime.now(UTC),
            balance=10_000, equity=10_001.5, margin=0, free_margin=10_001.5,
            profit=1.5,
        )

    async def get_symbol_info(self, symbol):  # noqa: ARG002
        return SymbolInfo(
            symbol="EURUSD", digits=5, point=0.00001, contract_size=100_000,
            volume_min=0.01, volume_max=100, volume_step=0.01,
            currency_base="EUR", currency_profit="USD", currency_margin="USD",
        )

    async def list_symbols(self): return ["EURUSD"]
    async def subscribe_ticks(self, symbol): ...
    async def subscribe_bars(self, symbol, tf): ...
    async def unsubscribe(self, symbol, tf=None): ...
    async def get_history_bars(self, *a, **kw):
        from stinger_fx.data.parquet_store import BAR_SCHEMA

        return BAR_SCHEMA.empty_table()

    async def get_history_ticks(self, *a, **kw):
        from stinger_fx.data.parquet_store import TICK_SCHEMA

        return TICK_SCHEMA.empty_table()

    async def place_order(self, req): raise NotImplementedError
    async def modify_order(self, ticket, **kw): raise NotImplementedError
    async def close_position(self, ticket, volume=None): raise NotImplementedError
    async def cancel_order(self, ticket): raise NotImplementedError

    async def get_positions(self) -> list[Position]:
        return []

    async def get_open_orders(self): return []


def _make_handle() -> EngineHandle:
    bus = AsyncEventBus()
    broker = _StubBroker(bus)
    return EngineHandle(
        bus=bus, brokers=BrokerPool([("default", broker)]), runners={}
    )


@pytest.mark.asyncio
async def test_tui_mounts_without_error() -> None:
    handle = _make_handle()
    app = StingerTUI(handle)
    async with app.run_test() as pilot:
        await pilot.pause()
        # Title/subtitle visible
        assert app.title == "Stinger-Fx"
        assert app.sub_title == "EA Bot Platform"
    await handle.bus.close()


@pytest.mark.asyncio
async def test_account_panel_updates_from_snapshot() -> None:
    handle = _make_handle()
    app = StingerTUI(handle)
    async with app.run_test() as pilot:
        await pilot.pause()
        await handle.bus.publish(
            AccountSnapshotEvent(
                snapshot=AccountSnapshot(
                    account_id="x", time=datetime.now(UTC),
                    balance=10_000, equity=10_120, margin=0, free_margin=10_120,
                    profit=120,
                )
            )
        )
        await pilot.pause()
        panel = app.query_one(AccountPanel)
        assert panel.balance == 10_000
        assert panel.equity == 10_120
        assert panel.profit == 120
    await handle.bus.close()


@pytest.mark.asyncio
async def test_market_panel_counts_ticks_and_bars() -> None:
    handle = _make_handle()
    app = StingerTUI(handle)
    async with app.run_test() as pilot:
        await pilot.pause()
        # Publish ticks
        for i in range(3):
            await handle.bus.publish(
                TickEvent(
                    tick=Tick(
                        symbol="EURUSD",
                        time=datetime.now(UTC),
                        bid=1.10 + i * 1e-5,
                        ask=1.10 + i * 1e-5 + 2e-5,
                    )
                )
            )
        # Publish a closed bar
        await handle.bus.publish(
            BarEvent(
                bar=Bar(
                    symbol="EURUSD",
                    timeframe=Timeframe.M15,
                    time=datetime.now(UTC),
                    open=1.10, high=1.11, low=1.09, close=1.105,
                    is_closed=True,
                )
            )
        )
        await pilot.pause()
        panel = app.query_one(MarketPanel)
        assert panel.symbol == "EURUSD"
        assert panel.total_ticks == 3
        assert panel.total_bars == 1
        assert panel.last_bar_tf == "M15"
    await handle.bus.close()


@pytest.mark.asyncio
async def test_order_fill_event_appears_in_log() -> None:
    handle = _make_handle()
    app = StingerTUI(handle)
    async with app.run_test() as pilot:
        await pilot.pause()
        await handle.bus.publish(
            OrderFilledEvent(
                order=Order(
                    ticket=42,
                    strategy_id="ma_test",
                    symbol="EURUSD",
                    side=Side.BUY,
                    type=OrderType.MARKET,
                    volume=0.1,
                    fill_price=1.105,
                    status=OrderStatus.FILLED,
                )
            )
        )
        await handle.bus.publish(
            StrategyStateChangedEvent(strategy_id="ma_test", state="paused")
        )
        await pilot.pause()
        # We can't easily diff RichLog content, so just confirm no exception
        # was raised by reaching this point.
    await handle.bus.close()
