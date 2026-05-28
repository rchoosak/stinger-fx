"""Regression tests for StingerApp engine assembly order.

Bug: setup() called broker.subscribe_bars() before engine.start() had
connected the broker, raising BrokerNotConnectedError on `stinger-fx run`.
Bug: BarAggregator was subscribed to its own class instead of TickEvent,
so it never received ticks even after the first bug was fixed.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime as _dt
from pathlib import Path

import pyarrow as pa
import pytest

from stinger_fx.brokers.base import BaseBroker
from stinger_fx.core import AsyncEventBus
from stinger_fx.domain import (
    AccountInfo,
    AccountSnapshot,
    Order,
    OrderRequest,
    OrderResult,
    OrderStatus,
    Position,
    SymbolInfo,
    Timeframe,
)
from stinger_fx.runtime import StingerApp


class _RecordingBroker(BaseBroker):
    """Tracks whether subscribe_bars was called while disconnected."""

    name = "recording"

    def __init__(self, bus: AsyncEventBus) -> None:
        super().__init__(bus)
        self._connected = False
        self.subscribe_calls_while_disconnected: list[tuple[str, Timeframe]] = []
        self.subscribe_calls_while_connected: list[tuple[str, Timeframe]] = []

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def is_connected(self) -> bool:
        return self._connected

    async def get_account_info(self) -> AccountInfo:
        return AccountInfo(account_id="x", broker="x", server="x", currency="USD", leverage=1)

    async def get_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            account_id="x", time=_dt.now(UTC),
            balance=10_000, equity=10_000, margin=0, free_margin=10_000,
        )

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        return SymbolInfo(
            symbol=symbol, digits=5, point=0.00001, contract_size=100_000,
            volume_min=0.01, volume_max=100, volume_step=0.01,
            currency_base="EUR", currency_profit="USD", currency_margin="USD",
        )

    async def list_symbols(self) -> list[str]:
        return ["EURUSD"]

    async def subscribe_ticks(self, symbol: str) -> None:
        if not self._connected:
            raise RuntimeError(f"subscribe_ticks called while disconnected: {symbol}")

    async def subscribe_bars(self, symbol: str, tf: Timeframe) -> None:
        target = self.subscribe_calls_while_connected if self._connected else self.subscribe_calls_while_disconnected
        target.append((symbol, tf))

    async def unsubscribe(self, symbol: str, tf: Timeframe | None = None) -> None:
        pass

    async def get_history_bars(self, symbol, tf, start, end) -> pa.Table:
        from stinger_fx.data.parquet_store import BAR_SCHEMA
        return BAR_SCHEMA.empty_table()

    async def get_history_ticks(self, symbol, start, end) -> pa.Table:
        from stinger_fx.data.parquet_store import TICK_SCHEMA
        return TICK_SCHEMA.empty_table()

    async def place_order(self, req: OrderRequest) -> OrderResult:
        return OrderResult(ok=False, status=OrderStatus.REJECTED, message="test broker")

    async def modify_order(
        self, ticket, *, sl=None, tp=None, price=None, volume=None,
    ) -> OrderResult:
        return OrderResult(ok=False, status=OrderStatus.REJECTED)

    async def close_position(self, ticket, volume=None) -> OrderResult:
        return OrderResult(ok=False, status=OrderStatus.REJECTED)

    async def cancel_order(self, ticket) -> OrderResult:
        return OrderResult(ok=False, status=OrderStatus.REJECTED)

    async def get_positions(self) -> list[Position]:
        return []

    async def get_open_orders(self) -> list[Order]:
        return []


@pytest.mark.asyncio
async def test_subscribe_bars_runs_after_broker_connects(monkeypatch, tmp_path: Path) -> None:
    """setup() must not call broker.subscribe_bars; that must happen after start()."""
    # Build a minimal config dir
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "app.yaml").write_text(
        "mode: normal\n"
        "broker:\n  type: mt5\n  mt5: {}\n"
        f"data_dir: {tmp_path / 'data'}\n"
    )
    (config_dir / "strategies.yaml").write_text(
        "strategies:\n"
        "  - id: ma\n"
        "    class_path: stinger_fx.strategies.examples.ma_crossover:MACrossover\n"
        "    enabled: true\n"
        "    params: {symbol: EURUSD, timeframe: M15}\n"
    )
    (config_dir / "backtest.yaml").write_text("runs: []\n")

    # Force the broker registry to give us a RecordingBroker instead of MT5Broker
    recording: dict[str, _RecordingBroker] = {}

    def fake_build_broker(cfg, bus):
        b = _RecordingBroker(bus)
        recording["broker"] = b
        return b

    monkeypatch.setattr("stinger_fx.runtime.build_broker", fake_build_broker)

    app = StingerApp(config_dir)
    await app.setup()
    broker = recording["broker"]
    # After setup, broker must not be connected yet AND no subscriptions should have happened.
    assert broker.subscribe_calls_while_disconnected == []
    assert broker.subscribe_calls_while_connected == []

    # Now simulate engine.start by calling broker.start (which is what TradingEngine does).
    await broker.start()
    # Manually call the wiring step (this is what run_until_signal does after engine.start).
    await app._wire_broker_subscriptions(broker)

    assert broker.subscribe_calls_while_disconnected == []
    assert ("EURUSD", Timeframe.M15) in broker.subscribe_calls_while_connected

    # Cleanup
    for runner in list(app.runners.values()):
        await runner.stop()
    await app.engine.bus.close()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_account_snapshot_is_published_to_bus(monkeypatch, tmp_path: Path) -> None:
    """The runtime should fan account snapshots out on the bus so UIs see them."""
    from stinger_fx.core.events import AccountSnapshotEvent

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "app.yaml").write_text(
        "mode: normal\n"
        "broker:\n  type: mt5\n  mt5: {}\n"
        f"data_dir: {tmp_path / 'data'}\n"
    )
    (config_dir / "strategies.yaml").write_text("strategies: []\n")
    (config_dir / "backtest.yaml").write_text("runs: []\n")

    recording: dict[str, _RecordingBroker] = {}
    monkeypatch.setattr(
        "stinger_fx.runtime.build_broker",
        lambda cfg, bus: recording.setdefault("broker", _RecordingBroker(bus)),
    )

    app = StingerApp(config_dir)
    await app.setup()

    received: list[AccountSnapshotEvent] = []

    async def collect(evt: AccountSnapshotEvent) -> None:
        received.append(evt)

    assert app.engine is not None and app.engine.bus is not None
    app.engine.bus.subscribe(AccountSnapshotEvent, collect)

    # Simulate broker connect + manually fire the scheduler job once
    await recording["broker"].start()
    await app._publish_account_snapshot()
    # Yield twice so the bus delivers to subscribers
    for _ in range(3):
        await __import__("asyncio").sleep(0)

    assert len(received) == 1
    assert received[0].snapshot.balance == 10_000

    await app.engine.bus.close()
