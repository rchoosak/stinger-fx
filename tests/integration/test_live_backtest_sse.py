"""Integration test for /backtest-live/* endpoints + SSE feed.

Drives the FastAPI app via `TestClient` (Starlette's in-process ASGI
driver), kicks off a tiny live backtest, and asserts that:

  * `POST /backtest-live/{id}/start` flips `/status` to `running`.
  * `GET /stream/backtest-live` (SSE) yields at least one `equity` and
    one `bar` message while the backtest runs.
  * `POST /backtest-live/{id}/stop` cleans up and `/status` returns to idle.

The test uses a 60-second sim window + speed=100, so the whole run
completes in well under a second of wall time.
"""
from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from stinger_fx.brokers.pool import BrokerPool
from stinger_fx.config.models import (
    AppConfig,
    BacktestConfig,
    BacktestRunConfig,
    BrokerConfig,
    FullConfig,
    StrategiesConfig,
    StrategyEntry,
)
from stinger_fx.core import AsyncEventBus
from stinger_fx.data import in_memory_store
from stinger_fx.data.parquet_store import ParquetStore
from stinger_fx.domain import Tick, Timeframe
from stinger_fx.ui.handle import EngineHandle
from stinger_fx.ui.web.live_backtest import LiveBacktestController
from stinger_fx.ui.web.server import create_app


@pytest.fixture
def tick_root(tmp_path: Path) -> Path:
    root = tmp_path / "parquet"
    base = datetime(2024, 1, 1, tzinfo=UTC)
    bids = [1.1000 + 0.00005 * i for i in range(180)]   # 3 minutes of 1s ticks
    ParquetStore(root).append_ticks(
        "EURUSD",
        [
            Tick(symbol="EURUSD", time=base + timedelta(seconds=i),
                 bid=b, ask=b + 2e-5)
            for i, b in enumerate(bids)
        ],
    )
    return root


def _build_app(tick_root: Path, data_dir: Path):
    cfg = FullConfig(
        app=AppConfig(broker=BrokerConfig(type="mt5")),
        strategies=StrategiesConfig(strategies=[
            StrategyEntry(
                id="ma_tick",
                class_path="stinger_fx.strategies.examples.ma_crossover:MACrossover",
                enabled=True,
                params={
                    "symbol": "EURUSD", "timeframe": "M1",
                    "fast": 2, "slow": 5, "volume": 0.1,
                },
            ),
        ]),
        backtest=BacktestConfig(runs=[
            BacktestRunConfig(
                id="ut",
                mode="file",
                strategy_id="ma_tick",
                symbol="EURUSD",
                timeframe=Timeframe.M1,
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=3),
                initial_balance=10_000.0,
                granularity="tick",
                data_source=tick_root,
            ),
        ]),
    )
    handle = EngineHandle(bus=AsyncEventBus(), brokers=BrokerPool())
    app = create_app(handle, data_dir=data_dir, sqlite_store=in_memory_store())
    app.state.live_bt_controller = LiveBacktestController(
        handle=handle,
        cfg=cfg,
        sqlite_store=in_memory_store(),
        report_dir=data_dir / "backtests",
    )
    return app, handle


def test_status_disabled_when_no_controller(tmp_path: Path) -> None:
    """Without a controller attached, /status reports enabled=False rather
    than 500, so the UI can render a graceful disabled state."""
    handle = EngineHandle(bus=AsyncEventBus(), brokers=BrokerPool())
    app = create_app(handle, data_dir=tmp_path, sqlite_store=in_memory_store())
    # NOTE: do NOT attach a controller — default is None.
    with TestClient(app) as client:
        resp = client.get("/backtest-live/x/status")
        assert resp.status_code == 200
        assert resp.json() == {"running": False, "enabled": False}


def test_start_unknown_run_id_returns_404(tick_root: Path, tmp_path: Path) -> None:
    app, _ = _build_app(tick_root, tmp_path)
    with TestClient(app) as client:
        resp = client.post("/backtest-live/no-such-id/start?speed=10")
        assert resp.status_code == 404
        assert "no backtest run" in resp.json()["detail"]


def test_start_then_stop_lifecycle(tick_root: Path, tmp_path: Path) -> None:
    """Full happy path: start → /status running → /stop → /status idle."""
    app, _ = _build_app(tick_root, tmp_path)
    with TestClient(app) as client:
        # Pre-start: idle
        st = client.get("/backtest-live/ut/status").json()
        assert st == {"running": False, "run_id": None, "speed": None, "enabled": True}

        # Start (low speed so the task is still alive when we poll).
        resp = client.post("/backtest-live/ut/start?speed=1")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["running"] is True
        assert body["run_id"] == "ut"
        assert body["speed"] == 1.0

        # Stop — must return cleanly.
        resp = client.post("/backtest-live/ut/stop")
        assert resp.status_code == 200
        assert resp.json() == {"running": False, "stopped": "ut"}

        # Post-stop: idle again, ready for another run.
        st = client.get("/backtest-live/ut/status").json()
        assert st["running"] is False
        assert st["enabled"] is True


def test_double_start_returns_409(tick_root: Path, tmp_path: Path) -> None:
    app, _ = _build_app(tick_root, tmp_path)
    with TestClient(app) as client:
        r1 = client.post("/backtest-live/ut/start?speed=1")
        assert r1.status_code == 200
        r2 = client.post("/backtest-live/ut/start?speed=10")
        assert r2.status_code == 409
        assert "already running" in r2.json()["detail"]
        # cleanup
        client.post("/backtest-live/ut/stop")


def test_page_html_renders(tick_root: Path, tmp_path: Path) -> None:
    """/backtest-live/<id> renders the page template even for an id that
    hasn't been started yet — Start button is what kicks it off."""
    app, _ = _build_app(tick_root, tmp_path)
    with TestClient(app) as client:
        resp = client.get("/backtest-live/ut")
        assert resp.status_code == 200
        body = resp.text
        # Sanity: key DOM hooks the JS depends on are present.
        assert 'id="btn-start"' in body
        assert 'id="btn-stop"' in body
        assert 'id="equity-chart"' in body
        assert 'id="candle-chart"' in body


def test_sse_stream_delivers_events_during_run(
    tick_root: Path, tmp_path: Path,
) -> None:
    """End-to-end: subscribe to SSE in a real uvicorn (needed because the
    in-process TestClient doesn't support streaming bodies properly for
    SSE), trigger a run, assert we see at least one ``equity`` event."""
    app, _ = _build_app(tick_root, tmp_path)

    # Pick a free port by binding ephemerally.
    import socket
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="error",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait for the server to be listening.
    import time as _time
    deadline = _time.monotonic() + 5.0
    while _time.monotonic() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/backtest-live/ut/status",
                          timeout=0.5)
            if r.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        _time.sleep(0.05)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        pytest.fail("uvicorn did not start within 5s")

    base_url = f"http://127.0.0.1:{port}"
    try:
        # Start a fast backtest first.
        r = httpx.post(f"{base_url}/backtest-live/ut/start?speed=50",
                       timeout=5.0)
        assert r.status_code == 200, r.text

        # Subscribe to the SSE stream. Collect for up to 6 seconds or until
        # we've seen what we want.
        seen_events: dict[str, int] = {}
        with httpx.stream(
            "GET",
            f"{base_url}/stream/backtest-live",
            timeout=httpx.Timeout(8.0, read=8.0),
        ) as stream:
            buf_event: str | None = None
            deadline = _time.monotonic() + 6.0
            for line in stream.iter_lines():
                if _time.monotonic() > deadline:
                    break
                if not line:
                    continue
                if line.startswith("event: "):
                    buf_event = line[len("event: "):].strip()
                elif line.startswith("data: "):
                    if buf_event and buf_event != "ping":
                        seen_events[buf_event] = seen_events.get(buf_event, 0) + 1
                    if seen_events.get("equity", 0) >= 1 and seen_events.get("bar", 0) >= 1:
                        break

        # We expect both equity samples (per-minute) and at least one
        # closed M1 bar over a 3-minute sim window.
        assert seen_events.get("equity", 0) >= 1, (
            f"expected ≥1 equity event; got {seen_events}"
        )
        # bar events depend on enough ticks crossing a minute boundary —
        # 3-minute fixture should produce a couple. If the test flakes,
        # increasing the tick range is the fix.
        assert seen_events.get("bar", 0) >= 1, (
            f"expected ≥1 bar event; got {seen_events}"
        )
    finally:
        httpx.post(f"{base_url}/backtest-live/ut/stop", timeout=5.0)
        server.should_exit = True
        thread.join(timeout=5)
