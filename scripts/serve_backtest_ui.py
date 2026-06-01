#!/usr/bin/env python3
"""Standalone backtest-viewer web server (no live broker required).

`stinger-fx run --mode web` boots the FULL trading engine (subscribes the
live broker, starts strategies, attaches the order router) which on macOS
fails immediately because the `MetaTrader5` SDK is Windows-only. This
script bypasses all of that: it just builds the FastAPI app with a stub
``EngineHandle`` so the ``/backtest`` views can serve the JSON/parquet
sidecars under ``data/backtests/`` and the rows in ``data/stinger.db``.

Pages that work
  * /backtest                          — list of completed runs (from SQLite)
  * /backtest/{run_id}                 — equity curve + trades + candles
  * /backtest/{run_id}/data.json
  * /backtest/{run_id}/candles.json
  * /backtest/{run_id}/monte_carlo.json
  * /sweep, /portfolio                 — same idea (read-only)

Pages that will 500 (need a live engine, which we deliberately skip)
  * /                                  — dashboard
  * /partial/account, /partial/positions, /partial/strategies
  * /stream/events, /stream/market

Usage
    python scripts/serve_backtest_ui.py            # 127.0.0.1:8765
    python scripts/serve_backtest_ui.py --port 9000
    python scripts/serve_backtest_ui.py --host 0.0.0.0 --port 8765
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer
import uvicorn

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stinger_fx.brokers.pool import BrokerPool  # noqa: E402
from stinger_fx.config import load_all  # noqa: E402
from stinger_fx.core.event_bus import AsyncEventBus  # noqa: E402
from stinger_fx.data import SqliteStore  # noqa: E402
from stinger_fx.ui.handle import EngineHandle  # noqa: E402
from stinger_fx.ui.web.live_backtest import LiveBacktestController  # noqa: E402
from stinger_fx.ui.web.server import create_app  # noqa: E402


app = typer.Typer(add_completion=False)


@app.command()
def main(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    data_dir: Path = typer.Option(
        Path("data"), "--data-dir",
        help="Directory holding backtests/ and stinger.db",
    ),
    config_dir: Path = typer.Option(
        Path("config"), "--config-dir", "-c",
        help="YAML config dir — used to look up live-backtest run ids",
    ),
) -> None:
    """Start a read-only backtest-viewer web UI."""
    sqlite_path = data_dir / "stinger.db"
    if not sqlite_path.exists():
        typer.echo(
            f"ERROR: {sqlite_path} not found. Run a backtest first:\n"
            f"  uv run stinger-fx backtest run --run-id xauusd_may2026",
            err=True,
        )
        raise typer.Exit(code=1)

    sqlite_store = SqliteStore(sqlite_path)
    # Empty BrokerPool — pages that require a live broker will 500, but
    # the /backtest/* and /backtest-live/* pages do not touch it.
    handle = EngineHandle(
        bus=AsyncEventBus(),
        brokers=BrokerPool(),
    )
    fastapi_app = create_app(
        handle,
        data_dir=data_dir,
        sqlite_store=sqlite_store,
    )

    # Attach the live-backtest controller so /backtest-live/* endpoints
    # work. The controller spawns the backtester on the SAME asyncio loop
    # as the FastAPI server + shares the bus, so SSE clients see every
    # event the replay publishes.
    cfg = load_all(config_dir)
    fastapi_app.state.live_bt_controller = LiveBacktestController(
        handle=handle,
        cfg=cfg,
        sqlite_store=sqlite_store,
        report_dir=data_dir / "backtests",
    )

    typer.echo(f"backtest viewer → http://{host}:{port}/backtest")
    typer.echo(f"live backtest  → http://{host}:{port}/backtest-live/<run_id>")
    typer.echo("(dashboard / live tick stream will 500 — no engine attached)")
    # Configure Python's root logger BEFORE uvicorn.run so live-backtest
    # status messages (and any backtester errors) actually surface.
    # uvicorn's own access log stays at WARNING via its log_config so
    # we don't drown in request lines.
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(fastapi_app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()
