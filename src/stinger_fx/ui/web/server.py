"""FastAPI app factory for the web UI.

The CLI's `--mode web` runs this in the same asyncio loop as the engine.
All endpoints read state through the `EngineHandle` stashed on
`app.state.handle`; SSE endpoints subscribe directly to the engine's bus.

Templates live in `templates/`, static assets in `static/`. HTMX handles
partial swaps for non-realtime panels (account/strategies/positions polled
every few seconds); SSE streams ticks + events to the browser as they
happen.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from stinger_fx.core.event_bus import AsyncEventBus
from stinger_fx.core.events import (
    BarEvent,
    DecisionEvent,
    OrderFilledEvent,
    OrderRejectedEvent,
    StrategyStateChangedEvent,
    TickEvent,
)
from stinger_fx.ui.handle import EngineHandle

logger = logging.getLogger("stinger.ui.web")

_HERE = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(_HERE / "templates"))


def _money(value: float | None, currency: str = "USD") -> str:
    if value is None:
        return "—"
    return f"{value:,.2f} {currency}"


TEMPLATES.env.filters["money"] = _money


def create_app(
    handle: EngineHandle,
    *,
    data_dir: Path | None = None,
) -> FastAPI:
    from stinger_fx.core.events import AccountSnapshotEvent

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Bus subscription needs a running loop, so it goes here instead of
        # at app-construction time (the TestClient drives startup/shutdown
        # too, which is why this is also test-safe).
        async def _snapshot_cache(evt) -> None:
            app.state.latest_snapshot = evt.snapshot

        sub = handle.bus.subscribe(
            AccountSnapshotEvent, _snapshot_cache, name="web.account_cache"
        )
        try:
            yield
        finally:
            await sub.unsubscribe()

    app = FastAPI(
        title="Stinger-Fx",
        default_response_class=HTMLResponse,
        lifespan=lifespan,
    )
    app.state.handle = handle
    # Tiny in-memory cache for the most-recent account snapshot so partial
    # GETs don't have to await the broker every refresh.
    app.state.latest_snapshot = None
    # data_dir lets the backtest views resolve <run_id>_trades.json etc.
    app.state.data_dir = Path(data_dir or "./data")

    app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

    # --- Routes -----------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return TEMPLATES.TemplateResponse(
            request=request, name="dashboard.html", context={}
        )

    @app.get("/partial/account", response_class=HTMLResponse)
    async def partial_account(request: Request):
        account = await handle.get_account()
        return TEMPLATES.TemplateResponse(
            request=request,
            name="partials/account.html",
            context={"account": account, "snapshot": app.state.latest_snapshot},
        )

    @app.get("/partial/strategies", response_class=HTMLResponse)
    async def partial_strategies(request: Request):
        return TEMPLATES.TemplateResponse(
            request=request,
            name="partials/strategies.html",
            context={"strategies": await handle.list_strategies()},
        )

    @app.get("/partial/positions", response_class=HTMLResponse)
    async def partial_positions(request: Request):
        return TEMPLATES.TemplateResponse(
            request=request,
            name="partials/positions.html",
            context={"positions": await handle.get_positions()},
        )

    @app.post("/strategy/{sid}/pause", response_class=HTMLResponse)
    async def pause_strategy(sid: str, request: Request):
        try:
            await handle.pause_strategy(sid)
        except KeyError as e:
            raise HTTPException(404, f"unknown strategy {sid}") from e
        return TEMPLATES.TemplateResponse(
            request=request,
            name="partials/strategies.html",
            context={"strategies": await handle.list_strategies()},
        )

    @app.post("/strategy/{sid}/resume", response_class=HTMLResponse)
    async def resume_strategy(sid: str, request: Request):
        try:
            await handle.resume_strategy(sid)
        except KeyError as e:
            raise HTTPException(404, f"unknown strategy {sid}") from e
        return TEMPLATES.TemplateResponse(
            request=request,
            name="partials/strategies.html",
            context={"strategies": await handle.list_strategies()},
        )

    # --- SSE streams ------------------------------------------------------

    @app.get("/stream/events")
    async def stream_events(request: Request):
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=200)

        async def publish_line(event_class, formatter) -> None:
            async def handler(evt) -> None:
                line = formatter(evt)
                if queue.full():
                    # drop the oldest to keep up with bursty publishers
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await queue.put(line)

            handle.bus.subscribe(event_class, handler, name=f"web.sse.{event_class.__name__}")

        await publish_line(
            OrderFilledEvent,
            lambda e: _li(
                "info",
                f"order_filled strategy={e.order.strategy_id} symbol={e.order.symbol} "
                f"side={e.order.side.value} vol={e.order.volume} price={e.order.fill_price}",
            ),
        )
        await publish_line(
            OrderRejectedEvent,
            lambda e: _li(
                "warning",
                f"order_rejected strategy={e.order.strategy_id} symbol={e.order.symbol} "
                f"reason={e.reason}",
            ),
        )
        await publish_line(
            DecisionEvent,
            lambda e: _li(
                "warning",
                f"signal_rejected strategy={e.decision.signal.strategy_id} "
                f"reason={e.decision.reason}",
            )
            if e.decision.action == "rejected"
            else None,
        )
        await publish_line(
            StrategyStateChangedEvent,
            lambda e: _li("info", f"strategy_state id={e.strategy_id} state={e.state}"),
        )
        await publish_line(
            BarEvent,
            lambda e: _li(
                "info",
                f"bar_closed {e.bar.symbol}@{e.bar.timeframe.value} close={e.bar.close:.5f}",
            )
            if e.bar.is_closed
            else None,
        )

        async def gen():
            try:
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        line = await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        # heartbeat to keep the connection alive
                        yield {"event": "ping", "data": ""}
                        continue
                    if line is None:
                        continue
                    yield {"event": "event", "data": line}
            except asyncio.CancelledError:
                return

        return EventSourceResponse(gen())

    @app.get("/stream/market")
    async def stream_market(request: Request):
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=10)

        async def on_tick(evt: TickEvent) -> None:
            html = (
                "<h2>Market</h2>"
                f"<dl class='kv'>"
                f"<dt>Symbol</dt><dd>{evt.tick.symbol}</dd>"
                f"<dt>Bid</dt><dd>{evt.tick.bid:.5f}</dd>"
                f"<dt>Ask</dt><dd>{evt.tick.ask:.5f}</dd>"
                f"<dt>Spread</dt><dd>{(evt.tick.ask - evt.tick.bid) * 1e5:.1f} pip</dd>"
                f"<dt>Last update</dt><dd>{datetime.now().strftime('%H:%M:%S')}</dd>"
                f"</dl>"
            )
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await queue.put(html)

        handle.bus.subscribe(TickEvent, on_tick, name="web.sse.market")

        async def gen():
            try:
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        html = await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield {"event": "ping", "data": ""}
                        continue
                    yield {"event": "snapshot", "data": html}
            except asyncio.CancelledError:
                return

        return EventSourceResponse(gen())

    # --- Backtest trade-replay views --------------------------------------

    @app.get("/backtest", response_class=HTMLResponse)
    async def backtest_list(request: Request):
        runs = _list_backtest_runs(app.state.data_dir)
        return TEMPLATES.TemplateResponse(
            request=request, name="backtest_list.html", context={"runs": runs}
        )

    @app.get("/backtest/{run_id}", response_class=HTMLResponse)
    async def backtest_replay(run_id: str, request: Request):
        data = _load_replay_data(app.state.data_dir, run_id)
        if data is None:
            raise HTTPException(404, f"no backtest run with id {run_id!r}")
        return TEMPLATES.TemplateResponse(
            request=request,
            name="backtest_replay.html",
            context={"run_id": run_id, "meta": data["meta"]},
        )

    @app.get("/backtest/{run_id}/data.json")
    async def backtest_data(run_id: str):
        from fastapi.responses import JSONResponse

        data = _load_replay_data(app.state.data_dir, run_id)
        if data is None:
            raise HTTPException(404, f"no backtest run with id {run_id!r}")
        return JSONResponse(data)

    return app


def _li(level: str, message: str) -> str:
    ts = datetime.now().strftime("%H:%M:%S")
    safe = json.dumps(message)[1:-1]  # crude HTML-safe-ish: avoid injecting raw tags
    return f"<li class='level-{level}'>[{ts}] {safe}</li>"


# --- Backtest replay data helpers ---------------------------------------------


def _list_backtest_runs(data_dir: Path) -> list[dict]:
    """Enumerate `<data_dir>/backtests/*_trades.json` and return a list of
    summary dicts sorted by `end` descending.

    Bare metrics-only runs (no trades.json sidecar) are skipped so the
    replay list only shows runs the view can actually render.
    """
    bt_dir = data_dir / "backtests"
    if not bt_dir.is_dir():
        return []
    out: list[dict] = []
    for path in sorted(bt_dir.glob("*_trades.json")):
        try:
            meta = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append(
            {
                "run_id": meta.get("run_id", path.stem.removesuffix("_trades")),
                "strategy_id": meta.get("strategy_id", "—"),
                "symbol": meta.get("symbol", "—"),
                "timeframe": meta.get("timeframe", "—"),
                "start": meta.get("start", ""),
                "end": meta.get("end", ""),
                "trade_count": len(meta.get("trades", [])),
                "net_pnl": round(
                    meta.get("final_balance", 0) - meta.get("initial_balance", 0), 2
                ),
            }
        )
    # Most-recently-finished first
    out.sort(key=lambda r: r["end"], reverse=True)
    return out


def _load_replay_data(data_dir: Path, run_id: str) -> dict | None:
    """Read trades + equity curve for `run_id`. Returns None if not found."""
    bt_dir = data_dir / "backtests"
    trades_path = bt_dir / f"{run_id}_trades.json"
    equity_path = bt_dir / f"{run_id}_equity.parquet"
    metrics_path = bt_dir / f"{run_id}_metrics.json"
    if not trades_path.exists():
        return None
    meta = json.loads(trades_path.read_text())
    metrics: dict = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            metrics = {}
    equity: list[dict] = []
    if equity_path.exists():
        try:
            import pyarrow.parquet as pq

            tbl = pq.read_table(equity_path)
            for row in tbl.to_pylist():
                ts = row["time"]
                equity.append(
                    {
                        "time": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                        "equity": row["equity"],
                    }
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("equity curve read failed run_id=%s err=%s", run_id, e)
    return {"meta": meta, "metrics": metrics, "equity": equity}
