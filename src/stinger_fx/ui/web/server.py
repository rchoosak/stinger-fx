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

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
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
    # Best-known time the engine entered service — used by /health.
    app.state.started_at = datetime.now().isoformat()

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

    # --- Strategy live params editor (Phase 5 — Batch E) -----------------

    @app.get("/strategy/{sid}/params", response_class=HTMLResponse)
    async def strategy_params_form(sid: str, request: Request):
        """Render an HTMX-driven edit form for the strategy's current params."""
        try:
            current = handle.get_strategy_params(sid)
            schema = handle.get_strategy_param_schema(sid)
        except KeyError as e:
            raise HTTPException(404, f"unknown strategy {sid}") from e
        fields = [
            {
                "name": name,
                "value": current.get(name, info["default"]),
                "type": info["type"],
                "description": info["description"],
            }
            for name, info in schema.items()
        ]
        return TEMPLATES.TemplateResponse(
            request=request,
            name="partials/strategy_params.html",
            context={"sid": sid, "fields": fields, "error": None},
        )

    @app.post("/strategy/{sid}/params", response_class=HTMLResponse)
    async def strategy_params_update(sid: str, request: Request):
        """Apply form-encoded param updates atomically via runner.update_params."""
        form = await request.form()
        try:
            schema = handle.get_strategy_param_schema(sid)
        except KeyError as e:
            raise HTTPException(404, f"unknown strategy {sid}") from e
        # Coerce raw form strings into the field's declared type. Pydantic
        # would do this anyway, but doing it here lets us surface a more
        # actionable error message when e.g. "abc" was typed into an int field.
        new_values: dict = {}
        for name, info in schema.items():
            if name not in form:
                continue
            raw = form[name]
            new_values[name] = _coerce_form_value(raw, info["type"])

        try:
            await handle.update_strategy_params(sid, new_values)
        except ValueError as e:
            # Re-render the form with the error banner.
            current = handle.get_strategy_params(sid)
            fields = [
                {
                    "name": name,
                    "value": new_values.get(name, current.get(name, info["default"])),
                    "type": info["type"],
                    "description": info["description"],
                }
                for name, info in schema.items()
            ]
            return TEMPLATES.TemplateResponse(
                request=request,
                name="partials/strategy_params.html",
                context={"sid": sid, "fields": fields, "error": str(e)},
            )

        # Success: HTMX swaps the strategies list back in.
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

    @app.get("/backtest/{run_id}/candles.json")
    async def backtest_candles(run_id: str):
        """Return OHLC bars for the run's primary symbol/timeframe.

        Used by the candlestick overlay on the replay page. Returns an empty
        list (not a 404) when bars aren't available — the chart can fall back
        to equity-only when this happens.
        """
        from fastapi.responses import JSONResponse

        candles = _load_replay_candles(app.state.data_dir, run_id)
        return JSONResponse({"candles": candles})

    # --- Control plane ---------------------------------------------------
    # Two tiny endpoints the `--detach` flow leans on. They're broker- and
    # mode-agnostic so they work the same way for any running engine.

    @app.get("/health")
    async def health():
        from fastapi.responses import JSONResponse

        from stinger_fx import __version__

        strategies = await handle.list_strategies()
        accounts: list[dict] = []
        try:
            for account_id, info in await handle.list_accounts():
                accounts.append({"account_id": account_id, "broker": info.broker})
        except Exception:  # noqa: BLE001 — multi-account introduced list_accounts
            try:
                info = await handle.get_account()
                accounts = [{"account_id": info.account_id, "broker": info.broker}]
            except Exception:  # noqa: BLE001
                accounts = []
        return JSONResponse(
            {
                "status": "ok",
                "version": __version__,
                "started_at": app.state.started_at,
                "strategies": len(strategies),
                "accounts": accounts,
            }
        )

    @app.post("/control/shutdown")
    async def shutdown(background_tasks: "BackgroundTasks"):
        """Politely terminate the engine process.

        SIGINT is what the existing run-loop already handles in its cleanup
        path, so this gives us the same graceful drain as Ctrl+C — except
        triggered by `stinger-fx stop`.

        The signal is fired from a Starlette BackgroundTask so the response
        flushes first and the loop stays alive long enough to deliver it.
        """
        import os
        import signal as _signal
        import time

        from fastapi.responses import JSONResponse

        def _send_signal() -> None:
            time.sleep(0.05)
            os.kill(os.getpid(), _signal.SIGINT)

        background_tasks.add_task(_send_signal)
        return JSONResponse({"status": "shutting_down"})

    return app


def _coerce_form_value(raw, type_name: str):
    """Best-effort string-to-type coercion for HTMX form inputs.

    Falls back to the raw string when coercion fails so Pydantic can surface
    the real validation error from `update_strategy_params`.
    """
    # Form values can also be UploadFile when a file input is in the form —
    # we never expect that here, but guard anyway.
    if not isinstance(raw, str):
        return raw
    if type_name == "int":
        try:
            return int(raw)
        except (TypeError, ValueError):
            return raw
    if type_name == "float":
        try:
            return float(raw)
        except (TypeError, ValueError):
            return raw
    if type_name == "bool":
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return raw


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


def _load_replay_candles(data_dir: Path, run_id: str) -> list[dict]:
    """Read OHLC bars for ``run_id``'s primary feed from the parquet store.

    Reads `meta.symbol`, `meta.timeframe`, `meta.start`, `meta.end` from the
    ``<run_id>_trades.json`` sidecar, then streams matching bars out of
    ``<data_dir>/parquet``. Returns an empty list (not a 404) when the
    sidecar is missing or there are no bars — the chart falls back to
    equity-only when the response is empty.

    Capped at 5000 bars so a year of M1 doesn't blow the JSON response.
    """
    bt_dir = data_dir / "backtests"
    trades_path = bt_dir / f"{run_id}_trades.json"
    if not trades_path.exists():
        return []
    try:
        meta = json.loads(trades_path.read_text())
    except json.JSONDecodeError:
        return []
    symbol = meta.get("symbol")
    tf_str = meta.get("timeframe")
    start_iso = meta.get("start")
    end_iso = meta.get("end")
    if not (symbol and tf_str and start_iso and end_iso):
        return []

    from stinger_fx.data import iter_bars
    from stinger_fx.domain import Timeframe

    try:
        tf = Timeframe(tf_str)
    except ValueError:
        return []
    try:
        start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
    except ValueError:
        return []

    parquet_root = data_dir / "parquet"
    if not parquet_root.exists():
        return []

    out: list[dict] = []
    try:
        for bar in iter_bars(parquet_root, symbol, tf, start, end):
            out.append(
                {
                    "time": bar.time.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.tick_volume,
                }
            )
            if len(out) >= 5000:
                logger.info(
                    "candles_truncated run_id=%s symbol=%s tf=%s capped=5000",
                    run_id, symbol, tf_str,
                )
                break
    except Exception as e:  # noqa: BLE001 — gracefully degrade on missing data
        logger.warning("candles_read_failed run_id=%s err=%s", run_id, e)
        return []
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
