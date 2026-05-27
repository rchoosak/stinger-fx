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
    sqlite_store=None,
    user_strategies_dir: Path | None = None,
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
    # sqlite_store, if provided, enables the /audit page to read recent
    # decisions / order modifications / reconciliation mismatches.
    app.state.sqlite_store = sqlite_store
    # user_strategies_dir, if provided, enables the /editor pages that let
    # the operator create / edit / save Python strategy files in the
    # browser. Defaults to None → editor returns 503 with a clear message.
    app.state.user_strategies_dir = (
        Path(user_strategies_dir) if user_strategies_dir is not None else None
    )
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

    @app.get("/backtest/{run_id}/monte_carlo.json")
    async def backtest_monte_carlo(run_id: str, n: int = 500, seed: int | None = None):
        """Run a Monte Carlo bootstrap over this run's trade list (Phase 6.3.D).

        Returns percentile bands for net_pnl, max_drawdown, sharpe plus
        an equity-curve envelope (5th/50th/95th). The result isn't cached
        — each call re-runs the simulation. ``n`` capped at 5000 to keep
        latency bounded; ``seed`` lets the operator reproduce a band.
        """
        from fastapi.responses import JSONResponse

        from stinger_fx.backtest.monte_carlo import run_monte_carlo

        data = _load_replay_data(app.state.data_dir, run_id)
        if data is None:
            raise HTTPException(404, f"no backtest run with id {run_id!r}")
        trades = data["meta"].get("trades", [])
        if not trades:
            return JSONResponse({"error": "no trades to bootstrap", "n_trades": 0})
        pnls = [float(t.get("pnl", 0.0)) for t in trades]
        n_sims = min(max(n, 10), 5000)
        result = run_monte_carlo(
            pnls,
            n_simulations=n_sims,
            initial_balance=float(data["meta"].get("initial_balance", 10_000.0)),
            random_seed=seed,
        )
        return JSONResponse(result.to_json())

    # --- Portfolio aggregation (Phase 7.A) -------------------------------

    @app.get("/portfolio", response_class=HTMLResponse)
    async def portfolio_form(request: Request):
        """List backtest runs and let the operator pick which to combine."""
        runs = _list_backtest_runs(app.state.data_dir)
        return TEMPLATES.TemplateResponse(
            request=request,
            name="portfolio_form.html",
            context={"runs": runs},
        )

    @app.get("/portfolio/view", response_class=HTMLResponse)
    async def portfolio_view(request: Request, runs: str = ""):
        """Aggregate the supplied comma-separated run_ids into a portfolio
        report and render the view. The form posts here as a GET with
        ``runs=a,b,c`` so URLs are bookmarkable."""
        run_ids = [r.strip() for r in runs.split(",") if r.strip()]
        if not run_ids:
            raise HTTPException(400, "no run_ids supplied")
        summary = _build_portfolio_summary(app.state.data_dir, run_ids)
        if summary is None:
            raise HTTPException(404, "one or more run_ids not found")
        return TEMPLATES.TemplateResponse(
            request=request,
            name="portfolio_view.html",
            context={"summary": summary, "run_ids": run_ids},
        )

    @app.get("/portfolio/data.json")
    async def portfolio_data(runs: str = ""):
        """JSON payload for the portfolio chart — equity curve + correlations."""
        from fastapi.responses import JSONResponse

        run_ids = [r.strip() for r in runs.split(",") if r.strip()]
        if not run_ids:
            raise HTTPException(400, "no run_ids supplied")
        summary = _build_portfolio_summary(app.state.data_dir, run_ids)
        if summary is None:
            raise HTTPException(404, "one or more run_ids not found")
        return JSONResponse(summary)

    # --- Sweep + walk-forward viewers (Phase 6.3.D) ----------------------

    @app.get("/sweep", response_class=HTMLResponse)
    async def sweep_list(request: Request):
        runs = _list_sweep_runs(app.state.data_dir)
        return TEMPLATES.TemplateResponse(
            request=request, name="sweep_list.html", context={"runs": runs}
        )

    @app.get("/sweep/{sweep_id}", response_class=HTMLResponse)
    async def sweep_view(sweep_id: str, request: Request):
        data = _load_sweep_summary(app.state.data_dir, sweep_id)
        if data is None:
            raise HTTPException(404, f"no sweep run with id {sweep_id!r}")
        # Detect 2-param case — enables a heatmap; otherwise show ranked table only
        best = data.get("best_params") or {}
        param_keys = sorted(best.keys()) if best else []
        is_2d = len(param_keys) == 2
        return TEMPLATES.TemplateResponse(
            request=request,
            name="sweep_view.html",
            context={
                "sweep_id": sweep_id,
                "summary": data,
                "is_2d": is_2d,
                "param_keys": param_keys,
            },
        )

    @app.get("/walkforward", response_class=HTMLResponse)
    async def wf_list(request: Request):
        runs = _list_walk_forward_runs(app.state.data_dir)
        return TEMPLATES.TemplateResponse(
            request=request, name="walk_forward_list.html", context={"runs": runs}
        )

    @app.get("/walkforward/{wf_id}", response_class=HTMLResponse)
    async def wf_view(wf_id: str, request: Request):
        data = _load_walk_forward_summary(app.state.data_dir, wf_id)
        if data is None:
            raise HTTPException(404, f"no walk-forward run with id {wf_id!r}")
        return TEMPLATES.TemplateResponse(
            request=request,
            name="walk_forward_view.html",
            context={"wf_id": wf_id, "summary": data},
        )

    # --- Audit / reconciliation (Phase 6.1.D) ----------------------------

    @app.get("/audit", response_class=HTMLResponse)
    async def audit(request: Request):
        """Render decisions + modifications + reconciliation mismatches.

        Read-only operator view. Pulls recent rows from each table; HTMX
        polls the page periodically. When no SqliteStore is configured the
        tables show "—".
        """
        store = getattr(app.state, "sqlite_store", None)
        decisions: list[dict] = []
        modifications: list[dict] = []
        mismatches: list[dict] = []
        if store is not None:
            from stinger_fx.data.repositories import (
                OrderModificationRepo,
                ReconciliationRepo,
            )
            from stinger_fx.data.schemas import DecisionRow
            from sqlmodel import desc, select as _select

            with store.session() as s:
                d_rows = list(
                    s.exec(
                        _select(DecisionRow).order_by(desc(DecisionRow.ts)).limit(50)
                    )
                )
                decisions = [
                    {
                        "ts": r.ts.isoformat() if hasattr(r.ts, "isoformat") else str(r.ts),
                        "action": r.action,
                        "reason": r.reason,
                    }
                    for r in d_rows
                ]
            for row in OrderModificationRepo(store).recent(50):
                modifications.append(
                    {
                        "ts": row.ts.isoformat() if hasattr(row.ts, "isoformat") else str(row.ts),
                        "ticket": row.ticket,
                        "strategy_id": row.strategy_id,
                        "type": row.modification_type,
                        "reason": row.reason,
                    }
                )
            for row in ReconciliationRepo(store).recent(50):
                mismatches.append(
                    {
                        "ts": row.ts.isoformat() if hasattr(row.ts, "isoformat") else str(row.ts),
                        "ticket": row.ticket,
                        "strategy_id": row.strategy_id,
                        "type": row.mismatch_type,
                        "expected": row.expected_value,
                        "actual": row.actual_value,
                        "details": row.details,
                    }
                )
        return TEMPLATES.TemplateResponse(
            request=request,
            name="audit.html",
            context={
                "decisions": decisions,
                "modifications": modifications,
                "mismatches": mismatches,
                "has_store": store is not None,
            },
        )

    # --- Strategy code editor (Phase 6.4.D) ------------------------------

    def _require_editor_dir() -> Path:
        d = app.state.user_strategies_dir
        if d is None:
            raise HTTPException(
                503,
                "strategy editor disabled: pass user_strategies_dir to create_app",
            )
        return d

    @app.get("/editor", response_class=HTMLResponse)
    async def editor_index(request: Request):
        from stinger_fx.ui.web.strategy_editor import list_strategies

        d = _require_editor_dir()
        # Create on first access so the directory exists for new scaffolds
        d.mkdir(parents=True, exist_ok=True)
        return TEMPLATES.TemplateResponse(
            request=request,
            name="editor_list.html",
            context={
                "names": list_strategies(d),
                "user_dir": str(d),
            },
        )

    @app.get("/editor/{name}", response_class=HTMLResponse)
    async def editor_view(name: str, request: Request):
        from stinger_fx.ui.web.strategy_editor import is_safe_name, resolve_path

        d = _require_editor_dir()
        if not is_safe_name(name):
            raise HTTPException(400, f"invalid strategy name: {name!r}")
        try:
            path = resolve_path(d, name)
        except ValueError as e:
            raise HTTPException(403, str(e)) from e
        if not path.exists():
            raise HTTPException(404, f"strategy file not found: {name}")
        return TEMPLATES.TemplateResponse(
            request=request,
            name="editor_view.html",
            context={"name": name, "filename": path.name},
        )

    @app.get("/editor/{name}/source")
    async def editor_get_source(name: str):
        from fastapi.responses import PlainTextResponse

        from stinger_fx.ui.web.strategy_editor import is_safe_name, resolve_path

        d = _require_editor_dir()
        if not is_safe_name(name):
            raise HTTPException(400, f"invalid strategy name: {name!r}")
        try:
            path = resolve_path(d, name)
        except ValueError as e:
            raise HTTPException(403, str(e)) from e
        if not path.exists():
            raise HTTPException(404, f"strategy file not found: {name}")
        return PlainTextResponse(path.read_text())

    @app.put("/editor/{name}/source")
    async def editor_put_source(name: str, request: Request):
        from fastapi.responses import JSONResponse

        from stinger_fx.ui.web.strategy_editor import (
            is_safe_name,
            resolve_path,
            validate_source,
        )

        d = _require_editor_dir()
        if not is_safe_name(name):
            raise HTTPException(400, f"invalid strategy name: {name!r}")
        try:
            path = resolve_path(d, name)
        except ValueError as e:
            raise HTTPException(403, str(e)) from e
        body = (await request.body()).decode("utf-8")
        err = validate_source(body)
        if err is not None:
            return JSONResponse(
                {
                    "ok": False,
                    "error": err.message,
                    "line": err.line,
                    "column": err.column,
                },
                status_code=400,
            )
        d.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        return JSONResponse({"ok": True, "name": name, "bytes": len(body)})

    @app.post("/editor/new")
    async def editor_new(request: Request):
        from fastapi.responses import JSONResponse

        from stinger_fx.ui.web.strategy_editor import (
            is_safe_name,
            resolve_path,
            scaffold_source,
        )

        d = _require_editor_dir()
        form = await request.form()
        name = str(form.get("name", "")).strip().lower()
        if not is_safe_name(name):
            raise HTTPException(
                400,
                "name must match [a-z][a-z0-9_]* (lowercase letters / digits / underscores)",
            )
        try:
            path = resolve_path(d, name)
        except ValueError as e:
            raise HTTPException(403, str(e)) from e
        if path.exists():
            raise HTTPException(409, f"strategy {name!r} already exists")
        d.mkdir(parents=True, exist_ok=True)
        path.write_text(scaffold_source(name))
        return JSONResponse({"ok": True, "name": name, "path": str(path)})

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


# --- Portfolio aggregation helpers (Phase 7.A) ------------------------------


def _build_portfolio_summary(data_dir: Path, run_ids: list[str]) -> dict | None:
    """Load each run's trades + equity, aggregate via aggregate_portfolio,
    return the summary dict. Returns None if any run is missing."""
    from datetime import datetime as _dt

    from stinger_fx.backtest.portfolio import aggregate_portfolio
    from stinger_fx.backtest.reports import BacktestReport, TradeRecord

    reports: list[BacktestReport] = []
    for run_id in run_ids:
        data = _load_replay_data(data_dir, run_id)
        if data is None:
            return None
        meta = data["meta"]
        equity = data.get("equity", [])
        # Rebuild minimal BacktestReport — only fields aggregator needs
        equity_curve = []
        for row in equity:
            try:
                ts = _dt.fromisoformat(row["time"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            equity_curve.append((ts, float(row["equity"])))
        trades = []
        for t in meta.get("trades", []):
            try:
                trades.append(TradeRecord(
                    open_ts=_dt.fromisoformat(t["open_ts"].replace("Z", "+00:00")),
                    close_ts=_dt.fromisoformat(t["close_ts"].replace("Z", "+00:00")),
                    side=t.get("side", "buy"),
                    open_price=float(t.get("open_price", 0.0)),
                    close_price=float(t.get("close_price", 0.0)),
                    volume=float(t.get("volume", 0.0)),
                    pnl=float(t.get("pnl", 0.0)),
                ))
            except (ValueError, KeyError, AttributeError):
                continue
        report = BacktestReport(
            run_id=run_id,
            strategy_id=meta.get("strategy_id", "—"),
            started_at=_dt.now(),
            finished_at=_dt.now(),
            trades=trades,
            equity_curve=equity_curve,
            initial_balance=float(meta.get("initial_balance", 10_000.0)),
            final_balance=float(meta.get("final_balance", 10_000.0)),
        )
        reports.append(report)
    portfolio = aggregate_portfolio(reports, portfolio_id="+".join(run_ids))
    return portfolio.to_summary()


# --- Sweep + walk-forward summary helpers (Phase 6.3.D) ---------------------


def _list_sweep_runs(data_dir: Path) -> list[dict]:
    """Enumerate ``<data_dir>/sweeps/*_summary.json`` files."""
    sweep_dir = data_dir / "sweeps"
    if not sweep_dir.is_dir():
        return []
    out: list[dict] = []
    for path in sorted(sweep_dir.glob("*_summary.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append(
            {
                "sweep_id": data.get("sweep_id", path.stem.removesuffix("_summary")),
                "strategy_id": data.get("strategy_id", "—"),
                "total_combos": data.get("total_combos", 0),
                "rank_by": data.get("rank_by", "—"),
                "best_metric_value": data.get("best_metric_value"),
                "finished_at": data.get("finished_at", ""),
            }
        )
    out.sort(key=lambda r: r["finished_at"], reverse=True)
    return out


def _load_sweep_summary(data_dir: Path, sweep_id: str) -> dict | None:
    path = data_dir / "sweeps" / f"{sweep_id}_summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _list_walk_forward_runs(data_dir: Path) -> list[dict]:
    wf_dir = data_dir / "walk_forward"
    if not wf_dir.is_dir():
        return []
    out: list[dict] = []
    for path in sorted(wf_dir.glob("*_summary.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append(
            {
                "wf_id": data.get("id", path.stem.removesuffix("_summary")),
                "strategy_id": data.get("strategy_id", "—"),
                "scheme": data.get("scheme", "—"),
                "n_folds": data.get("n_folds", 0),
                "rank_by": data.get("rank_by", "—"),
                "consistency_score": data.get("consistency_score"),
                "avg_oos_metric": data.get("avg_oos_metric"),
                "finished_at": data.get("finished_at", ""),
            }
        )
    out.sort(key=lambda r: r["finished_at"], reverse=True)
    return out


def _load_walk_forward_summary(data_dir: Path, wf_id: str) -> dict | None:
    path = data_dir / "walk_forward" / f"{wf_id}_summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
