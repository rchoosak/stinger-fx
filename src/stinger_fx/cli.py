"""Stinger-Fx CLI — Typer entry point.

Subcommands:
  • run               start the engine (normal/tui/web)
  • backtest          backtest run / list / show
  • config            validate / show YAML config
  • strategy          list strategies known to the registry
  • data              download bars from the broker into Parquet
  • db                initialize the SQLite schema
  • version           print version
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from stinger_fx import __version__

app = typer.Typer(
    name="stinger-fx",
    help="EA Bot Platform — multi-strategy Forex trading on MT5/MT4.",
    no_args_is_help=True,
)
backtest_app = typer.Typer(name="backtest", help="Backtest commands.", no_args_is_help=True)
config_app = typer.Typer(name="config", help="Config commands.", no_args_is_help=True)
strategy_app = typer.Typer(name="strategy", help="Strategy commands.", no_args_is_help=True)
data_app = typer.Typer(name="data", help="Data import / download.", no_args_is_help=True)
db_app = typer.Typer(name="db", help="Database commands.", no_args_is_help=True)
app.add_typer(backtest_app)
app.add_typer(config_app)
app.add_typer(strategy_app)
app.add_typer(data_app)
app.add_typer(db_app)


@app.callback()
def _root() -> None:
    """Stinger-Fx — EA Bot Platform for Forex trading on MT5/MT4."""


# --- version ----------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the installed Stinger-Fx version."""
    typer.echo(f"stinger-fx {__version__}")


# --- run --------------------------------------------------------------------


@app.command("run")
def run_cmd(
    config_dir: Path = typer.Option(Path("config"), "--config-dir", "-c", help="YAML config dir"),
    mode: str | None = typer.Option(
        None, "--mode", "-m", help="Override app.yaml: normal | tui | web"
    ),
    detach: bool = typer.Option(
        False, "--detach", help="Spawn the engine in the background and return immediately.",
    ),
) -> None:
    """Start the trading engine in the configured (or overridden) mode."""
    if mode is not None and mode not in {"normal", "tui", "web"}:
        typer.echo(f"ERROR: --mode must be one of normal|tui|web, got {mode!r}", err=True)
        raise typer.Exit(code=1)

    if detach:
        from stinger_fx.config import load_all
        from stinger_fx.detach import DetachError, detach_spawn

        cfg = load_all(config_dir)
        try:
            info = detach_spawn(config_dir, cfg.app.data_dir)
        except DetachError as e:
            typer.echo(f"ERROR: {e}", err=True)
            raise typer.Exit(code=1) from e
        typer.echo(f"engine detached pid={info['pid']}")
        typer.echo(f"web UI: {info['web_url']}")
        typer.echo("  · status: stinger-fx status")
        typer.echo("  · stop:   stinger-fx stop")
        typer.echo(f"  · logs:   tail -f {info['log']}")
        return

    from stinger_fx.runtime import assemble_and_run

    try:
        asyncio.run(assemble_and_run(config_dir, mode_override=mode))
    except KeyboardInterrupt:
        typer.echo("interrupted")


@app.command("status")
def status_cmd(
    config_dir: Path = typer.Option(Path("config"), "--config-dir", "-c"),
) -> None:
    """Check the status of the detached engine, if any."""
    from stinger_fx.config import load_all
    from stinger_fx.detach import detach_status

    cfg = load_all(config_dir)
    info = detach_status(cfg.app.data_dir)
    typer.echo(json.dumps(info, indent=2))
    if info["status"] in ("not_running", "stale_pidfile"):
        raise typer.Exit(code=1)


@app.command("stop")
def stop_cmd(
    config_dir: Path = typer.Option(Path("config"), "--config-dir", "-c"),
    timeout: float = typer.Option(15.0, "--timeout", help="Seconds to wait for clean shutdown."),
) -> None:
    """Politely shut down the detached engine."""
    from stinger_fx.config import load_all
    from stinger_fx.detach import DetachError, detach_stop

    cfg = load_all(config_dir)
    try:
        info = detach_stop(cfg.app.data_dir, timeout_seconds=timeout)
    except DetachError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=1) from e
    typer.echo(json.dumps(info, indent=2))


# --- config -----------------------------------------------------------------


@config_app.command("validate")
def config_validate(
    config_dir: Path = typer.Option(Path("config"), "--config-dir", "-c"),
) -> None:
    """Validate every YAML file under the config dir."""
    from stinger_fx.config import load_all
    from stinger_fx.core.errors import ConfigError

    try:
        cfg = load_all(config_dir)
    except ConfigError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=1) from e
    typer.echo(
        f"OK: app.mode={cfg.app.mode} broker={cfg.app.primary_broker_config.type} "
        f"strategies={len(cfg.strategies.strategies)} "
        f"backtest_runs={len(cfg.backtest.runs)}"
    )


@config_app.command("show")
def config_show(
    config_dir: Path = typer.Option(Path("config"), "--config-dir", "-c"),
) -> None:
    """Print the loaded config as JSON."""
    from stinger_fx.config import load_all

    cfg = load_all(config_dir)
    typer.echo(cfg.model_dump_json(indent=2))


# --- strategy ---------------------------------------------------------------


@strategy_app.command("list")
def strategy_list(
    config_dir: Path = typer.Option(Path("config"), "--config-dir", "-c"),
) -> None:
    """List every strategy entry from strategies.yaml."""
    from stinger_fx.config import load_all

    cfg = load_all(config_dir)
    console = Console()
    table = Table("id", "class_path", "enabled", "params")
    for s in cfg.strategies.strategies:
        table.add_row(s.id, s.class_path, str(s.enabled), json.dumps(s.params))
    console.print(table)


@strategy_app.command("scaffold")
def strategy_scaffold(
    name: str = typer.Argument(..., help="snake_case strategy id (also the file name)"),
    target_dir: Path = typer.Option(
        Path("src/stinger_fx/strategies/user"),
        "--dir",
        "-d",
        help="Directory the new strategy file is written to.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite if the file exists."),
) -> None:
    """Generate a new strategy stub under --dir.

    Prints a YAML snippet to paste into config/strategies.yaml so the strategy
    can be activated immediately.
    """
    from stinger_fx.strategies.scaffold import (
        ScaffoldError,
        derive_module_path,
        scaffold,
        yaml_snippet,
    )

    try:
        out = scaffold(name, target_dir, force=force)
    except ScaffoldError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=1) from e
    module_path = derive_module_path(out, repo_root=Path.cwd())
    typer.echo(f"wrote {out}")
    typer.echo("")
    typer.echo("Add this to config/strategies.yaml:")
    typer.echo("")
    typer.echo(yaml_snippet(name, module_path))


# --- backtest ---------------------------------------------------------------


@backtest_app.command("run")
def backtest_run(
    run_id: str = typer.Option(..., "--run-id"),
    config_dir: Path = typer.Option(Path("config"), "--config-dir", "-c"),
    speed: float | None = typer.Option(
        None,
        "--speed",
        help=(
            "Playback throttle. 0 = max speed (default). 1 = real-time. "
            "60 = 1 sim-minute per wall-second. 0.5 = half real-time. "
            "Overrides the run's `speed` field in backtest.yaml when provided."
        ),
        min=0,
    ),
) -> None:
    """Execute one backtest run (by id) from backtest.yaml."""
    from stinger_fx.backtest import FileBacktester
    from stinger_fx.backtest.base import BaseBacktester
    from stinger_fx.config import load_all
    from stinger_fx.core.errors import BacktestError
    from stinger_fx.data import SqliteStore

    cfg = load_all(config_dir)
    run = next((r for r in cfg.backtest.runs if r.id == run_id), None)
    if run is None:
        typer.echo(f"ERROR: no backtest run with id {run_id!r}", err=True)
        raise typer.Exit(code=1)
    strategy = next(
        (s for s in cfg.strategies.strategies if s.id == run.strategy_id),
        None,
    )
    if strategy is None:
        typer.echo(f"ERROR: backtest references unknown strategy {run.strategy_id!r}", err=True)
        raise typer.Exit(code=1)

    # CLI --speed wins over the YAML value when supplied.
    if speed is not None:
        run = run.model_copy(update={"speed": speed})

    sqlite = SqliteStore(cfg.app.data_dir / "stinger.db")
    sqlite.create_all()

    bt: BaseBacktester
    if run.mode == "file":
        if run.data_source is None:
            typer.echo("ERROR: file mode requires data_source: <parquet root>", err=True)
            raise typer.Exit(code=1)
        bt = FileBacktester(
            strategy=strategy,
            parquet_root=run.data_source,
            sqlite_store=sqlite,
            report_dir=cfg.app.data_dir / "backtests",
            # Enforce the same pre-trade risk gates a live engine would —
            # backtest results now reflect daily-loss / kill-switch /
            # max-position rejections instead of raw strategy edge.
            risk_config=cfg.app.risk,
        )
    elif run.mode == "mt5_tester":
        from stinger_fx.backtest.mt5_backtester import MT5StrategyTester

        primary_broker = cfg.app.primary_broker_config
        terminal = (
            Path(primary_broker.mt5.terminal_path or "terminal64.exe")
            if primary_broker.mt5
            else None
        )
        if terminal is None:
            typer.echo("ERROR: mt5_tester mode requires broker.mt5.terminal_path", err=True)
            raise typer.Exit(code=1)
        bt = MT5StrategyTester(
            strategy=strategy,
            terminal_path=terminal,
            tester_workdir=cfg.app.data_dir / "mt5_tester",
            report_dir=cfg.app.data_dir / "backtests",
        )
    else:
        typer.echo(f"ERROR: backtest mode {run.mode!r} not implemented", err=True)
        raise typer.Exit(code=1)

    try:
        report = asyncio.run(bt.run(run))
    except BacktestError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=1) from e

    typer.echo("=== Backtest Report ===")
    typer.echo(json.dumps(report.to_metrics_dict(), indent=2))


@backtest_app.command("list")
def backtest_list(
    config_dir: Path = typer.Option(Path("config"), "--config-dir", "-c"),
) -> None:
    """List all backtest runs from SQLite."""
    from stinger_fx.config import load_all
    from stinger_fx.data import BacktestRepo, SqliteStore

    cfg = load_all(config_dir)
    sqlite = SqliteStore(cfg.app.data_dir / "stinger.db")
    sqlite.create_all()
    rows = BacktestRepo(sqlite).list_runs(limit=100)
    console = Console()
    table = Table("run_id", "strategy", "started_at", "finished_at", "metrics")
    for r in rows:
        table.add_row(
            r.run_id,
            r.strategy_id,
            str(r.started_at),
            str(r.finished_at),
            r.metrics_json or "",
        )
    console.print(table)


@backtest_app.command("show")
def backtest_show(
    run_id: str = typer.Argument(...),
    config_dir: Path = typer.Option(Path("config"), "--config-dir", "-c"),
) -> None:
    """Show one backtest run's metrics + report path."""
    from sqlmodel import select

    from stinger_fx.config import load_all
    from stinger_fx.data import SqliteStore
    from stinger_fx.data.schemas import BacktestRunRow

    cfg = load_all(config_dir)
    sqlite = SqliteStore(cfg.app.data_dir / "stinger.db")
    sqlite.create_all()
    with sqlite.session() as s:
        row = s.exec(select(BacktestRunRow).where(BacktestRunRow.run_id == run_id)).first()
    if row is None:
        typer.echo(f"no backtest run with id {run_id!r}", err=True)
        raise typer.Exit(code=1)
    typer.echo(json.dumps({
        "run_id": row.run_id,
        "strategy_id": row.strategy_id,
        "started_at": str(row.started_at),
        "finished_at": str(row.finished_at),
        "metrics": json.loads(row.metrics_json) if row.metrics_json else None,
        "report_path": row.report_path,
    }, indent=2))


@backtest_app.command("sweep")
def backtest_sweep(
    sweep_id: str = typer.Option(..., "--sweep-id"),
    config_dir: Path = typer.Option(Path("config"), "--config-dir", "-c"),
) -> None:
    """Run a parameter sweep defined in backtest.yaml (sweeps: section)."""
    from stinger_fx.backtest import ParameterSweep
    from stinger_fx.config import load_all
    from stinger_fx.data import SqliteStore

    cfg = load_all(config_dir)
    sweep_cfg = next((s for s in cfg.backtest.sweeps if s.id == sweep_id), None)
    if sweep_cfg is None:
        typer.echo(f"ERROR: no sweep with id {sweep_id!r}", err=True)
        raise typer.Exit(code=1)
    strategy = next(
        (s for s in cfg.strategies.strategies if s.id == sweep_cfg.strategy_id), None
    )
    if strategy is None:
        typer.echo(
            f"ERROR: sweep references unknown strategy {sweep_cfg.strategy_id!r}",
            err=True,
        )
        raise typer.Exit(code=1)

    sqlite = SqliteStore(cfg.app.data_dir / "stinger.db")
    sqlite.create_all()

    sweep = ParameterSweep(
        base_strategy=strategy,
        sqlite_store=sqlite,
        report_dir=cfg.app.data_dir / "sweeps",
    )
    report = asyncio.run(sweep.run(sweep_cfg))

    console = Console()
    console.print(f"\n=== Sweep [bold]{sweep_id}[/] — top {min(sweep_cfg.top_n, len(report.ranked))} by [bold]{sweep_cfg.rank_by}[/] ===\n")
    top = report.top(sweep_cfg.top_n)
    if not top:
        console.print("[red]no results[/]")
        return
    # Build table headers from union of param keys + key metrics
    param_keys = sorted({k for r in top for k in r.params})
    metric_keys = ["net_pnl", "sharpe", "profit_factor", "win_rate", "max_drawdown", "trades"]
    table = Table("rank", *param_keys, *metric_keys)
    for rank, r in enumerate(top, start=1):
        row = [str(rank)]
        row.extend(str(r.params.get(k, "—")) for k in param_keys)
        for m in metric_keys:
            v = r.metrics.get(m)
            row.append("—" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)))
        table.add_row(*row)
    console.print(table)


@backtest_app.command("sweep-list")
def backtest_sweep_list(
    config_dir: Path = typer.Option(Path("config"), "--config-dir", "-c"),
) -> None:
    """List all sweep runs from SQLite."""
    from stinger_fx.config import load_all
    from stinger_fx.data import SqliteStore, SweepRepo

    cfg = load_all(config_dir)
    sqlite = SqliteStore(cfg.app.data_dir / "stinger.db")
    sqlite.create_all()
    rows = SweepRepo(sqlite).list_sweeps(limit=100)
    console = Console()
    table = Table("sweep_id", "strategy", "rank_by", "combos", "best_value", "finished_at")
    for r in rows:
        table.add_row(
            r.sweep_id,
            r.strategy_id,
            r.rank_by,
            str(r.total_combos),
            f"{r.best_metric_value:.4f}" if r.best_metric_value is not None else "—",
            str(r.finished_at),
        )
    console.print(table)


# --- data -------------------------------------------------------------------


@data_app.command("download")
def data_download(
    symbol: str = typer.Option(..., "--symbol"),
    timeframe: str = typer.Option(..., "--timeframe"),
    start: datetime = typer.Option(..., "--start", formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]),
    end: datetime | None = typer.Option(None, "--end"),
    config_dir: Path = typer.Option(Path("config"), "--config-dir", "-c"),
) -> None:
    """Download historical bars from the configured broker into Parquet."""
    from stinger_fx.brokers import build_broker
    from stinger_fx.config import load_all
    from stinger_fx.core import AsyncEventBus
    from stinger_fx.data import ParquetStore
    from stinger_fx.domain.timeframes import Timeframe

    cfg = load_all(config_dir)
    tf = Timeframe(timeframe)
    end = end or datetime.now(UTC)
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)

    async def _do() -> None:
        bus = AsyncEventBus()
        broker = build_broker(cfg.app.primary_broker_config, bus)
        await broker.connect()
        try:
            tbl = await broker.get_history_bars(symbol, tf, start, end)
            typer.echo(f"fetched {tbl.num_rows} bars")
            if tbl.num_rows:
                store = ParquetStore(cfg.app.data_dir / "parquet")
                # Convert arrow rows to Bar domain objects via append_bars
                from stinger_fx.domain import Bar

                bars = [
                    Bar(
                        symbol=symbol,
                        timeframe=tf,
                        time=row["time"],
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        tick_volume=row["tick_volume"],
                        real_volume=row["real_volume"],
                        spread=row["spread"],
                        is_closed=True,
                    )
                    for row in tbl.to_pylist()
                ]
                n = store.append_bars(symbol, tf, bars)
                typer.echo(f"wrote {n} bars to {cfg.app.data_dir / 'parquet' / symbol / tf.value}")
        finally:
            await broker.disconnect()
            await bus.close()

    asyncio.run(_do())


@data_app.command("import-ticks")
def data_import_ticks(
    csv: Path = typer.Option(..., "--csv", help="Path to the tick CSV file."),
    symbol: str = typer.Option(..., "--symbol", help="Symbol to file the ticks under."),
    time_col: str = typer.Option("time", "--time-col", help="CSV column holding the timestamp."),
    bid_col: str = typer.Option("bid", "--bid-col", help="CSV column holding the bid price."),
    ask_col: str = typer.Option("ask", "--ask-col", help="CSV column holding the ask price."),
    last_col: str | None = typer.Option(None, "--last-col", help="Optional 'last' price column."),
    volume_col: str | None = typer.Option(None, "--volume-col", help="Optional volume column."),
    flags_col: str | None = typer.Option(None, "--flags-col", help="Optional flags column."),
    tz: str = typer.Option("UTC", "--tz", help="Timezone for naive timestamps (IANA name)."),
    config_dir: Path = typer.Option(Path("config"), "--config-dir", "-c"),
) -> None:
    """Import a tick CSV into the Parquet store under data/parquet/<symbol>/TICK.

    Accepts ISO 8601 timestamps natively; for "2024.01.15 00:00:00" style
    timestamps pandas is used as a fallback parser. Naive timestamps are
    interpreted in --tz (default UTC) and stored as UTC.
    """
    from stinger_fx.config import load_all
    from stinger_fx.data.csv_import import CsvImportError, import_ticks_csv

    cfg = load_all(config_dir)
    parquet_root = cfg.app.data_dir / "parquet"
    try:
        n = import_ticks_csv(
            csv,
            symbol=symbol,
            parquet_root=parquet_root,
            time_col=time_col,
            bid_col=bid_col,
            ask_col=ask_col,
            last_col=last_col,
            volume_col=volume_col,
            flags_col=flags_col,
            tz=tz,
        )
    except CsvImportError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=1) from e
    typer.echo(f"imported {n} ticks → {parquet_root / symbol / 'TICK'}")


# --- db ---------------------------------------------------------------------


@db_app.command("migrate")
def db_migrate(
    config_dir: Path = typer.Option(Path("config"), "--config-dir", "-c"),
) -> None:
    """Create the SQLite tables if they don't exist."""
    from stinger_fx.config import load_all
    from stinger_fx.data import SqliteStore

    cfg = load_all(config_dir)
    sqlite = SqliteStore(cfg.app.data_dir / "stinger.db")
    sqlite.create_all()
    typer.echo(f"OK: schema applied at {sqlite.db_path}")


if __name__ == "__main__":
    app()
