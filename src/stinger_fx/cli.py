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
) -> None:
    """Start the trading engine in the configured (or overridden) mode."""
    from stinger_fx.runtime import assemble_and_run

    if mode is not None:
        # Apply override by patching the in-memory config after load. Simpler
        # to set an env var so the loader picks it up; for Phase 1 we just
        # warn if it conflicts.
        typer.echo("--mode override is not yet applied; using config/app.yaml mode")

    try:
        asyncio.run(assemble_and_run(config_dir))
    except KeyboardInterrupt:
        typer.echo("interrupted")


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
        f"OK: app.mode={cfg.app.mode} broker={cfg.app.broker.type} "
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


# --- backtest ---------------------------------------------------------------


@backtest_app.command("run")
def backtest_run(
    run_id: str = typer.Option(..., "--run-id"),
    config_dir: Path = typer.Option(Path("config"), "--config-dir", "-c"),
) -> None:
    """Execute one backtest run (by id) from backtest.yaml."""
    from stinger_fx.backtest import FileBacktester
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

    sqlite = SqliteStore(cfg.app.data_dir / "stinger.db")
    sqlite.create_all()

    if run.mode == "file":
        if run.data_source is None:
            typer.echo("ERROR: file mode requires data_source: <parquet root>", err=True)
            raise typer.Exit(code=1)
        bt = FileBacktester(
            strategy=strategy,
            parquet_root=run.data_source,
            sqlite_store=sqlite,
            report_dir=cfg.app.data_dir / "backtests",
        )
    elif run.mode == "mt5_tester":
        from stinger_fx.backtest.mt5_backtester import MT5StrategyTester

        terminal = Path(cfg.app.broker.mt5.terminal_path or "terminal64.exe") if cfg.app.broker.mt5 else None
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
        broker = build_broker(cfg.app.broker, bus)
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
