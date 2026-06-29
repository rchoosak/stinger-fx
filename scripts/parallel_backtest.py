#!/usr/bin/env python
"""Parallel multi-strategy / multi-window backtest comparison runner.

Runs a grid of (strategy_id x window) backtests across a process pool so a
comparison sweep that takes ~N x single-run time sequentially finishes in
~ceil(N / workers) x instead. Each strategy's params come straight from
``config/strategies.yaml`` (so e.g. PRS picks up its trend filter + stop
floor automatically); only the symbol-cost / risk knobs are set here to
match the research backtests.

Resource etiquette: workers default to 6 (not all 12 cores) and run at
lowered OS priority (``nice``), so a sweep doesn't peg the machine while
you're using it. Bump ``--workers`` when AFK for max speed.

Examples
--------
    # the standard 4-strategy x 4-window comparison
    python scripts/parallel_backtest.py

    # one strategy across the long window only, all cores, no nice
    python scripts/parallel_backtest.py -s prs_scalper_xau -w 29mo \
        --workers 12 --nice 0
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# (start, end) ISO bounds for each named window. end is exclusive.
WINDOWS: dict[str, tuple[str, str]] = {
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
    "2026": ("2026-01-01", "2026-06-01"),
    "29mo": ("2024-01-01", "2026-06-01"),
}
DEFAULT_STRATEGIES = [
    "d1h4_xauusd",
    "bbr_xauusd",
    "prs_scalper_xau",
    "orb_xauusd_may2026",
]


def _init_worker(nice_delta: int) -> None:
    """Per-worker setup: silence structlog and lower scheduling priority so the
    pool stays friendly to foreground apps."""
    import logging

    logging.disable(logging.INFO)
    if nice_delta:
        try:
            os.nice(nice_delta)
        except (OSError, PermissionError):
            pass


def run_job(job: dict) -> dict:
    """Run one (strategy, window) backtest. Top-level + primitives-only args so
    it pickles cleanly to a spawned worker; the heavy objects (config,
    backtester) are built here, inside the worker process."""
    import asyncio
    import tempfile
    from datetime import UTC, datetime
    from pathlib import Path

    import yaml

    from stinger_fx.backtest import FileBacktester
    from stinger_fx.config.models import (
        BacktestRunConfig,
        PositionSizingConfig,
        RiskConfig,
        StrategiesConfig,
    )
    from stinger_fx.data import in_memory_store
    from stinger_fx.strategies.registry import load_strategy_class, validate_params

    root = Path(job["root"])
    with open(job["config"]) as fh:
        scfg = StrategiesConfig.model_validate(yaml.safe_load(fh))
    entry = next(s for s in scfg.strategies if s.id == job["strategy_id"])
    cls = load_strategy_class(entry.class_path)
    params = validate_params(cls, entry.params)
    feeds = cls.subscriptions(params)
    risk = RiskConfig(
        max_daily_loss_pct=0.0,
        kill_switch_drawdown_pct=0.0,
        position_sizing=PositionSizingConfig(
            enabled=True, risk_per_trade_pct=job["risk_pct"]
        ),
    )
    cfg = BacktestRunConfig(
        id=entry.id, mode="file", strategy_id=entry.id, feeds=feeds,
        granularity="bar", data_source=root,
        start=datetime.fromisoformat(job["start"]).replace(tzinfo=UTC),
        end=datetime.fromisoformat(job["end"]).replace(tzinfo=UTC),
        initial_balance=job["balance"],
        symbol_contract_sizes={job["symbol"]: job["contract"]},
        slippage_pips=0.5, commission_per_lot=3.0,
        swap_long_per_lot=-12.0, swap_short_per_lot=-7.0,
    )

    async def _go() -> dict:
        with tempfile.TemporaryDirectory() as d:
            rep = await FileBacktester(
                strategy=entry, parquet_root=root,
                sqlite_store=in_memory_store(), report_dir=Path(d),
                risk_config=risk,
            ).run(cfg)
        return rep.to_metrics_dict()

    m = asyncio.run(_go())
    return {
        "win": job["win"],
        "label": job["label"],
        "trades": m["trades"],
        "ret": (m["final_balance"] / job["balance"] - 1) * 100,
        "win_rate": m["win_rate"],
        "pf": m["profit_factor"] or 0.0,
        "dd": m["max_drawdown"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-s", "--strategies", default=",".join(DEFAULT_STRATEGIES),
                    help="comma-separated strategy ids (from config)")
    ap.add_argument("-w", "--windows", default="2024,2025,2026,29mo",
                    help=f"comma-separated window names {list(WINDOWS)}")
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel worker processes (default 6; you have 12 cores)")
    ap.add_argument("--nice", type=int, default=10,
                    help="niceness added to each worker (default 10; 0 = normal)")
    ap.add_argument("--risk-pct", type=float, default=1.0)
    ap.add_argument("--balance", type=float, default=100_000.0)
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--contract", type=float, default=100.0)
    ap.add_argument("--config", default="config/strategies.yaml")
    ap.add_argument("--root", default="./data/parquet")
    args = ap.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    windows = [w.strip() for w in args.windows.split(",") if w.strip()]
    for w in windows:
        if w not in WINDOWS:
            ap.error(f"unknown window {w!r}; choose from {list(WINDOWS)}")

    jobs = [
        {
            "strategy_id": sid, "win": win,
            "label": f"{sid}",
            "start": WINDOWS[win][0], "end": WINDOWS[win][1],
            "risk_pct": args.risk_pct, "balance": args.balance,
            "symbol": args.symbol, "contract": args.contract,
            "config": args.config, "root": args.root,
        }
        for win in windows for sid in strategies
    ]

    print(f"# {len(jobs)} backtests across {args.workers} workers "
          f"(nice +{args.nice}) — {len(strategies)} strategies x {len(windows)} windows")
    t0 = time.perf_counter()
    results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(args.nice,)
    ) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"RESULT [{r['win']:>4}] {r['label']:20} "
                      f"trades={r['trades']:>4} return={r['ret']:>+6.1f}% "
                      f"win={r['win_rate']:.2f} PF={r['pf']:>4.2f} "
                      f"maxDD={r['dd']/1000:>5.1f}k", flush=True)
            except Exception as ex_:  # surface per-cell failures, keep going
                print(f"RESULT [{j['win']:>4}] {j['strategy_id']:20} "
                      f"ERROR {type(ex_).__name__}: {str(ex_)[:60]}", flush=True)
    dt = time.perf_counter() - t0
    print(f"# done: {len(results)}/{len(jobs)} ok in {dt:.1f}s "
          f"({dt/max(1, len(jobs)):.1f}s/backtest wall-clock)")


if __name__ == "__main__":
    main()
