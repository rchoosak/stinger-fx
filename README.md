# Stinger-Fx

EA (Expert Advisor) Bot Platform for Forex trading.

- Trade via **MT5** (Phase 1) and **MT4** (Phase 2), configurable per app
- Pull market data from broker across timeframes: tick, 1m, 2m, 3m, 5m, 10m, 15m, 30m, 45m, 1H, 2H, 4H, 1D, 1W, 1MN
- **Class-based Python strategies** with isolated runtime — many strategies active concurrently
- **Backtest** via MT5 Strategy Tester or file-based replay (same strategy code as live)
- Three runtime UIs: **Normal** (CLI), **TUI** (Textual), **Web** (FastAPI + HTMX)
- **Structured JSONL logs** + SQLite mirror for trade analytics
- **Hot-reloadable YAML config** for app, strategies, and backtests

> **Status:** Phase 1 in progress. Windows-only at runtime (`MetaTrader5` Python package is Windows-native). Unit tests run on any OS — broker integration tests are marked `@pytest.mark.mt5` and skipped off-Windows.

## Quick start

```bash
uv sync --extra dev          # install everything except MetaTrader5
# On Windows:
uv sync --extra dev --extra mt5

uv run stinger-fx version
uv run stinger-fx config validate
uv run stinger-fx run --mode normal
```

## Project layout

See [docs/architecture.md](docs/architecture.md) (TODO) — and the implementation plan checked into the repo issue tracker.

## License

MIT — see [LICENSE](LICENSE).
