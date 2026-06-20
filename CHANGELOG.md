# Changelog

All notable changes to Stinger-Fx are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-06-20

First stable release. A multi-strategy Forex/metals trading platform for
MetaTrader 5, with a full backtesting suite, a live web UI, and hot-reloadable
configuration. The live engine and the backtester share the same strategy,
runner, order-router, and risk code paths — what you tune in a backtest is what
runs live.

### Trading engine
- **MT5 broker** over the official MetaTrader5 SDK, with all SDK calls funnelled
  through a single-worker executor to keep the non-thread-safe SDK serialized.
- **Auto-reconnect** with exponential backoff, plus **tick gap-fill** on
  reconnect so live BarAggregators don't miss bars during an outage, and a
  **tick-stream watchdog** that surfaces a dead feed as a climbing gauge.
- **Multi-account** support (one engine, several MT5 accounts) with per-strategy
  account routing.
- **Hot-reloadable config** — param changes swap live, strategies start/stop on
  add/remove/enable without restarting the engine.
- **Order management** — market + pending (STOP/LIMIT/STOP_LIMIT), partial
  close, modify SL/TP, OCO brackets, and an order queue with retry on transient
  broker errors.
- **Detach mode** — run the engine in the background with `status` / `stop`.

### Strategies & indicators
- Example strategies: MA crossover, regime-filtered MA, Opening-Range Breakout,
  VWAP Pullback Continuation, Liquidity Sweep Reversal, Pullback Reversal
  Scalper, Momentum Breakout Scalper, Multi-Timeframe Confluence Scalper, and
  Order-Flow Imbalance Scalper.
- 21 technical indicators (RSI, ATR, Stoch RSI, ADX, VWAP, Donchian, Ichimoku,
  Keltner, MACD, PSAR, and more), all pure functions over bar/close sequences.
- **Multi-timeframe** strategies — declare M1/M5/M15 feeds, read each via
  `ctx.history_for(...)`.
- **Position managers** — trailing stop, attachable per-entry.
- `stinger-fx strategy scaffold` to generate a new strategy stub.

### Backtesting
- **FileBacktester** replays historical Parquet data through the same engine
  path as live — bar mode and tick-precise mode.
- **Playback throttle** (`--speed`) — replay at max speed, real-time, or any
  multiplier for debugging and demos.
- **Live backtest web UI** — run a backtest inside the web process and watch
  candles, equity, and an Orders table update over SSE, with Start/Stop + speed
  controls and a 50/50 chart-vs-orders layout.
- **Risk parity** — backtests enforce the same daily-loss / kill-switch /
  max-position gates the live engine does, driven by the simulated clock.
- **Parameter sweeps**, **Pareto optimization**, **walk-forward**, and
  **Monte-Carlo** bootstrap, plus a **metric DSL** for custom ranking.

### Risk management
- Per-strategy max-open-positions, account-wide daily-loss limit, and an
  equity-drawdown **kill switch**, all configurable in `app.yaml`.

### Data
- **Dukascopy tick downloader** script + a **Parquet store** with daily
  partitioning, and **CSV tick import**.

### UI & observability
- **Web UI** (FastAPI) with dashboard, backtest replay, portfolio, and sweep
  views; **Textual TUI**; **Prometheus metrics**; latency telemetry; trade
  journal; reconciliation auditing; and Telegram/Discord **notification sinks**.

### Performance
- Stoch RSI computed via an incremental `rsi_series` — ~14× faster than the
  per-offset recompute, with bit-for-bit identical output.

[1.0.0]: https://github.com/rchoosak/stinger-fx/releases/tag/v1.0.0
