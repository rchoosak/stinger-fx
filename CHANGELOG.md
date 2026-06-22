# Changelog

All notable changes to Stinger-Fx are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [1.2.0] — 2026-06-22

Risk-management round 2 — turn detection into **action** and add **proactive**
exposure protection for running several strategies on one account. Everything is
**opt-in / off by default**, so existing 1.1.0 configs run unchanged.

### Auto-pause (circuit breaker)
- **Strategy circuit breaker** (#94) — auto-pause a degrading strategy via
  `StrategyRunner.pause()` (enforced by `_active()`). Triggers on a drift-monitor
  alert (`pause_on_drift`) or a consecutive-loss streak
  (`max_consecutive_losses`). Idempotent; alert-only behaviour is unchanged.
  Config: `risk.circuit_breaker`.

### Reconciliation
- **Reconciler wired into the live engine** (#95) — one per account; verifies
  each fill landed at the broker after a grace period and emits a
  `ReconciliationMismatchEvent` on a discrepancy. Previously implemented but
  never instantiated. Config: `risk.reconciliation`.
- **Startup orphan-position audit** — at startup, flag every broker position not
  owned by a configured strategy (manual trades / foreign EAs) before trading
  resumes.
- **Multi-account fix** — the Reconciler now scopes fill/close handling to its
  own broker's account, so a fill on one account no longer false-flags a
  `position_missing` mismatch on every other account's Reconciler.

### Profit-lock
- **Profit-lock equity stop** (#96) — once equity rises `activate_pct` above the
  session-open watermark, arm a trailing floor; if it gives back more than
  `giveback_pct` of the gain, trip and reject all new signals until an operator
  `reset_profit_lock()`. Snapshot-driven (live + backtest); the tripped flag is
  persisted in `risk_state` and survives a restart. Config: `risk.profit_lock`.

### Exposure caps
- **Margin floor** (#97) — reject new orders when the account's margin level or
  free margin falls below a configured floor. Snapshot-driven; fails open before
  the first snapshot and when margin level is 0 (no open margin).
  Config: `risk.margin_floor`.
- **Aggregate open-risk cap** (#97) — block a new order when total open risk
  (Σ `|entry − SL| × volume × contract` over all open positions, plus the new
  order) would exceed `equity × max_aggregate_risk_pct`. Bounds the
  simultaneous-stop loss across all strategies sharing one account.
  Config: `risk.max_aggregate_risk_pct`.

## [1.1.0] — 2026-06-21

Production-safety hardening, backtest cost/size fidelity, and trading-quality
features. Everything is **opt-in / off by default**, so existing 1.0.0 configs
run unchanged. New SQLite tables (`risk_state`) are created automatically.

### Risk & crash recovery
- **RiskMonitor rehydration** — on restart the engine rebuilds open-position
  counts (from the broker), today's realized P&L (from the trade log), and
  restores peak equity + a tripped kill switch from a persisted `risk_state`
  row. A restart no longer silently resets the daily-loss cap or un-trips the
  kill switch.
- **Kill-switch reset** now rebases the drawdown peak to current equity, so a
  reset during an active drawdown actually lets trading resume instead of
  re-tripping on the next snapshot.
- **Live trade persistence** — fully- and partially-closed positions are written
  to the `trades` table, which feeds daily-loss recovery and the drift monitor.
- **Hardened realized P&L** — per-account ticket attribution, partial-close P&L
  counted toward the daily limit, and net P&L taken from MT5 deal history
  (commission/swap included), queried by the close order's ticket.
- OrderRouter idempotency contract documented and locked with a test.

### Position sizing
- **Risk-based position sizing** — size each order so it risks a configured % of
  account equity at its stop (`risk.position_sizing`), rounded down to the
  symbol's lot step. Lives in the shared order router, so live and backtests
  size identically.

### Pre-trade trading filter
- **Engine-level filter** (`risk.trading_filter`) — blocks orders on wide spread,
  outside a UTC session window (wraps midnight), within ± minutes of the daily
  rollover, or inside a news-blackout window. Applied in the shared router so
  backtests model the same guard; news windows must be timezone-aware.

### Backtest fidelity
- **Commission + swap** charged in the SimBroker fill — per-side commission and
  per-night swap flow through net P&L, the equity curve, the trade log, and the
  live Orders table (`total_commission` / `total_swap` in the metrics).
- SimBroker reports the live **spread** from the last quote so the spread filter
  is meaningful in backtests.

### Observability
- **Live-vs-backtest drift monitor** (`risk.drift_monitor`) — alerts (log +
  notification sink, event `strategy_drift`) when a strategy's recent live
  win-rate or **per-lot** expectancy falls below its backtest baseline, with
  hysteresis to avoid spam.

[1.1.0]: https://github.com/rchoosak/stinger-fx/releases/tag/v1.1.0

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
