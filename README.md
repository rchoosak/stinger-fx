# Stinger-Fx

EA (Expert Advisor) Bot Platform for Forex trading on MetaTrader 5.

- Trade via **MT5** — multi-account broker pool, auto-reconnect, order retry
- Pull market data across timeframes: tick, 1m, 2m, 3m, 5m, 10m, 15m, 30m, 45m, 1H, 2H, 4H, 1D, 1W, 1MN
- **Class-based Python strategies** with isolated runtime — many strategies active concurrently
- **Full order suite**: MARKET, BUY/SELL STOP & LIMIT, OCO brackets, modify, partial close
- **Position managers** (composable): trailing stop, break-even, ladder pyramiding, time-based exit, OCO
- **Backtest**: file-based replay (bar or tick-precise), MT5 Strategy Tester, parameter sweep, walk-forward
- **Optimization**: grid / Optuna (TPE) / random / genetic search with Monte Carlo bootstrap
- **15 built-in indicators**: SMA, EMA, RSI, MACD, ATR, ADX, Bollinger, Donchian, Stochastic, Ichimoku, VWAP, Keltner, CCI, Pivot Points, rolling Correlation
- **Risk management**: per-strategy + per-symbol position caps, daily loss limits, kill switch on drawdown
- **3 runtime UIs**: Normal (CLI), TUI (Textual), Web (FastAPI + HTMX + SSE)
- **Web UI**: live params editor, candlestick replay, sweep heatmaps, walk-forward folds, Monte Carlo, audit log, in-browser strategy code editor
- **Observability**: structured JSONL logs + SQLite mirror, Prometheus metrics endpoint, Telegram / Discord notifications, reconciliation engine
- **Hot-reloadable YAML config** for app, strategies, and backtests
- **Multi-account** + **--detach mode** for running engine as a background service

> **Status:** Phase 1–6 complete. **452 tests** passing.
>
> Live trading on MT5 requires Windows (the `MetaTrader5` Python package is Windows-native). Development, file-based backtests, optimization, and the full test suite run on macOS / Linux / Windows. Broker integration tests are marked `@pytest.mark.mt5` and skipped off-Windows.

---

## 1. Prerequisites

| | Required version | Notes |
|---|---|---|
| Python | **3.12+** | check with `python --version` |
| [uv](https://docs.astral.sh/uv/) | **0.4+** | Python package manager — replaces pip / venv / pip-tools |
| Git | any | for cloning |
| MetaTrader 5 terminal | latest | **Windows only**; needed for live trading and the Strategy Tester backtest mode |
| ZeroMQ DLL (`libzmq`) | 4.3+ | **Windows + MT5 Strategy Tester only**; needed by the MQL5 shim EA |

Install `uv` (one-time):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## 2. Clone & install

```bash
git clone https://github.com/rchoosak/stinger-fx.git
cd stinger-fx
```

Pick the right install command for your platform + use case:

```bash
# macOS / Linux — develop and run file-based backtests
uv sync --extra dev

# Windows — adds the MetaTrader5 SDK
uv sync --extra dev --extra mt5

# Add Bayesian optimization (Optuna)
uv sync --extra dev --extra optimize

# Add pairs-trading ADF test (statsmodels)
uv sync --extra dev --extra pairs

# Everything
uv sync --extra dev --extra mt5 --extra optimize --extra pairs
```

`uv sync` reads `pyproject.toml` + `uv.lock`, creates a project virtualenv at `.venv/`, and installs everything. All subsequent commands run via `uv run …` and don't require activating the venv manually.

Verify the install:

```bash
uv run stinger-fx version
# → stinger-fx 0.1.0
```

---

## 3. Initial configuration

The repo ships with three YAML files under `config/` that work out of the box for development:

| File | What it controls |
|---|---|
| `config/app.yaml` | runtime mode (`normal` / `tui` / `web`), broker(s), log level, web bind, risk caps, notifications |
| `config/strategies.yaml` | the list of active strategies and their parameters |
| `config/backtest.yaml` | named backtest runs, parameter sweeps, walk-forward runs |

Validate them at any time:

```bash
uv run stinger-fx config validate
```

Pretty-print the merged config as JSON:

```bash
uv run stinger-fx config show
```

### Configure your MT5 account (Windows only)

Edit `config/app.yaml` and either:

- Leave `login: 0` to use whichever account is currently logged in inside the MT5 terminal, **or**
- Fill in `login`, `password`, `server` for a specific account.

```yaml
broker:
  type: mt5
  mt5:
    terminal_path: ""        # auto-detect; or "C:/Program Files/MetaTrader 5/terminal64.exe"
    login: 0                 # 0 = use whichever account is logged into the terminal
    password: ""
    server: ""
    timeout_ms: 60000
```

Multi-account setups use a `brokers:` list instead of singular `broker:` — see [`config/app.yaml`](config/app.yaml) for the schema.

> Files are watched at runtime. Most changes are picked up live without restarting — broker switches, web `host/port`, and `mode` require a restart.

---

## 4. Initialize the database

Create the SQLite schema under `data/stinger.db`:

```bash
uv run stinger-fx db migrate
```

`data/` is gitignored and holds everything the runtime generates:

```
data/
├── parquet/{symbol}/{tf}/*.parquet     # tick + bar history
├── backtests/                          # equity curves, metrics, trade JSON sidecars
├── sweeps/                             # parameter-sweep summaries
├── walk_forward/                       # walk-forward fold-by-fold results
├── logs/                               # structured JSONL per category
└── stinger.db                          # SQLite — orders, trades, decisions, audit
```

---

## 5. Run a backtest (works on any OS)

This is the fastest way to confirm the whole pipeline works end-to-end before touching MT5.

### 5.1 Get historical data

If you're on Windows with MT5 connected:

```bash
uv run stinger-fx data download \
    --symbol EURUSD \
    --timeframe M15 \
    --start 2024-01-01 \
    --end 2024-04-01
```

Or import a Dukascopy-style CSV tick file on any OS:

```bash
uv run stinger-fx data import-ticks \
    --symbol EURUSD --csv ~/Downloads/eurusd_2024_01.csv --tz UTC
```

### 5.2 Run a configured backtest

```bash
uv run stinger-fx backtest run --run-id ma_eurusd_m15_2024Q1
```

For **tick-precise** SL/TP and pending-order triggering, set `granularity: tick` in the run config.

Artifacts land in `data/backtests/`:

- `<run-id>_metrics.json` — Net P&L, Sharpe, MaxDD, profit factor, win rate, expectancy
- `<run-id>_equity.parquet` — equity curve
- `<run-id>_trades.json` — trade-by-trade detail (consumed by the Web replay viewer)
- `backtest_runs` row in `data/stinger.db`

### 5.3 Parameter sweep

Define a sweep in `config/backtest.yaml`:

```yaml
sweeps:
  - id: ma_optuna_2024
    strategy_id: ma_crossover
    algo: optuna           # grid | optuna | random | genetic
    n_trials: 50
    random_seed: 42
    rank_by: net_pnl
    parameter_grid:
      fast: [5, 8, 10, 13, 15]
      slow: [20, 30, 40, 50, 60]
```

Run it:

```bash
uv run stinger-fx backtest sweep --sweep-id ma_optuna_2024
```

### 5.4 Walk-forward validation

```yaml
walk_forwards:
  - id: ma_wf_2024
    strategy_id: ma_crossover
    n_folds: 6
    in_sample_pct: 0.7
    scheme: expanding          # or "rolling"
    algo: optuna
    n_trials: 30
    parameter_grid:
      fast: [5, 8, 10, 13]
      slow: [20, 30, 40, 50]
```

The Web UI viewer shows in-sample vs out-of-sample metrics per fold with a Pearson **consistency score** (positive = generalises, negative = overfit).

---

## 6. Run live (Windows + MT5 only)

1. **Start the MT5 terminal** and log in to your account (demo recommended for first run).
2. **Enable algorithmic trading** in MT5: `Tools → Options → Expert Advisors → Allow algorithmic trading`.
3. **Run Stinger-Fx**:

```bash
uv run stinger-fx run
# uses --mode from config/app.yaml; --mode normal by default
```

You should see:

- structured status lines printed to stdout (`[INFO] order_filled symbol=EURUSD ...`)
- `data/logs/engine.jsonl` and category-specific JSONL files growing
- order/position/account rows landing in SQLite
- if web mode: a Web UI on `http://127.0.0.1:8765` (configurable)

Stop with `Ctrl+C` — the engine drains in-flight events, disconnects the broker, and flushes logs.

### Detached mode (background service)

```bash
# Start in background
uv run stinger-fx run --detach

# Check status
uv run stinger-fx status

# Stop
uv run stinger-fx stop
```

### Live reliability features

When live, the engine:

- **Auto-reconnects** MT5 with exponential backoff (1s → 60s) if the terminal drops
- **Retries** transient order errors (REQUOTE, REJECT, PRICE_OFF, TIMEOUT) up to 3× with backoff; permanent errors (INVALID_VOLUME, NO_MONEY) fail fast
- **Persists every order** via the OrderQueue outbox so a crash mid-submission is replayable on restart (idempotent via `client_order_id`)
- **Reconciles** broker positions vs internal DB 5s after every fill; mismatches go to the `/audit` page + can fire Telegram / Discord alerts
- Exposes a **Prometheus metrics endpoint** at `http://127.0.0.1:9100/metrics` (configurable) with order-submission histograms, tick lag gauges, broker disconnect counters, downtime histograms

### Hot-reload smoke test

While the engine is running, edit `config/strategies.yaml` (e.g. change `fast: 10` → `fast: 12`). Within ~500ms you should see a `config reload applied` log line and the strategy will receive a `on_params_reloaded` callback. Adding a strategy entry starts it without a restart; removing one stops it gracefully.

---

## 7. Web UI

Run with `--mode web` (or set `mode: web` in `config/app.yaml`):

```bash
uv run stinger-fx run --mode web
# → http://127.0.0.1:8765
```

| Page | Purpose |
|---|---|
| `/` | dashboard: account, recent fills, strategy states |
| `/strategies/{id}/params` | edit strategy params live (HTMX form, validated, atomic swap) |
| `/backtest/{run_id}` | equity curve + candlestick replay with entry/exit markers |
| `/backtest/{run_id}/monte_carlo.json` | bootstrap percentile bands JSON endpoint |
| `/sweep/{id}` | parameter sweep results — heatmap for 2-param, ranked table for N-param |
| `/walkforward/{id}` | per-fold in-sample vs OOS breakdown + consistency score |
| `/audit` | recent decisions, order modifications, reconciliation mismatches |
| `/editor` | in-browser strategy code editor (CodeMirror) — create / edit / save with AST validation |

Enable the in-browser editor by passing `user_strategies_dir`:

```bash
uv run stinger-fx run --mode web --user-strategies-dir ./user_strategies
```

---

## 8. MT5 Strategy Tester backtest (Windows only)

This runs the same Python strategy through MT5's official Strategy Tester via the MQL5 shim EA shipped at `src/stinger_fx/backtest/mt5_shim/stinger_fx_shim.mq5`. See [`src/stinger_fx/backtest/mt5_shim/README.md`](src/stinger_fx/backtest/mt5_shim/README.md) for the full setup. High-level steps:

1. Copy `libzmq-mt-4_3_5.dll` into `<MT5>/MQL5/Libraries/`.
2. Copy `stinger_fx_shim.mq5` into `<MT5>/MQL5/Experts/StingerFx/` and compile it in MetaEditor → produces `stinger_fx_shim.ex5`.
3. Enable *Allow DLL imports* in MT5 options.
4. Run:

```bash
uv run stinger-fx backtest run --run-id ma_eurusd_m15_2024Q1_mt5
```

Python spawns `terminal64.exe` in tester mode, hosts a ZeroMQ REP socket, feeds ticks from the shim through the same engine path used in live mode, and parses the MT5 report on completion.

---

## 9. Writing your own strategy

### Option A: scaffold from the CLI

```bash
uv run stinger-fx strategy scaffold my_breakout \
    --dir ./user_strategies
```

This drops a minimal template at `./user_strategies/my_breakout.py` you can fill in.

### Option B: scaffold from the Web UI

Browse to `/editor` (with the editor enabled — see §7), click "Create new", name it, edit, save. Hot-reload picks up the file the moment it's written.

### Strategy anatomy

```python
from pydantic import Field
from stinger_fx.domain import Bar, Subscription, Timeframe
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.parameters import StrategyParams
from stinger_fx.strategies.indicators import sma, adx
from stinger_fx.strategies.managers.trailing import TrailingStopManager
from stinger_fx.strategies.regime import TrendingFilter


class MyBreakoutParams(StrategyParams):
    symbol: str = "EURUSD"
    timeframe: Timeframe = Timeframe.M15
    fast: int = Field(10, ge=2, le=200)
    slow: int = Field(30, ge=5, le=500)
    volume: float = Field(0.01, gt=0)


class MyBreakout(BaseStrategy):
    name = "my_breakout"
    Params = MyBreakoutParams

    @classmethod
    def subscriptions(cls, params):
        return [Subscription(symbol=params.symbol, timeframe=params.timeframe)]

    async def on_start(self, ctx):
        # Attach a position manager — trails SL by 15 pips
        ctx.attach_manager(TrailingStopManager(ctx, distance_pips=15))
        # Only trade in trending markets
        self._filter = TrendingFilter(threshold=25.0)

    async def on_bar(self, ctx, bar):
        bars = ctx.history.bars()
        if not self._filter.allows(bars):
            return  # regime says "no, this is chop"

        closes = ctx.history.closes()
        fast_now = sma(closes, ctx.params.fast)
        slow_now = sma(closes, ctx.params.slow)
        if fast_now is None or slow_now is None:
            return
        if fast_now > slow_now and self._was_below:
            await ctx.buy_stop(price=bar.high + 0.0010, volume=ctx.params.volume)
        self._was_below = fast_now < slow_now
```

Add an entry in `config/strategies.yaml`:

```yaml
strategies:
  - id: my_breakout
    class_path: user_strategies.my_breakout:MyBreakout
    enabled: true
    params:
      fast: 8
      slow: 21
      volume: 0.05
```

If the engine is running, the strategy starts immediately via hot-reload. Otherwise: `uv run stinger-fx run`.

### Available primitives

**Trading actions on `ctx`:**

- `await ctx.buy(volume, sl=, tp=)` / `ctx.sell(...)` — market orders
- `await ctx.buy_stop(price, volume, ...)` / `sell_stop`, `buy_limit`, `sell_limit` — pending orders
- `await ctx.move_stop(ticket, sl=, tp=)` — update SL/TP on a position
- `await ctx.move_pending(ticket, price=, volume=, ...)` — adjust a pending order in flight
- `await ctx.partial_close(ticket, volume)` — reduce a position
- `await ctx.close(ticket)` — full close
- `await ctx.cancel_order(ticket)` — cancel a pending order

**Position managers** (composable via `ctx.attach_manager`):

- `TrailingStopManager` — ratchets SL toward market
- `BreakEvenMover` — moves SL to entry once profit hits trigger
- `LadderManager` — pyramids in as price moves favourably
- `TimeExitManager` — closes after N seconds or N bars
- `OCOGroupManager` — one-cancels-other for pending or position groups

**Indicators** (all importable from `stinger_fx.strategies.indicators`):

- Trend: `sma`, `ema`, `macd`, `adx`, `ichimoku`
- Momentum: `rsi`, `stochastic`, `cci`
- Volatility: `atr`, `bollinger`, `keltner`, `donchian`
- Volume: `vwap_rolling`, `vwap_session`
- S/R: `pivot_points` (classic / fibonacci / camarilla)
- Cross-asset: `correlation`

**Regime filters** (from `stinger_fx.strategies.regime`):

- `TrendingFilter` / `RangingFilter` — ADX-based
- `HighVolatilityFilter` / `LowVolatilityFilter` — ATR-percentile
- `CompositeFilter(...)` — logical AND of any subset

**Pairs trading** (from `stinger_fx.strategies.cointegration`):

- `engle_granger_test(a, b)` — two-step cointegration test (ADF via statsmodels if installed, heuristic otherwise)
- `rolling_hedge_ratio(a, b, window)` / `spread_zscore(spread, window)`

See the working examples in [`src/stinger_fx/strategies/examples/`](src/stinger_fx/strategies/examples/) — `ma_crossover.py`, `pairs_trading.py`, `regime_filtered_ma.py`.

---

## 10. CLI reference

```
stinger-fx version
stinger-fx run [--mode normal|tui|web] [--detach] [--config-dir ./config]
                [--user-strategies-dir ./user_strategies]
stinger-fx status
stinger-fx stop

stinger-fx backtest run --run-id <id> [--config-dir ./config]
stinger-fx backtest list
stinger-fx backtest show <run-id>
stinger-fx backtest sweep --sweep-id <id>

stinger-fx config validate [--config-dir ./config]
stinger-fx config show

stinger-fx strategy list
stinger-fx strategy scaffold <name> [--dir ./user_strategies]

stinger-fx data download --symbol <S> --timeframe <TF> --start <YYYY-MM-DD> [--end <YYYY-MM-DD>]
stinger-fx data import-ticks --symbol <S> --csv <PATH> [--tz UTC]

stinger-fx db migrate
```

---

## 11. Development workflow

```bash
# Run the full test suite (452 tests; ~6s on a modern laptop)
uv run pytest -q

# Lint + format
uv run ruff check src tests
uv run ruff format src tests

# Type-check (strict on core/ and domain/)
uv run mypy src/stinger_fx
```

---

## 12. Project layout

```
stinger-fx/
├── config/                          # YAML config (app / strategies / backtest)
├── data/                            # gitignored runtime artifacts
├── src/stinger_fx/
│   ├── core/                        # event bus, engine, clock, scheduler
│   ├── domain/                      # frozen Pydantic value objects
│   ├── brokers/                     # BaseBroker, MT5Broker, BrokerPool,
│   │                                #   OrderQueue, BarAggregator
│   ├── strategies/                  # BaseStrategy, runner, indicators (15),
│   │                                #   managers (5), regime, cointegration,
│   │                                #   examples
│   ├── config/                      # Pydantic schemas, YAML loader, watcher
│   ├── data/                        # SQLite + Parquet stores, repositories,
│   │                                #   reconciliation, modification_logger,
│   │                                #   CSV importer
│   ├── backtest/                    # FileBacktester, MT5StrategyTester,
│   │                                #   ParameterSweep, WalkForward,
│   │                                #   MonteCarlo, search (grid/optuna/
│   │                                #   random/genetic), slippage models
│   ├── risk/                        # RiskMonitor with per-strategy +
│   │                                #   per-symbol limits + kill switch
│   ├── observability/               # Prometheus metrics, notifications
│   ├── log/                         # structlog setup
│   ├── ui/                          # EngineHandle + Normal / TUI / Web UI
│   │                                #   + strategy editor
│   ├── runtime.py                   # engine assembler
│   └── cli.py                       # Typer entrypoint
└── tests/                           # 452 tests across unit + integration
```

---

## 13. Troubleshooting

**`uv run stinger-fx ...` says command not found**
You're inside an unrelated venv. Either run from the repo root so `uv` picks up `pyproject.toml`, or explicitly `uv run --project /path/to/stinger-fx stinger-fx ...`.

**`broker.type=mt5 but broker.mt5 block is missing`**
`config/app.yaml` is missing the `broker.mt5:` block — even an empty `mt5: {}` works.

**`MetaTrader5 SDK is unavailable`**
You're not on Windows, or you installed without `--extra mt5`. Re-run `uv sync --extra dev --extra mt5` on Windows.

**`MT5 initialize() failed`**
The MT5 terminal isn't running, or `Tools → Options → Expert Advisors → Allow algorithmic trading` is off, or the credentials in `config/app.yaml` don't match a live session.

**`optuna is required for OptunaSearch`**
Install the optimize extra: `uv sync --extra dev --extra optimize`.

**Backtest reports `trades: 0`**
There's no Parquet data under the configured `data_source`. Run `stinger-fx data download` (Windows) or `data import-ticks` (any OS) first.

**Hot reload didn't take effect**
Check `data/logs/config.jsonl` for the audit row — if validation failed, the old config is kept and the error is logged. Use `stinger-fx config validate` to see the exact message.

**Strategy editor returns 503**
`create_app` was called without `user_strategies_dir`. Start the engine with `--user-strategies-dir ./user_strategies`.

**Reconciliation page shows mismatches**
After every fill, the engine waits 5s then queries the broker for the resulting position. Any divergence (volume drift, price drift > 2 pips, missing position) becomes a row in `/audit`. Common cause: another EA on the same MT5 account interfering with the magic-number-tagged positions.

---

## License

MIT — see [LICENSE](LICENSE).
