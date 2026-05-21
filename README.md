# Stinger-Fx

EA (Expert Advisor) Bot Platform for Forex trading.

- Trade via **MT5** (Phase 1) and **MT4** (Phase 2), configurable per app
- Pull market data from broker across timeframes: tick, 1m, 2m, 3m, 5m, 10m, 15m, 30m, 45m, 1H, 2H, 4H, 1D, 1W, 1MN
- **Class-based Python strategies** with isolated runtime — many strategies active concurrently
- **Backtest** via MT5 Strategy Tester or file-based replay (same strategy code as live)
- Three runtime UIs: **Normal** (CLI), **TUI** (Textual, Phase 2), **Web** (FastAPI + HTMX, Phase 2)
- **Structured JSONL logs** + SQLite mirror for trade analytics
- **Hot-reloadable YAML config** for app, strategies, and backtests

> **Status:** Phase 1 ready. Live trading on MT5 requires Windows (the `MetaTrader5` Python package is Windows-native). Development, file-based backtests, and the full test suite run on macOS / Linux / Windows. Broker integration tests are marked `@pytest.mark.mt5` and skipped off-Windows.

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

Pick the right install command for your platform:

```bash
# macOS / Linux — develop and run file-based backtests (no MT5 SDK)
uv sync --extra dev

# Windows — adds the MetaTrader5 SDK on top of dev tooling
uv sync --extra dev --extra mt5
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
| `config/app.yaml` | runtime mode (`normal` / `tui` / `web`), broker selection, log level, web bind, risk caps |
| `config/strategies.yaml` | the list of active strategies and their parameters |
| `config/backtest.yaml` | named backtest runs (file or MT5-tester) |

Validate them at any time:

```bash
uv run stinger-fx config validate
# → OK: app.mode=normal broker=mt5 strategies=1 backtest_runs=2
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

> Files are watched at runtime. Most changes are picked up live without restarting — broker switches, web `host/port`, and `mode` require a restart.

---

## 4. Initialize the database

Create the SQLite schema under `data/stinger.db`:

```bash
uv run stinger-fx db migrate
# → OK: schema applied at data/stinger.db
```

`data/` is gitignored and holds everything the runtime generates: SQLite DB, Parquet partitions under `parquet/`, backtest reports under `backtests/`, JSONL logs under `logs/`.

---

## 5. Run a backtest (works on any OS)

This is the fastest way to confirm the whole pipeline works end-to-end before touching MT5.

### 5.1 Download historical bars (Windows + MT5 only)

If you're on Windows with MT5 connected, pull historical data into Parquet:

```bash
uv run stinger-fx data download \
    --symbol EURUSD \
    --timeframe M15 \
    --start 2024-01-01 \
    --end 2024-04-01
```

Bars are written to `data/parquet/EURUSD/M15/<yyyy-mm-dd>.parquet`.

> On macOS / Linux you can't use MT5, but the file backtester runs on any Parquet data of the schema documented in [`src/stinger_fx/data/parquet_store.py`](src/stinger_fx/data/parquet_store.py) (the integration test seeds synthetic bars on the fly).

### 5.2 Run a configured backtest

```bash
uv run stinger-fx backtest run --run-id ma_eurusd_m15_2024Q1_file
```

You'll see a metrics summary on stdout. The full artifacts land in `data/backtests/`:

- `data/backtests/<run-id>_metrics.json` — Net P&L, Sharpe, MaxDD, profit factor, win rate, expectancy
- `data/backtests/<run-id>_equity.parquet` — equity curve
- `backtest_runs` row in `data/stinger.db`

List past runs:

```bash
uv run stinger-fx backtest list
uv run stinger-fx backtest show <run-id>
```

---

## 6. Run live (Windows + MT5 only)

1. **Start the MT5 terminal** and log in to your account (demo recommended for first run).
2. **Enable algorithmic trading** in MT5: `Tools → Options → Expert Advisors → Allow algorithmic trading`. For Strategy Tester you also need `Allow DLL imports`.
3. **Run Stinger-Fx**:

```bash
uv run stinger-fx run
# (uses --mode from config/app.yaml; --mode normal by default)
```

You should see:

- structured status lines printed to stdout (`[INFO] order_filled symbol=EURUSD ...`)
- `data/logs/engine.jsonl` and category-specific JSONL files growing
- order/position/account rows landing in SQLite

Stop with `Ctrl+C` — the engine drains in-flight events, disconnects the broker, and flushes logs.

### Hot-reload smoke test

While the engine is running, edit `config/strategies.yaml` (e.g. change `fast: 10` → `fast: 12`). Within ~500ms you should see a `config reload applied` log line and the strategy will receive a `on_params_reloaded` callback. Adding a strategy entry starts it without a restart; removing one stops it gracefully.

---

## 7. MT5 Strategy Tester backtest (Windows only, optional)

This runs the same Python strategy through MT5's official Strategy Tester via the MQL5 shim EA shipped at `src/stinger_fx/backtest/mt5_shim/stinger_fx_shim.mq5`. See the full instructions in [`src/stinger_fx/backtest/mt5_shim/README.md`](src/stinger_fx/backtest/mt5_shim/README.md). High-level steps:

1. Copy `libzmq-mt-4_3_5.dll` into `<MT5>/MQL5/Libraries/`.
2. Copy `stinger_fx_shim.mq5` into `<MT5>/MQL5/Experts/StingerFx/` and compile it in MetaEditor → produces `stinger_fx_shim.ex5`.
3. Enable *Allow DLL imports* in MT5 options.
4. Run:

```bash
uv run stinger-fx backtest run --run-id ma_eurusd_m15_2024Q1_mt5
```

Python spawns `terminal64.exe` in tester mode, hosts a ZeroMQ REP socket on `tcp://127.0.0.1:5555`, feeds ticks from the shim through the same engine path used in live mode, and parses the MT5 report on completion.

---

## 8. Writing your own strategy

1. Create a Python file under `src/stinger_fx/strategies/` (or any importable package).
2. Subclass `BaseStrategy`, declare typed parameters with `StrategyParams`, declare your `(symbol, timeframe)` subscriptions, and override one or more lifecycle hooks (`on_bar`, `on_tick`, `on_order_filled`, …). See [`src/stinger_fx/strategies/examples/ma_crossover.py`](src/stinger_fx/strategies/examples/ma_crossover.py) for a minimal template.
3. Add an entry in `config/strategies.yaml`:

```yaml
strategies:
  - id: my_strategy
    class_path: stinger_fx.strategies.examples.ma_crossover:MACrossover
    enabled: true
    params:
      fast: 8
      slow: 21
      volume: 0.05
```

4. If the engine is already running, the new strategy starts automatically (hot-reload). Otherwise: `uv run stinger-fx run`.

List currently-configured strategies any time:

```bash
uv run stinger-fx strategy list
```

---

## 9. CLI reference

```
stinger-fx version
stinger-fx run [--mode normal|tui|web] [--config-dir ./config]
stinger-fx backtest run --run-id <id> [--config-dir ./config]
stinger-fx backtest list
stinger-fx backtest show <run-id>
stinger-fx config validate [--config-dir ./config]
stinger-fx config show
stinger-fx strategy list
stinger-fx data download --symbol <S> --timeframe <TF> --start <YYYY-MM-DD> [--end <YYYY-MM-DD>]
stinger-fx db migrate
```

---

## 10. Development workflow

```bash
# Run the test suite (47 tests; ~1s on a modern laptop)
uv run pytest -q

# Lint + format
uv run ruff check src tests
uv run ruff format src tests

# Type-check (strict on core/ and domain/)
uv run mypy src/stinger_fx
```

The Phase-1 test suite covers: event bus pub/sub + overflow policies, clock invariants, Pydantic config validation, hot-reload diff categorisation, indicator math, strategy quarantine + parameter hot-swap, backtest metrics, and a deterministic end-to-end file backtest.

---

## 11. Project layout (Phase 1)

```
stinger-fx/
├── config/                          # YAML config (app / strategies / backtest)
├── data/                            # gitignored runtime artifacts
│   ├── parquet/{symbol}/{tf}/*.parquet
│   ├── backtests/                   # equity curves + metrics
│   ├── logs/                        # JSONL logs per category
│   └── stinger.db                   # SQLite
├── src/stinger_fx/
│   ├── core/                        # event bus, engine, clock, scheduler
│   ├── domain/                      # frozen Pydantic value objects
│   ├── brokers/                     # BaseBroker + MT5Broker + BarAggregator
│   ├── strategies/                  # BaseStrategy, runner, indicators, examples
│   ├── config/                      # Pydantic schemas, YAML loader, watcher, reloader
│   ├── data/                        # SQLite + Parquet stores, repositories
│   ├── backtest/                    # FileBacktester, MT5StrategyTester, MQL5 shim
│   ├── log/                         # structlog setup
│   ├── ui/                          # EngineHandle + NormalUI (TUI/Web in Phase 2)
│   ├── runtime.py                   # engine assembler
│   └── cli.py                       # Typer entrypoint
└── tests/                           # unit + integration
```

---

## 12. Troubleshooting

**`uv run stinger-fx ...` says command not found**
You're inside an unrelated venv. Either run from the repo root so `uv` picks up `pyproject.toml`, or explicitly `uv run --project /path/to/stinger-fx stinger-fx ...`.

**`broker.type=mt5 but broker.mt5 block is missing`**
`config/app.yaml` is missing the `broker.mt5:` block — even an empty `mt5: {}` works.

**`MetaTrader5 SDK is unavailable`**
You're not on Windows, or you installed without `--extra mt5`. Re-run `uv sync --extra dev --extra mt5` on Windows.

**`MT5 initialize() failed`**
The MT5 terminal isn't running, or `Tools → Options → Expert Advisors → Allow algorithmic trading` is off, or the credentials in `config/app.yaml` don't match a live session.

**Backtest reports `trades: 0`**
There's no Parquet data under the configured `data_source`. Run `stinger-fx data download` first (or check that your custom path is correct).

**Hot reload didn't take effect**
Check `data/logs/config.jsonl` for the audit row — if validation failed, the old config is kept and the error is logged. Use `stinger-fx config validate` to see the exact message.

---

## License

MIT — see [LICENSE](LICENSE).
