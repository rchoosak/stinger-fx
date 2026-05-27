# Strategy Development Guide

A comprehensive guide to building, testing, and optimising trading strategies on Stinger-Fx.

This document covers everything from "hello-world" to advanced patterns: multi-feed strategies, pending-order brackets, regime filtering, pairs trading, walk-forward validation, and custom metric optimisation.

---

## Table of contents

1. [Quick start — 30 lines to your first strategy](#1-quick-start--30-lines-to-your-first-strategy)
2. [The strategy class anatomy](#2-the-strategy-class-anatomy)
3. [Lifecycle hooks](#3-lifecycle-hooks)
4. [`StrategyContext` API reference](#4-strategycontext-api-reference)
5. [Indicators (15 built-in)](#5-indicators-15-built-in)
6. [Position managers (composable)](#6-position-managers-composable)
7. [Regime filters](#7-regime-filters)
8. [Multi-feed strategies](#8-multi-feed-strategies)
9. [Pending orders + OCO brackets](#9-pending-orders--oco-brackets)
10. [Hot reload + parameter swap](#10-hot-reload--parameter-swap)
11. [Pairs trading template](#11-pairs-trading-template)
12. [Testing strategies — file backtests](#12-testing-strategies--file-backtests)
13. [Optimisation — sweep, walk-forward, Pareto, custom metrics](#13-optimisation--sweep-walk-forward-pareto-custom-metrics)
14. [Risk management](#14-risk-management)
15. [Logging + observability](#15-logging--observability)
16. [Common pitfalls](#16-common-pitfalls)
17. [Best practices](#17-best-practices)
18. [Worked examples](#18-worked-examples)

---

## 1. Quick start — 30 lines to your first strategy

Create `user_strategies/my_first.py`:

```python
from pydantic import Field

from stinger_fx.domain import Bar, Subscription, Timeframe
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.indicators import sma
from stinger_fx.strategies.parameters import StrategyParams


class MyFirstParams(StrategyParams):
    symbol: str = "EURUSD"
    timeframe: Timeframe = Timeframe.M15
    fast: int = Field(10, ge=2, le=200)
    slow: int = Field(30, ge=5, le=500)
    volume: float = Field(0.01, gt=0)


class MyFirst(BaseStrategy):
    name = "my_first"
    Params = MyFirstParams

    @classmethod
    def subscriptions(cls, params):
        return [Subscription(symbol=params.symbol, timeframe=params.timeframe)]

    async def on_bar(self, ctx: StrategyContext, bar: Bar) -> None:
        closes = ctx.history.closes()
        if len(closes) < ctx.params.slow + 2:
            return
        fast = sma(closes, ctx.params.fast)
        slow = sma(closes, ctx.params.slow)
        if fast is not None and slow is not None and fast > slow:
            if not ctx.position.for_symbol(ctx.symbol):
                await ctx.buy(ctx.params.volume)
```

Register it in `config/strategies.yaml`:

```yaml
strategies:
  - id: my_first_001
    class_path: user_strategies.my_first:MyFirst
    enabled: true
    params:
      fast: 10
      slow: 30
      volume: 0.05
```

Run it:

```bash
uv run stinger-fx run               # live (Windows + MT5)
uv run stinger-fx backtest run --run-id my_first_test    # backtest
```

That's the complete loop. Everything below is the *why* and the *how-to-extend*.

---

## 2. The strategy class anatomy

Every strategy has four parts:

```python
class MyStrategyParams(StrategyParams):       # 1. Typed parameters (Pydantic)
    ...

class MyStrategy(BaseStrategy):
    name = "my_strategy"                      # 2. Identifier (used in logs)
    Params = MyStrategyParams                 # 3. Wire params class

    @classmethod
    def subscriptions(cls, params):           # 4. Which feeds we need
        return [...]

    async def on_bar(self, ctx, bar):         # 5+. Lifecycle hooks
        ...
```

### 2.1 `StrategyParams` subclass

Define typed parameters with Pydantic `Field()` constraints. Validation happens once at startup and again on every hot-reload — invalid params keep the old config running and surface the error in the audit log.

```python
from pydantic import Field
from stinger_fx.strategies.parameters import StrategyParams
from stinger_fx.domain import Timeframe

class MAParams(StrategyParams):
    symbol: str = "EURUSD"
    timeframe: Timeframe = Timeframe.M15            # Pydantic auto-coerces "M15" → enum
    fast: int = Field(10, ge=2, le=200)             # bounded — caught at load time
    slow: int = Field(30, ge=5, le=500)
    risk_per_trade: float = Field(0.01, gt=0, le=0.1)  # 0 < x ≤ 10 %
    enabled_during_news: bool = False               # bool flag
```

The class is a regular Pydantic v2 model — full validation, type coercion, JSON schema generation, the works.

### 2.2 `name` and `version`

```python
class MyStrategy(BaseStrategy):
    name = "my_strategy"      # appears in logs, metrics labels, magic number derivation
    version = "1.2.0"         # informational; useful for A/B in walk-forward
```

`name` should be `snake_case`. The engine derives a stable 63-bit **magic number** from `strategy_id` (the YAML `id:` field) — that magic tags every order, position, and trade to prevent cross-strategy interference on shared MT5 accounts.

### 2.3 `subscriptions(params)`

Return a list of `Subscription(symbol, timeframe)` declaring which market feeds the strategy needs.

```python
@classmethod
def subscriptions(cls, params):
    return [
        Subscription(symbol=params.symbol, timeframe=params.timeframe),    # primary
        Subscription(symbol="GBPUSD", timeframe=Timeframe.H1),             # secondary
    ]
```

The **first** subscription is the primary feed — `ctx.symbol`, `ctx.timeframe`, `ctx.history` all point at it. Additional feeds are reachable via `ctx.history_for(symbol, tf)`.

For tick-mode backtests or live trading, the engine auto-derives bar feeds from ticks via `BarAggregator`. You always get `on_bar(...)` calls when bars close, regardless of whether the source is bars or ticks.

---

## 3. Lifecycle hooks

All optional. Override what you need. They're async — `await ctx.buy(...)` works directly.

| Hook | Fires when |
|---|---|
| `on_start(ctx)` | After the engine wires the strategy + before any market event |
| `on_stop(ctx)` | Before the engine drains and shuts down |
| `on_tick(ctx, tick)` | Every tick on the **primary** symbol (or any subscribed symbol — see notes) |
| `on_bar(ctx, bar)` | Every **closed** bar on any declared feed |
| `on_order_filled(ctx, order)` | A market order or pending order filled |
| `on_order_rejected(ctx, order, reason)` | A broker rejected the order (margin, invalid price, etc.) |
| `on_position_closed(ctx, position)` | A position closed (SL/TP/manual/manager) |
| `on_params_reloaded(ctx, old, new)` | After a YAML hot-reload swaps params atomically |

### 3.1 Tick vs bar

```python
async def on_tick(self, ctx, tick):
    # Fires every market quote — typically thousands per minute per symbol.
    # Don't compute heavy indicators here; use it for SL/TP precision
    # or watching for a specific price level.
    if tick.bid > self._target_price:
        await ctx.close(self._ticket)

async def on_bar(self, ctx, bar):
    # Fires only when a bar CLOSES. Compute indicators here.
    closes = ctx.history.closes()
    fast = sma(closes, ctx.params.fast)
```

Rule of thumb: **bar handlers for indicator-based logic; tick handlers for price-level alerts and managers**.

### 3.2 `on_start` is your constructor

The strategy instance lives across reloads — `on_start` runs once per engine session. Initialise per-instance state here:

```python
async def on_start(self, ctx):
    self._last_signal = None
    self._cooldown_until = ctx.clock.now()
    # Attach managers — they live for the strategy's lifetime
    ctx.attach_manager(TrailingStopManager(ctx, distance_pips=15))
    ctx.attach_manager(BreakEvenMover(ctx, trigger_pips=20))
```

### 3.3 Reacting to fills

Pending orders (`buy_stop`, `buy_limit`, etc.) and OCO managers fire `on_order_filled` AFTER the trigger:

```python
async def on_order_filled(self, ctx, order):
    if order.type == OrderType.STOP:
        # Our breakout entry just triggered — register for OCO
        self._oco.add(order.ticket, group_id=self._current_bracket_id)
```

### 3.4 Reacting to closes

```python
async def on_position_closed(self, ctx, position):
    ctx.logger.info(
        "position_closed",
        ticket=position.ticket,
        pnl=position.profit,
    )
    # Maybe reset state for next trade
    self._in_position = False
```

### 3.5 Hot reload

When the operator edits `config/strategies.yaml`, the engine validates the new params, atomically swaps `ctx.params`, then calls:

```python
async def on_params_reloaded(self, ctx, old, new):
    # Both old and new are full Params instances.
    # ctx.params is already pointing at `new` by the time we run.
    if old.fast != new.fast:
        ctx.logger.info("fast period changed", from_=old.fast, to=new.fast)
        self._cached_fast_sma = None    # invalidate caches
```

You don't have to do anything here — most strategies just let the new params take effect on the next bar. Hook in only when you need to invalidate caches or reset state.

---

## 4. `StrategyContext` API reference

`ctx` is the only safe way to interact with the engine. Every method is annotated; this section documents the API surface.

### 4.1 Read-only state

| Attribute | Description |
|---|---|
| `ctx.strategy_id` | YAML `id:` for this instance |
| `ctx.symbol` | Primary feed symbol |
| `ctx.timeframe` | Primary feed timeframe |
| `ctx.params` | Current `StrategyParams` (atomically swapped on reload) |
| `ctx.clock` | `LiveClock` or `SimClock` — `.now()` is timezone-aware UTC |
| `ctx.logger` | `structlog.BoundLogger` — `strategy_id` and `symbol` are pre-bound |
| `ctx.magic` | 63-bit magic number derived from `strategy_id` |
| `ctx.position` | `PositionView` filtered by magic (only sees this strategy's positions) |
| `ctx.history` | `HistoryView` for the **primary** feed |
| `ctx.histories` | `dict[Subscription, HistoryView]` for every declared feed |
| `ctx.managers` | Read-only list of attached position managers |

### 4.2 Market data — `HistoryView`

```python
view = ctx.history                              # primary feed
view2 = ctx.history_for("GBPUSD", Timeframe.H1) # any declared feed

view.bars(n=50)     # → tuple[Bar, ...] — last 50 closed bars (or all if n=None)
view.closes(n=50)   # → list[float]    — close prices only
view.last_tick()    # → Tick | None    — most recent tick observed
```

Bars are **closed bars only** — the still-forming bar isn't in `bars()`. If you need it, look at `last_tick()`.

### 4.3 Position data — `PositionView`

```python
ctx.position.all()                    # → list[Position]  — all positions tagged by this strategy's magic
ctx.position.for_symbol("EURUSD")     # → list[Position]  — filtered by symbol
ctx.position.net_volume("EURUSD")     # → float           — signed sum (BUY +, SELL -)
```

The view is automatically maintained by the runner — it sees `OrderFilledEvent`, `PositionClosedEvent`, `OrderModifiedEvent`, `PartialClosedEvent` for tickets matching `ctx.magic`.

### 4.4 Trading actions

#### Market orders

```python
await ctx.buy(volume=0.1, sl=1.10, tp=1.105, comment="ma_xup")
await ctx.sell(volume=0.1, sl=1.105, tp=1.10)
```

#### Pending orders

```python
# Breakouts — wait for price to reach a level, then enter
await ctx.buy_stop(price=1.1020, volume=0.1, sl=1.1010, tp=1.1050)
await ctx.sell_stop(price=1.0980, volume=0.1, sl=1.0990)

# Pullbacks — wait for price to come back to a level, then enter
await ctx.buy_limit(price=1.0980, volume=0.1, sl=1.0950)
await ctx.sell_limit(price=1.1020, volume=0.1, sl=1.1050)
```

Pending orders sit in the broker until triggered or cancelled. They emit `OrderSubmittedEvent` at placement and `OrderFilledEvent` when triggered.

#### Position modification

```python
await ctx.move_stop(ticket=123, sl=1.10)              # update SL only
await ctx.move_stop(ticket=123, sl=1.10, tp=1.105)    # both
```

#### Pending order modification

```python
await ctx.move_pending(
    ticket=456,
    price=1.1050,    # new trigger price
    volume=0.2,      # new size
    sl=1.0980,       # new SL (attaches to the eventual position)
)
```

Pass any subset of `price`, `stop_price`, `volume`, `sl`, `tp`. `None` means "leave unchanged". Raises `ValueError` if you pass nothing.

#### Closing positions / cancelling pendings

```python
await ctx.close(ticket=123, reason="strategy_exit")              # full close
await ctx.partial_close(ticket=123, volume=0.05)                 # partial close
await ctx.cancel_order(ticket=456, reason="signal_invalidated")  # cancel pending
```

Magic-number ownership is enforced by the router — you cannot accidentally close another strategy's positions.

#### Low-level signal submission

For non-primary-symbol orders (pairs trading), `ctx.buy()` is hard-coded to `ctx.symbol`. Use the lower-level signal API:

```python
from stinger_fx.domain.signals import Signal, SignalStrength
from stinger_fx.domain.positions import Side

await ctx.submit_signal(Signal(
    strategy_id=ctx.strategy_id,
    time=ctx.clock.now(),
    symbol="GBPUSD",                # ANY symbol you've subscribed to
    side=Side.SELL,
    strength=SignalStrength.NORMAL,
    suggested_volume=0.1,
    suggested_sl=1.275,
    comment="pairs_short_leg",
))
```

### 4.5 Attaching managers

```python
async def on_start(self, ctx):
    self._trail = TrailingStopManager(ctx, distance_pips=15)
    ctx.attach_manager(self._trail)
```

Managers receive `on_tick` (and optionally `on_bar` / `on_position_closed` / `on_order_filled`) **before** your strategy's hooks fire on the same event. This lets a trailing stop ratchet SL closer in *before* a strategy's tick handler sees an about-to-close position.

---

## 5. Indicators (15 built-in)

All importable from `stinger_fx.strategies.indicators`. Single-valued helpers return `float | None`; multi-valued helpers return a typed `NamedTuple`.

`None` means "not enough data yet" — every strategy should handle it:

```python
fast = sma(closes, 10)
if fast is None:
    return  # wait for more bars
```

### 5.1 Moving averages + oscillators

```python
from stinger_fx.strategies.indicators import sma, ema, rsi, macd, stochastic, cci

sma_val = sma(closes, period=20)
ema_val = ema(closes, period=20)
rsi_val = rsi(closes, period=14)                         # 0–100

macd_result = macd(closes, fast=12, slow=26, signal=9)   # NamedTuple
if macd_result:
    macd_line = macd_result.macd
    signal_line = macd_result.signal
    histogram = macd_result.histogram

stoch = stochastic(bars, k_period=14, d_period=3)        # NamedTuple
if stoch:
    k = stoch.k                # 0–100
    d = stoch.d                # 0–100

cci_val = cci(bars, period=20)   # ±100 = significant deviation
```

### 5.2 Volatility envelopes

```python
from stinger_fx.strategies.indicators import atr, bollinger, donchian, keltner

atr_val = atr(bars, period=14)

bb = bollinger(closes, period=20, stddev_mult=2.0)
if bb:
    upper, middle, lower = bb.upper, bb.middle, bb.lower

dc = donchian(bars, period=20)
if dc:
    upper, lower = dc.upper, dc.lower

kelt = keltner(bars, ema_period=20, atr_period=10, atr_mult=2.0)
if kelt:
    upper, middle, lower = kelt.upper, kelt.middle, kelt.lower
```

### 5.3 Trend strength + direction

```python
from stinger_fx.strategies.indicators import adx

result = adx(bars, period=14)
if result and result.adx > 25:
    if result.plus_di > result.minus_di:
        # confirmed uptrend
```

ADX < 20: chop. 20–40: confirmed trend. > 40: strong (often exhausted).

### 5.4 Volume-weighted price

```python
from stinger_fx.strategies.indicators import vwap_rolling, vwap_session

# Rolling — last N bars
vwap = vwap_rolling(bars, period=20)

# Session — caller supplies session-filtered bars
todays_bars = [b for b in ctx.history.bars() if b.time >= session_start]
vwap = vwap_session(todays_bars)
```

### 5.5 Ichimoku Cloud

```python
from stinger_fx.strategies.indicators import ichimoku

ic = ichimoku(bars, tenkan_period=9, kijun_period=26, senkou_b_period=52)
if ic:
    cloud_top = max(ic.senkou_a, ic.senkou_b)
    cloud_bottom = min(ic.senkou_a, ic.senkou_b)
    if ic.chikou > cloud_top:
        # bullish above the cloud
```

### 5.6 Pivot points

```python
from stinger_fx.strategies.indicators import pivot_points

# Compute once per session from YESTERDAY's H/L/C
levels = pivot_points(
    prev_high=1.20, prev_low=1.10, prev_close=1.15,
    method="classic",     # or "fibonacci" or "camarilla"
)
# levels.pivot, .r1, .r2, .r3, .s1, .s2, .s3

await ctx.buy_limit(price=levels.s1, volume=0.1)
```

### 5.7 Cross-asset correlation

```python
from stinger_fx.strategies.indicators import correlation

eur = ctx.history_for("EURUSD", Timeframe.M15).closes()
gbp = ctx.history_for("GBPUSD", Timeframe.M15).closes()
rho = correlation(eur, gbp, period=50)
if rho is not None and rho > 0.85:
    # highly correlated — pair-tradeable
```

---

## 6. Position managers (composable)

Managers are small stateful classes that watch every tick and emit position management commands. Attach as many as you want — they fire in attachment order before the strategy's own hooks.

All implement at minimum `on_tick(ctx, tick)`; some also implement `on_bar`, `on_position_closed`, `on_order_filled`.

### 6.1 `TrailingStopManager`

Ratchets SL toward market on favourable moves.

```python
from stinger_fx.strategies.managers.trailing import TrailingStopManager

async def on_start(self, ctx):
    ctx.attach_manager(TrailingStopManager(
        ctx,
        distance_pips=15,          # SL trails 15 pips behind current price
        activate_after_pips=10,    # don't trail until profit > 10 pips
        symbol="EURUSD",           # optional — defaults to ctx.symbol
    ))
```

For BUY: SL = `bid - distance_pips`, only ever increases. For SELL: mirror.

### 6.2 `BreakEvenMover`

Once profit reaches `trigger_pips`, move SL to entry + `lock_pips`.

```python
from stinger_fx.strategies.managers.break_even import BreakEvenMover

ctx.attach_manager(BreakEvenMover(
    ctx,
    trigger_pips=20,    # profit must reach 20 pips first
    lock_pips=1,        # then lock in 1 pip past entry
))
```

Idempotent — fires once per ticket, then stays silent.

### 6.3 `LadderManager`

Pyramids into a position as price moves favourably.

```python
from stinger_fx.strategies.managers.ladder import LadderManager

ctx.attach_manager(LadderManager(
    ctx,
    step_pips=10,       # add a level every 10 pips of advance
    max_levels=3,       # max 3 additional entries per ticket
    level_volume=0.05,  # each rung is 0.05 lots
))
```

Each rung triggers a fresh market order on the same side. Idle when no position exists — doesn't open the initial position.

### 6.4 `TimeExitManager`

Closes positions that have been open too long.

```python
from stinger_fx.strategies.managers.time_exit import TimeExitManager

# Time-based: close after 4 hours
ctx.attach_manager(TimeExitManager(ctx, max_seconds=4 * 3600))

# Bar-based: close after 8 closed bars on the primary timeframe
ctx.attach_manager(TimeExitManager(ctx, max_bars=8))
```

Pick one or the other (mutually exclusive). Useful for strategies whose edge decays with time.

### 6.5 `OCOGroupManager`

One-cancels-other for position groups and pending-order brackets.

```python
from stinger_fx.strategies.managers.oco import OCOGroupManager

async def on_start(self, ctx):
    self._oco = OCOGroupManager(ctx)
    ctx.attach_manager(self._oco)

async def on_bar(self, ctx, bar):
    # Breakout bracket — BUY_STOP above, SELL_STOP below
    buy_result = await ctx.buy_stop(price=bar.high + 0.0010, volume=0.1)
    sell_result = await ctx.sell_stop(price=bar.low - 0.0010, volume=0.1)
    # Wire both into the same OCO group
    self._oco.add_bracket(
        buy_result.ticket, sell_result.ticket,
        group_id="breakout_1",
    )
```

Behaviour:
- When one **pending** in the group fills → other pendings get cancelled
- When one **position** in the group closes → other positions get closed
- Externally cancelled pendings just get removed from the group (no cascade)

The `_dissolving` set prevents re-entrant cascades (when the sibling's cancel event arrives back at the manager).

### 6.6 Writing your own manager

A manager is anything that implements `on_tick(ctx, tick)`:

```python
class HardStopAtNewYork:
    """Close all positions at 5pm NY time."""
    def __init__(self, ctx):
        self._ctx = ctx
        self._fired_today: set[int] = set()

    async def on_tick(self, ctx, tick):
        if tick.time.hour == 21 and tick.time.minute == 0:  # 5pm NY = 21:00 UTC
            for pos in ctx.position.all():
                if pos.ticket not in self._fired_today:
                    self._fired_today.add(pos.ticket)
                    await ctx.close(pos.ticket, reason="hard_stop_5pm_NY")
```

Optional hooks (the runner uses `hasattr` to dispatch):

```python
async def on_bar(self, ctx, bar): ...
async def on_order_filled(self, ctx, order): ...
async def on_position_closed(self, ctx, position): ...
async def on_order_cancelled(self, ctx, order): ...
```

---

## 7. Regime filters

Filters let you gate signal emission on market regime. Implement the `RegimeFilter` Protocol — `allows(bars) -> bool`.

```python
from stinger_fx.strategies.regime import (
    TrendingFilter, RangingFilter,
    HighVolatilityFilter, LowVolatilityFilter,
    CompositeFilter,
)

# Single-axis filters
self._trend = TrendingFilter(adx_period=14, threshold=25.0)
self._chop = RangingFilter(threshold=25.0)
self._high_vol = HighVolatilityFilter(atr_period=14, lookback=50, percentile=75)
self._low_vol = LowVolatilityFilter(percentile=25)

# Combined — logical AND
self._regime = CompositeFilter(
    TrendingFilter(threshold=20.0),
    LowVolatilityFilter(percentile=40),
)

async def on_bar(self, ctx, bar):
    bars = ctx.history.bars()
    if not self._regime.allows(bars):
        return  # wrong regime — sit it out
    # ... rest of strategy
```

All filters return `False` when there isn't enough data yet — strategies never trade on incomplete state.

---

## 8. Multi-feed strategies

Subscribe to multiple `(symbol, timeframe)` pairs:

```python
class MultiFeedParams(StrategyParams):
    primary_symbol: str = "EURUSD"
    secondary_symbol: str = "GBPUSD"
    timeframe: Timeframe = Timeframe.M15
    long_tf: Timeframe = Timeframe.H1

class MultiFeed(BaseStrategy):
    @classmethod
    def subscriptions(cls, params):
        return [
            Subscription(symbol=params.primary_symbol, timeframe=params.timeframe),    # primary
            Subscription(symbol=params.secondary_symbol, timeframe=params.timeframe),
            Subscription(symbol=params.primary_symbol, timeframe=params.long_tf),      # higher tf for trend confirmation
        ]

    async def on_bar(self, ctx, bar):
        # Bar arrives for ANY subscribed feed — route by (bar.symbol, bar.timeframe)
        if bar.symbol == ctx.params.primary_symbol and bar.timeframe == ctx.params.timeframe:
            await self._on_primary_bar(ctx, bar)
        elif bar.symbol == ctx.params.primary_symbol and bar.timeframe == ctx.params.long_tf:
            await self._on_higher_tf_bar(ctx, bar)
        elif bar.symbol == ctx.params.secondary_symbol:
            await self._on_secondary_bar(ctx, bar)

    async def _on_primary_bar(self, ctx, bar):
        # Read both feeds:
        primary_closes = ctx.history.closes()              # primary
        secondary_view = ctx.history_for(ctx.params.secondary_symbol, ctx.params.timeframe)
        secondary_closes = secondary_view.closes() if secondary_view else []
        # ... cross-feed logic
```

Key points:
- `ctx.history` → primary feed only
- `ctx.history_for(symbol, tf)` → any declared feed (returns `None` if not declared)
- `on_bar` fires for **any** declared feed — your code routes
- `on_tick` fires for **any** subscribed symbol (managers see them too)

---

## 9. Pending orders + OCO brackets

### 9.1 Breakout bracket

```python
class BreakoutBracket(BaseStrategy):
    name = "breakout_bracket"
    Params = BreakoutParams

    async def on_start(self, ctx):
        self._oco = OCOGroupManager(ctx)
        ctx.attach_manager(self._oco)
        self._has_active_bracket = False

    async def on_bar(self, ctx, bar):
        if self._has_active_bracket:
            return

        recent = ctx.history.bars(20)
        if len(recent) < 20:
            return

        # Find the recent range
        high = max(b.high for b in recent)
        low = min(b.low for b in recent)
        if (high - low) < 0.0030:    # too tight, skip
            return

        # Place brackets just outside the range
        buy_result = await ctx.buy_stop(
            price=high + 0.0005, volume=ctx.params.volume,
            sl=high - 0.0010, tp=high + 0.0050,
        )
        sell_result = await ctx.sell_stop(
            price=low - 0.0005, volume=ctx.params.volume,
            sl=low + 0.0010, tp=low - 0.0050,
        )

        if buy_result.ticket and sell_result.ticket:
            self._oco.add_bracket(
                buy_result.ticket, sell_result.ticket,
                group_id=f"bracket_{bar.time.isoformat()}",
            )
            self._has_active_bracket = True

    async def on_order_filled(self, ctx, order):
        # The triggered leg is now a position; the other was cancelled by OCO
        self._has_active_bracket = False

    async def on_position_closed(self, ctx, position):
        # Bracket fully resolved
        self._has_active_bracket = False
```

### 9.2 Dynamic trailing pending order

Trail a BUY_STOP higher as the range expands:

```python
async def on_bar(self, ctx, bar):
    if self._pending_ticket is None:
        # Place initial BUY_STOP
        result = await ctx.buy_stop(price=bar.high + 0.0010, volume=0.1)
        self._pending_ticket = result.ticket
        return

    # Trail the trigger higher
    recent_high = max(b.high for b in ctx.history.bars(10))
    new_trigger = recent_high + 0.0010
    await ctx.move_pending(self._pending_ticket, price=new_trigger)
```

---

## 10. Hot reload + parameter swap

Edit `config/strategies.yaml` while the engine is running. The reloader will:

1. Validate the new YAML against your `Params` schema
2. If valid: swap `ctx.params` atomically and call `on_params_reloaded`
3. If invalid: log to `data/logs/config.jsonl`, keep the old config running

```yaml
strategies:
  - id: my_strategy_001
    class_path: user_strategies.my_strategy:MyStrategy
    enabled: true
    params:
      fast: 10        # change to 12
      slow: 30
      volume: 0.05
```

Within ~500ms after saving:

```python
async def on_params_reloaded(self, ctx, old, new):
    # old.fast == 10, new.fast == 12
    # ctx.params already points at the new params
    if old.fast != new.fast:
        # invalidate cached indicators
        self._sma_cache.clear()
```

You don't need this hook — strategies that don't cache anything just see the new values on the next bar.

### Reload boundaries

- **Params:** hot-reloadable ✅
- **Adding a strategy:** hot-reloadable ✅ (starts automatically)
- **Removing a strategy:** hot-reloadable ✅ (drains gracefully)
- **`class_path`:** hot-reloadable ✅ (stops old + starts new with no state migration)
- **Switching broker:** requires restart ❌
- **Web host/port:** requires restart ❌

---

## 11. Pairs trading template

Pairs trading is a built-in template. Subscribes to two correlated symbols, computes a rolling hedge ratio, and trades the spread's z-score deviations.

```python
# user_strategies/my_pairs.py
from stinger_fx.strategies.cointegration import rolling_hedge_ratio, spread_zscore
from stinger_fx.strategies.examples.pairs_trading import PairsTradingParams, PairsTrading
```

See `src/stinger_fx/strategies/examples/pairs_trading.py` for the working implementation.

Validate cointegration before deploying:

```python
from stinger_fx.strategies.cointegration import engle_granger_test

result = engle_granger_test(eur_closes, gbp_closes)
print(f"Hedge ratio: {result.hedge_ratio:.4f}")
print(f"Stationary: {result.is_stationary}")
print(f"ADF p-value: {result.adf_pvalue}")    # None if statsmodels not installed
```

The heuristic falls back when `statsmodels` isn't installed; install `stinger-fx[pairs]` for the proper ADF p-value.

---

## 12. Testing strategies — file backtests

### 12.1 Basic backtest config

```yaml
# config/backtest.yaml
runs:
  - id: my_strategy_2024Q1
    mode: file
    strategy_id: my_strategy_001     # matches strategies.yaml id
    symbol: EURUSD
    timeframe: M15
    start: 2024-01-01T00:00:00Z
    end: 2024-04-01T00:00:00Z
    initial_balance: 10000.0
    data_source: ./data/parquet
    granularity: bar                 # or "tick" for tick-precise SL/TP
    slippage_pips: 0.5
    slippage_model: fixed            # or "spread" or "volatility"
```

Run:

```bash
uv run stinger-fx backtest run --run-id my_strategy_2024Q1
```

Results land in `data/backtests/<run-id>_*` and in SQLite. View the equity curve + trade markers at `http://localhost:8765/backtest/<run-id>` (web mode).

### 12.2 Tick-precise backtests

For strategies that depend on exact SL/TP timing or pending-order triggering, use `granularity: tick`:

```yaml
runs:
  - id: tick_test
    granularity: tick     # SL/TP fire on the exact tick that breaches them
    ...
```

50–500× slower than bar mode but unambiguous.

### 12.3 Multi-symbol backtests

```yaml
runs:
  - id: pairs_test
    symbols: [EURUSD, GBPUSD]
    timeframes: [M15]
    # or explicit:
    feeds:
      - {symbol: EURUSD, timeframe: M15}
      - {symbol: GBPUSD, timeframe: M15}
      - {symbol: EURUSD, timeframe: H1}
    ...
```

### 12.4 Programmatic backtests (for unit tests)

```python
from stinger_fx.backtest import FileBacktester
from stinger_fx.config.models import BacktestRunConfig, StrategyEntry
from stinger_fx.data import in_memory_store

entry = StrategyEntry(
    id="my_test",
    class_path="user_strategies.my_strategy:MyStrategy",
    enabled=True,
    params={"fast": 5, "slow": 20, "volume": 0.1},
)
cfg = BacktestRunConfig(
    id="ut_smoke", mode="file", strategy_id="my_test",
    symbol="EURUSD", timeframe=Timeframe.M15,
    start=..., end=...,
    initial_balance=10_000.0, data_source=parquet_root,
)
bt = FileBacktester(strategy=entry, parquet_root=parquet_root,
                    sqlite_store=in_memory_store(),
                    report_dir=tmp_path / "reports")
report = await bt.run(cfg)

assert report.net_pnl > 0
assert report.sharpe > 1.0
```

See `tests/integration/test_file_backtest.py` for the canonical pattern.

---

## 13. Optimisation — sweep, walk-forward, Pareto, custom metrics

### 13.1 Parameter sweep

Pick the algorithm that fits the space:

| Algo | When |
|---|---|
| `grid` | Small space (≤200 cells) — exhaustive |
| `random` | Large space, fast baseline |
| `optuna` | Smooth-ish objective, medium space — best convergence per trial |
| `genetic` | Large space, non-smooth objective — explores via population |

```yaml
sweeps:
  - id: ma_optuna_2024
    strategy_id: ma_crossover
    algo: optuna
    n_trials: 50
    random_seed: 42
    rank_by: net_pnl
    parameter_grid:
      fast: [5, 8, 10, 13, 15, 21]
      slow: [20, 30, 40, 50, 60, 80]
      volume: [0.01, 0.05, 0.1]
    symbol: EURUSD
    timeframe: M15
    start: 2024-01-01T00:00:00Z
    end: 2024-07-01T00:00:00Z
    data_source: ./data/parquet
```

```bash
uv run stinger-fx backtest sweep --sweep-id ma_optuna_2024
```

View at `/sweep/ma_optuna_2024` — 2-parameter sweeps show a heatmap, N-parameter sweeps show the ranked top-N.

### 13.2 Walk-forward validation

The honest test: fit on in-sample, evaluate on out-of-sample, repeat across many folds.

```yaml
walk_forwards:
  - id: ma_wf_2024
    strategy_id: ma_crossover
    n_folds: 6
    in_sample_pct: 0.7
    scheme: expanding              # or "rolling"
    algo: optuna
    n_trials: 30
    rank_by: net_pnl
    parameter_grid:
      fast: [5, 8, 10, 13]
      slow: [20, 30, 40, 50]
    symbol: EURUSD
    timeframe: M15
    start: 2024-01-01T00:00:00Z
    end: 2024-07-01T00:00:00Z
    data_source: ./data/parquet
```

View at `/walkforward/ma_wf_2024` — shows per-fold in-sample vs OOS metrics plus a Pearson **consistency score**:

- **+1.0** = strategy generalises (IS predicts OOS)
- **0.0** = no relationship
- **-1.0** = severe overfit (high IS → low OOS)

### 13.3 Multi-objective Pareto

```yaml
sweeps:
  - id: ma_pareto
    rank_by: net_pnl
    objectives:
      - {metric: net_pnl, direction: max}
      - {metric: max_drawdown, direction: min}
      - {metric: sharpe, direction: max}
    parameter_grid: {...}
```

View at `/sweep/ma_pareto/pareto` — scatter plot with Pareto-optimal cells highlighted in green.

### 13.4 Custom metric DSL

Compose new metrics from built-ins:

```yaml
sweeps:
  - id: ma_custom
    rank_by: risk_adjusted          # ← custom metric name
    custom_metrics:
      risk_adjusted: "sharpe - 0.5 * max_drawdown / 10"
      pnl_per_dd: "net_pnl / (max_drawdown + 1)"
      regime_aware: "sharpe if max_drawdown < 15 else sharpe * 0.5"
    objectives:
      - {metric: risk_adjusted, direction: max}
      - {metric: max_drawdown, direction: min}
    parameter_grid: {...}
```

Available built-in metrics: `net_pnl`, `gross_profit`, `gross_loss`, `win_rate`, `profit_factor`, `expectancy`, `max_drawdown`, `sharpe`, `trades`, `initial_balance`, `final_balance`.

Available safe functions in the DSL: `abs`, `min`, `max`, `round`, `sqrt`, `log`, `exp`, `floor`, `ceil`.

Operators: `+ - * / // % ** == != < <= > >= and or not`, ternary `a if cond else b`.

### 13.5 Monte Carlo confidence bands

For any backtest run, run a bootstrap simulation:

```bash
curl 'http://localhost:8765/backtest/<run-id>/monte_carlo.json?n=1000&seed=42'
```

Returns percentile bands for `net_pnl`, `max_drawdown`, `sharpe` plus an equity-curve envelope (5th/50th/95th percentile at each step). Wide envelope = lucky run; narrow = robust strategy.

### 13.6 Portfolio aggregation

After running several backtests, combine them:

- Web UI: `/portfolio` → select 2+ runs → `/portfolio/view?runs=a,b,c`
- JSON: `GET /portfolio/data.json?runs=a,b,c`

You get the combined equity curve, per-strategy contribution, and a cross-strategy correlation matrix.

---

## 14. Risk management

The engine has a global risk monitor that applies to **every** order. Configure in `config/app.yaml`:

```yaml
risk:
  max_open_positions_per_strategy: 5     # cap per strategy_id
  max_daily_loss_pct: 5.0                # halt new orders when day's loss exceeds X% of opening balance
  kill_switch_drawdown_pct: 20.0         # halt ALL new orders when peak-to-current drawdown exceeds X%
  per_symbol:                            # optional per-symbol limits
    EURUSD:
      max_open_positions: 2
      max_daily_loss_usd: 200.0
    GBPUSD:
      max_open_positions: 1
```

Blocks happen at the order router — your strategy's `await ctx.buy(...)` returns successfully but the order never reaches the broker. A `DecisionEvent` with `action="rejected"` is published so you see why in `/audit`.

The kill switch is sticky — once tripped, only `RiskMonitor.reset_kill_switch()` clears it.

---

## 15. Logging + observability

### 15.1 Structured logs in strategies

`ctx.logger` is a `structlog.BoundLogger` with `strategy_id` and `symbol` pre-bound:

```python
ctx.logger.info("entry_signal", reason="ma_cross_up", fast=10, slow=30)
ctx.logger.warning("regime_unstable", adx=result.adx)
```

Output lands in `data/logs/strategy_<id>.jsonl` and is also mirrored to SQLite (`signals`, `decisions` tables).

### 15.2 Prometheus metrics

Set `metrics.enabled: true` in `config/app.yaml` to expose `/metrics` (default port 9100). Key strategy-relevant metrics:

- `stinger_signals_total{strategy_id, side}` — total signals emitted
- `stinger_orders_filled_total{strategy_id, symbol, side}`
- `stinger_orders_rejected_total{strategy_id, symbol}`
- `stinger_signals_rejected_by_risk_total{strategy_id, rule}`
- `stinger_order_submission_seconds{strategy_id, symbol}` — round-trip latency histogram
- `stinger_tick_pump_lag_seconds{symbol}` — current tick staleness (alert when > 30s)

### 15.3 Notifications

Telegram / Discord webhooks fire on configured events. Add to `config/app.yaml`:

```yaml
notifications:
  - kind: telegram
    enabled: true
    bot_token: "..."
    chat_id: "..."
    events: [order_filled, order_rejected, kill_switch_tripped]
```

See `src/stinger_fx/observability/notifications.py` for the full event list.

---

## 16. Common pitfalls

### 16.1 Reading `bars()` without checking length

```python
# BAD — crashes when bars() is empty
fast = sma(ctx.history.closes(), ctx.params.fast)

# GOOD
closes = ctx.history.closes()
if len(closes) < ctx.params.slow + 2:
    return
fast = sma(closes, ctx.params.fast)
```

Indicators return `None` when there's not enough data — handle that too.

### 16.2 Forgetting `is_closed` semantics

`ctx.history.bars()` returns **closed** bars only. The currently-forming bar isn't there. If you need real-time price, use `ctx.history.last_tick()`.

### 16.3 Trying to trade other symbols via `ctx.buy()`

`ctx.buy()` is hard-coded to `ctx.symbol`. For pairs / cross-asset trading, use `ctx.submit_signal(Signal(symbol=other_symbol, ...))`.

### 16.4 Not handling `OrderResult.ok = False`

```python
# BAD — assumes the order went through
await ctx.buy(0.1)
self._has_position = True

# GOOD — react via on_order_filled / on_order_rejected
async def on_order_filled(self, ctx, order):
    self._has_position = True

async def on_order_rejected(self, ctx, order, reason):
    ctx.logger.warning("entry_rejected", reason=reason)
```

The async return value of `ctx.buy()` is fire-and-forget — the actual fill is asynchronous.

### 16.5 Modifying state from `on_tick` in tick-mode backtests

Tick mode fires `on_tick` for thousands of ticks per minute. If you write to disk, log every tick, or do anything O(n²) over bars, the backtest crawls.

```python
# BAD — recomputes ATR on every tick
async def on_tick(self, ctx, tick):
    atr_val = atr(ctx.history.bars(100), 14)   # O(100) per tick

# GOOD — compute on bar close, cache for tick reactions
async def on_bar(self, ctx, bar):
    self._cached_atr = atr(ctx.history.bars(100), 14)

async def on_tick(self, ctx, tick):
    if self._cached_atr is None:
        return
    # use self._cached_atr
```

### 16.6 Mutating Pydantic instances

`Signal`, `Order`, `Position` are frozen. Never try to mutate `position.sl = 1.10` — use `await ctx.move_stop(position.ticket, sl=1.10)` instead.

### 16.7 Mixing live and historical timestamps

`tick.time` is the broker's stamp, in UTC. `ctx.clock.now()` is sim-time in backtests, wall-clock in live. Comparing the two only makes sense when both are sim or both are live.

---

## 17. Best practices

### 17.1 Keep `on_bar` pure-functional where possible

```python
async def on_bar(self, ctx, bar):
    # 1. Compute signals (pure function of history)
    signal = self._compute_signal(ctx.history)

    # 2. Apply gates (regime, risk, time-of-day)
    if not self._regime.allows(ctx.history.bars()):
        return
    if not self._in_trading_hours(bar.time):
        return

    # 3. Submit orders (side effect)
    if signal == "buy" and not ctx.position.for_symbol(ctx.symbol):
        await ctx.buy(ctx.params.volume)
```

Stateless decision logic = trivially testable + trivially sweepable.

### 17.2 Parameter everything

```python
# BAD — magic numbers buried in code
class Params(StrategyParams):
    symbol: str = "EURUSD"

class Strat(BaseStrategy):
    async def on_bar(self, ctx, bar):
        fast = sma(closes, 10)              # ← 10
        slow = sma(closes, 30)              # ← 30
        if fast > slow * 1.001:             # ← 1.001
            await ctx.buy(0.05)             # ← 0.05

# GOOD — every magic number is a parameter
class Params(StrategyParams):
    symbol: str = "EURUSD"
    fast: int = 10
    slow: int = 30
    cross_threshold: float = 1.001
    volume: float = 0.05
```

Now you can sweep over `cross_threshold` to find the optimal value. Hidden constants are not optimisable.

### 17.3 Validate at the params layer

```python
from pydantic import Field, field_validator

class Params(StrategyParams):
    fast: int = Field(10, ge=2, le=200)
    slow: int = Field(30, ge=5, le=500)

    @field_validator("slow")
    @classmethod
    def slow_greater_than_fast(cls, v, info):
        if "fast" in info.data and v <= info.data["fast"]:
            raise ValueError("slow must be > fast")
        return v
```

Failed validation logs to `config_audit` and keeps the old config running — much better than crashing at runtime when `fast = slow`.

### 17.4 Use managers instead of in-line SL/TP tracking

```python
# BAD — manual trailing SL in on_tick
async def on_tick(self, ctx, tick):
    for pos in ctx.position.all():
        if pos.side == Side.BUY:
            new_sl = tick.bid - 0.0015
            if new_sl > pos.sl:
                await ctx.move_stop(pos.ticket, sl=new_sl)

# GOOD — attach the manager once
async def on_start(self, ctx):
    ctx.attach_manager(TrailingStopManager(ctx, distance_pips=15))
```

Managers handle re-entry, ticket lifecycle, position-state caching, and the multi-position case for you.

### 17.5 Backtest before live

Every change — parameters, indicator periods, regime filters — should be validated on at least one walk-forward backtest before going live. The platform's whole point is making that workflow trivial.

### 17.6 Idempotent state machines

For strategies with cooldowns, breakouts, or one-shot triggers:

```python
class Params(StrategyParams):
    cooldown_seconds: int = 3600

class Strat(BaseStrategy):
    async def on_start(self, ctx):
        self._next_entry_time = ctx.clock.now()

    async def on_bar(self, ctx, bar):
        if ctx.clock.now() < self._next_entry_time:
            return  # in cooldown
        if self._signal_fires():
            await ctx.buy(...)
            self._next_entry_time = ctx.clock.now() + timedelta(
                seconds=ctx.params.cooldown_seconds
            )
```

Compare time, not bar count — bar count is fragile if you skip bars during regime filtering.

### 17.7 Test the failure modes

```python
# In tests:
# - What happens if the broker rejects?
# - What if the position is closed externally (manual MT5 close, server)?
# - What if a partial fill leaves us with less volume than expected?
# - What if hot reload changes params mid-trade?
```

The platform handles most of these automatically (magic-tagged position tracking, on_position_closed, on_order_rejected), but write tests that cover your strategy's specific assumptions.

---

## 18. Worked examples

### 18.1 MA crossover with regime + trailing stop

```python
from pydantic import Field
from stinger_fx.domain import Bar, Subscription, Timeframe
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.indicators import sma, atr
from stinger_fx.strategies.managers.trailing import TrailingStopManager
from stinger_fx.strategies.parameters import StrategyParams
from stinger_fx.strategies.regime import TrendingFilter


class TrendMAParams(StrategyParams):
    symbol: str = "EURUSD"
    timeframe: Timeframe = Timeframe.M15
    fast: int = Field(10, ge=2, le=200)
    slow: int = Field(30, ge=5, le=500)
    atr_period: int = Field(14, ge=2)
    sl_atr_mult: float = Field(2.0, gt=0)
    trail_pips: float = Field(15.0, gt=0)
    adx_threshold: float = Field(25.0, gt=0)
    volume: float = Field(0.1, gt=0)


class TrendMA(BaseStrategy):
    name = "trend_ma"
    Params = TrendMAParams

    @classmethod
    def subscriptions(cls, params):
        return [Subscription(symbol=params.symbol, timeframe=params.timeframe)]

    async def on_start(self, ctx):
        self._regime = TrendingFilter(threshold=ctx.params.adx_threshold)
        ctx.attach_manager(TrailingStopManager(
            ctx, distance_pips=ctx.params.trail_pips
        ))

    async def on_bar(self, ctx, bar):
        bars = ctx.history.bars()
        if len(bars) < max(ctx.params.slow + 2, ctx.params.atr_period + 1):
            return

        # Regime gate
        if not self._regime.allows(bars):
            return

        closes = ctx.history.closes()
        fast_now = sma(closes, ctx.params.fast)
        slow_now = sma(closes, ctx.params.slow)
        fast_prev = sma(closes[:-1], ctx.params.fast)
        slow_prev = sma(closes[:-1], ctx.params.slow)
        if None in (fast_now, slow_now, fast_prev, slow_prev):
            return

        atr_val = atr(bars, ctx.params.atr_period)
        if atr_val is None:
            return

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now

        if crossed_up and not ctx.position.for_symbol(ctx.symbol):
            sl = bar.low - ctx.params.sl_atr_mult * atr_val
            await ctx.buy(ctx.params.volume, sl=sl, comment="ma_xup")
        elif crossed_down and not ctx.position.for_symbol(ctx.symbol):
            sl = bar.high + ctx.params.sl_atr_mult * atr_val
            await ctx.sell(ctx.params.volume, sl=sl, comment="ma_xdown")
```

### 18.2 Breakout bracket with OCO

See [Section 9.1](#91-breakout-bracket).

### 18.3 Pairs trading template

See `src/stinger_fx/strategies/examples/pairs_trading.py`.

### 18.4 Multi-timeframe confirmation

```python
class Params(StrategyParams):
    symbol: str = "EURUSD"
    entry_tf: Timeframe = Timeframe.M15
    confirm_tf: Timeframe = Timeframe.H1
    fast: int = 10
    slow: int = 30
    volume: float = 0.1


class MultiTFStrategy(BaseStrategy):
    name = "multi_tf"
    Params = Params

    @classmethod
    def subscriptions(cls, params):
        return [
            Subscription(symbol=params.symbol, timeframe=params.entry_tf),
            Subscription(symbol=params.symbol, timeframe=params.confirm_tf),
        ]

    async def on_bar(self, ctx, bar):
        # Route by timeframe
        if bar.timeframe != ctx.params.entry_tf:
            return  # we only ACT on the entry timeframe

        entry_view = ctx.history    # primary = entry_tf
        confirm_view = ctx.history_for(ctx.params.symbol, ctx.params.confirm_tf)
        if confirm_view is None:
            return

        # Entry signal on M15
        entry_fast = sma(entry_view.closes(), ctx.params.fast)
        entry_slow = sma(entry_view.closes(), ctx.params.slow)
        if entry_fast is None or entry_slow is None or entry_fast <= entry_slow:
            return

        # Confirmation on H1 — only trade if H1 trend is also up
        confirm_fast = sma(confirm_view.closes(), 10)
        confirm_slow = sma(confirm_view.closes(), 30)
        if confirm_fast is None or confirm_slow is None or confirm_fast <= confirm_slow:
            return

        await ctx.buy(ctx.params.volume, comment="multi_tf_confirmed")
```

---

## Where to next

- **Run a backtest** on the strategy you just wrote: `uv run stinger-fx backtest run --run-id ...`
- **Sweep its parameters**: see [Section 13.1](#131-parameter-sweep)
- **Walk-forward validate**: see [Section 13.2](#132-walk-forward-validation)
- **Browse the working examples** in `src/stinger_fx/strategies/examples/`:
  - `ma_crossover.py` — minimal trend-following
  - `pairs_trading.py` — statistical arbitrage
  - `regime_filtered_ma.py` — regime-aware variant
- **Read the README** for installation + ops-level docs

Questions? Open an issue on the repo or check `data/logs/` — every event the engine processes is logged for inspection.
