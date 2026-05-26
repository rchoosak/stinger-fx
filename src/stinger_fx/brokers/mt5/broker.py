"""MT5Broker — concrete BaseBroker over the official MetaTrader5 Python package.

The MetaTrader5 SDK is synchronous and not thread-safe, so:

  • One dedicated single-thread executor handles every SDK call. That gives
    us serialization for free (only one call at a time) and lets us treat
    the SDK as if it lived behind an async facade.
  • Tick subscriptions run in a separate daemon thread that polls
    `symbol_info_tick` and forwards changes to the asyncio loop via
    `call_soon_threadsafe(bus.publish, ...)`.

The SDK import is lazy so unit tests on macOS/Linux can still import this
module to inspect the class shape.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa

from stinger_fx.brokers.base import BaseBroker
from stinger_fx.config.models import MT5Config
from stinger_fx.core.errors import BrokerError, BrokerNotConnectedError
from stinger_fx.core.event_bus import AsyncEventBus
from stinger_fx.core.events import (
    BrokerDisconnectedEvent,
    BrokerReconnectedEvent,
    TickEvent,
)
from stinger_fx.domain import (
    AccountInfo,
    AccountSnapshot,
    Order,
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    Position,
    Side,
    SymbolInfo,
    Tick,
    Timeframe,
)

logger = logging.getLogger("stinger.broker.mt5")

# Tick poller interval — MT5 doesn't push, we pull.
TICK_POLL_INTERVAL = 0.05  # 50ms — broker-friendly while still responsive

# How often the reconnect loop probes terminal_info() while connected.
HEALTH_CHECK_INTERVAL = 5.0  # seconds

# Reconnect backoff schedule (seconds). Capped at the last value for further attempts.
RECONNECT_BACKOFF = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0]

# How many times to re-attempt order_send on retryable broker errors.
ORDER_MAX_RETRIES = 3
ORDER_RETRY_BACKOFF = [0.1, 0.5, 1.0]  # seconds between attempts

# MT5 retcodes treated as transient (re-attempt makes sense). Reference:
# https://www.mql5.com/en/docs/constants/structures/mqltraderequest
# We intentionally keep this list short and conservative — permanent errors
# (invalid volume, no money, market closed) must NOT be retried.
RETRYABLE_RETCODES: frozenset[int] = frozenset({
    10004,  # TRADE_RETCODE_REQUOTE — price moved during request
    10006,  # TRADE_RETCODE_REJECT — generic broker reject (transient on busy servers)
    10021,  # TRADE_RETCODE_PRICE_OFF — no prices to process the request
    10024,  # TRADE_RETCODE_TIMEOUT — server didn't respond in time
})


class MT5Broker(BaseBroker):
    name = "mt5"

    def __init__(self, bus: AsyncEventBus, cfg: MT5Config) -> None:
        super().__init__(bus)
        self._cfg = cfg
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt5-sync")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connected = False
        self._tick_subs: set[str] = set()
        self._bar_subs: set[tuple[str, Timeframe]] = set()
        self._tick_thread: threading.Thread | None = None
        self._tick_stop = threading.Event()
        self._last_tick_time: dict[str, datetime] = {}
        # Phase 6.1.A reconnection state ----------------------------------
        # Health-check task runs in the asyncio loop; tick pump (in a thread)
        # observes self._connected and pauses iteration while False.
        self._health_task: asyncio.Task[None] | None = None
        self._reconnecting = False
        self._disconnect_time: datetime | None = None
        # Allow tests to override pacing constants for fast simulations.
        self._reconnect_backoff: list[float] = list(RECONNECT_BACKOFF)
        self._health_check_interval: float = HEALTH_CHECK_INTERVAL
        self._order_retry_backoff: list[float] = list(ORDER_RETRY_BACKOFF)

    # --- helpers ------------------------------------------------------------

    async def _sdk(self, fn, *args, **kwargs):
        """Run a synchronous SDK call on the dedicated executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs))

    @staticmethod
    def _mt5():
        try:
            import MetaTrader5 as mt5

            return mt5
        except ImportError as e:
            raise BrokerError(
                "MetaTrader5 SDK is unavailable — install with `uv sync --extra mt5` on Windows"
            ) from e

    # --- Lifecycle ----------------------------------------------------------

    async def connect(self) -> None:
        self._loop = asyncio.get_running_loop()
        mt5 = self._mt5()

        def _do_init() -> bool:
            kwargs: dict[str, Any] = {"timeout": self._cfg.timeout_ms}
            if self._cfg.terminal_path:
                kwargs["path"] = self._cfg.terminal_path
            if self._cfg.login:
                kwargs.update(
                    login=self._cfg.login,
                    password=self._cfg.password,
                    server=self._cfg.server,
                )
            return bool(mt5.initialize(**kwargs))

        ok = await self._sdk(_do_init)
        if not ok:
            err = await self._sdk(mt5.last_error)
            raise BrokerError(f"MT5 initialize() failed: {err}")
        self._connected = True
        logger.info("mt5 connected")
        # Start the health-check task that detects unannounced disconnects
        # and drives the reconnect loop. Test fixtures that don't want the
        # background task running can call disconnect() right away or set
        # `_health_check_interval` to a large value before connect().
        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(
                self._health_loop(), name="mt5-health"
            )

    async def disconnect(self) -> None:
        if not self._connected and self._health_task is None:
            return
        # Stop the health task first so it doesn't try to reconnect during shutdown.
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._health_task = None
        self._tick_stop.set()
        if self._tick_thread is not None:
            self._tick_thread.join(timeout=2)
            self._tick_thread = None
        if self._connected:
            await self._sdk(self._mt5().shutdown)
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._connected = False
        logger.info("mt5 disconnected")

    async def is_connected(self) -> bool:
        return self._connected

    def _require_connected(self) -> None:
        if not self._connected:
            raise BrokerNotConnectedError("broker is not connected")

    # --- Account & symbols --------------------------------------------------

    async def get_account_info(self) -> AccountInfo:
        self._require_connected()
        mt5 = self._mt5()
        info = await self._sdk(mt5.account_info)
        if info is None:
            err = await self._sdk(mt5.last_error)
            raise BrokerError(f"account_info() failed: {err}")
        return AccountInfo(
            account_id=str(info.login),
            broker=info.company or "mt5",
            server=info.server,
            currency=info.currency,
            leverage=int(info.leverage) or 1,
            name=info.name or "",
        )

    async def get_account_snapshot(self) -> AccountSnapshot:
        self._require_connected()
        mt5 = self._mt5()
        info = await self._sdk(mt5.account_info)
        if info is None:
            err = await self._sdk(mt5.last_error)
            raise BrokerError(f"account_info() failed: {err}")
        margin = float(info.margin)
        equity = float(info.equity)
        return AccountSnapshot(
            account_id=str(info.login),
            time=datetime.now(UTC),
            balance=float(info.balance),
            equity=equity,
            margin=margin,
            free_margin=float(info.margin_free),
            margin_level=(equity / margin * 100.0) if margin > 0 else 0.0,
            profit=float(info.profit),
        )

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        self._require_connected()
        mt5 = self._mt5()
        # Make sure the symbol is visible in MarketWatch
        await self._sdk(mt5.symbol_select, symbol, True)
        info = await self._sdk(mt5.symbol_info, symbol)
        if info is None:
            err = await self._sdk(mt5.last_error)
            raise BrokerError(f"symbol_info({symbol}) failed: {err}")
        return SymbolInfo(
            symbol=info.name,
            digits=int(info.digits),
            point=float(info.point),
            contract_size=float(info.trade_contract_size),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
            spread=int(info.spread),
            currency_base=info.currency_base,
            currency_profit=info.currency_profit,
            currency_margin=info.currency_margin,
            trade_allowed=bool(info.trade_mode != 0),
        )

    async def list_symbols(self) -> list[str]:
        self._require_connected()
        symbols = await self._sdk(self._mt5().symbols_get)
        return [s.name for s in symbols or []]

    # --- Subscriptions ------------------------------------------------------

    async def subscribe_ticks(self, symbol: str) -> None:
        self._require_connected()
        await self._sdk(self._mt5().symbol_select, symbol, True)
        self._tick_subs.add(symbol)
        if self._tick_thread is None:
            self._tick_stop.clear()
            self._tick_thread = threading.Thread(
                target=self._tick_pump, name="mt5-tick-pump", daemon=True
            )
            self._tick_thread.start()
        logger.info("mt5 subscribed to ticks symbol=%s", symbol)

    async def subscribe_bars(self, symbol: str, tf: Timeframe) -> None:
        self._require_connected()
        await self._sdk(self._mt5().symbol_select, symbol, True)
        self._bar_subs.add((symbol, tf))
        # Bar events are derived from ticks by the BarAggregator in core/.
        # We still need a tick subscription to drive aggregation.
        await self.subscribe_ticks(symbol)
        logger.info("mt5 subscribed to bars symbol=%s tf=%s", symbol, tf.value)

    async def unsubscribe(self, symbol: str, tf: Timeframe | None = None) -> None:
        if tf is None:
            self._tick_subs.discard(symbol)
            self._bar_subs = {(s, t) for (s, t) in self._bar_subs if s != symbol}
        else:
            self._bar_subs.discard((symbol, tf))

    # --- Health-check + reconnect (Phase 6.1.A) -----------------------------

    async def _health_loop(self) -> None:
        """Periodic probe — detect MT5 disconnects and drive reconnect.

        Runs in the asyncio loop. Every `_health_check_interval` seconds it
        checks `terminal_info()`; if that returns None (or raises) while we
        believe we are connected, we mark the broker disconnected, publish
        a `BrokerDisconnectedEvent`, and enter the reconnect loop with
        exponential backoff. While reconnecting the tick pump pauses
        because it observes `self._connected == False`.
        """
        mt5 = self._mt5()
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                if not self._connected:
                    # Already in reconnect mode (or never connected) — nothing
                    # to detect; the reconnect loop is driving us.
                    continue
                try:
                    info = await self._sdk(mt5.terminal_info)
                except Exception as e:  # noqa: BLE001
                    info = None
                    err_reason = str(e)
                else:
                    err_reason = "terminal_info() returned None"
                if info is None:
                    await self._handle_disconnect(reason=err_reason)
            except asyncio.CancelledError:
                raise

    async def _handle_disconnect(self, *, reason: str) -> None:
        """Switch broker to disconnected state and launch reconnect attempts."""
        if self._reconnecting:
            return
        self._reconnecting = True
        self._connected = False
        self._disconnect_time = datetime.now(UTC)
        logger.warning("mt5 disconnect detected reason=%s", reason)
        await self.bus.publish(
            BrokerDisconnectedEvent(broker_name=self.name, reason=reason)
        )
        try:
            await self._reconnect_with_backoff()
        finally:
            self._reconnecting = False

    async def _reconnect_with_backoff(self) -> None:
        """Re-initialise MT5 with exponential backoff until success."""
        mt5 = self._mt5()
        attempt = 0
        while True:
            delay = self._reconnect_backoff[min(attempt, len(self._reconnect_backoff) - 1)]
            attempt += 1
            await asyncio.sleep(delay)
            logger.info("mt5 reconnect attempt=%s after_sleep=%.1fs", attempt, delay)
            try:
                # shutdown() before re-initialize so the SDK's internal state
                # is clean. Best-effort — may itself error if SDK is wedged.
                try:
                    await self._sdk(mt5.shutdown)
                except Exception:  # noqa: BLE001
                    pass

                def _do_init() -> bool:
                    kwargs: dict[str, Any] = {"timeout": self._cfg.timeout_ms}
                    if self._cfg.terminal_path:
                        kwargs["path"] = self._cfg.terminal_path
                    if self._cfg.login:
                        kwargs.update(
                            login=self._cfg.login,
                            password=self._cfg.password,
                            server=self._cfg.server,
                        )
                    return bool(mt5.initialize(**kwargs))

                ok = await self._sdk(_do_init)
            except Exception as e:  # noqa: BLE001
                logger.warning("mt5 reconnect attempt failed attempt=%s err=%s", attempt, e)
                continue
            if not ok:
                continue
            # Reconnected.
            downtime = (
                (datetime.now(UTC) - self._disconnect_time).total_seconds()
                if self._disconnect_time
                else 0.0
            )
            self._connected = True
            self._disconnect_time = None
            logger.info(
                "mt5 reconnected attempts=%s downtime_seconds=%.1f", attempt, downtime
            )
            await self.bus.publish(
                BrokerReconnectedEvent(broker_name=self.name, downtime_seconds=downtime)
            )
            # Re-arm any symbol subscriptions so the tick pump keeps working
            # after a reconnect — symbol_select is needed because MT5 forgets
            # MarketWatch state across SDK shutdown/initialize cycles.
            for sym in tuple(self._tick_subs):
                try:
                    await self._sdk(mt5.symbol_select, sym, True)
                except Exception:  # noqa: BLE001
                    logger.warning("mt5 resubscribe failed symbol=%s", sym)
            return

    def _tick_pump(self) -> None:
        """Background thread: poll the latest tick for each subscribed symbol."""
        mt5 = self._mt5()
        loop = self._loop
        if loop is None:
            return
        while not self._tick_stop.is_set():
            # Pause during reconnect — the health loop drives recovery; we
            # just wait. This avoids spinning calls into a dead SDK and
            # avoids stomping the reconnect attempt.
            if not self._connected:
                self._tick_stop.wait(self._health_check_interval)
                continue
            for symbol in tuple(self._tick_subs):
                try:
                    raw = mt5.symbol_info_tick(symbol)
                except Exception:
                    logger.exception("symbol_info_tick failed symbol=%s", symbol)
                    continue
                if raw is None:
                    continue
                # MT5 returns time as epoch seconds in broker server tz; convert to UTC.
                tick_time = datetime.fromtimestamp(raw.time, tz=UTC)
                if self._last_tick_time.get(symbol) == tick_time:
                    continue
                self._last_tick_time[symbol] = tick_time
                tick = Tick(
                    symbol=symbol,
                    time=tick_time,
                    bid=float(raw.bid),
                    ask=float(raw.ask),
                    last=float(raw.last),
                    volume=int(raw.volume),
                    flags=int(raw.flags),
                )
                asyncio.run_coroutine_threadsafe(self.bus.publish(TickEvent(tick=tick)), loop)
            self._tick_stop.wait(TICK_POLL_INTERVAL)

    # --- Historical ---------------------------------------------------------

    async def get_history_bars(
        self,
        symbol: str,
        tf: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pa.Table:
        from stinger_fx.brokers.mt5.mapping import to_mt5
        from stinger_fx.data.parquet_store import BAR_SCHEMA

        self._require_connected()
        mt5_tf = to_mt5(tf)
        rates = await self._sdk(self._mt5().copy_rates_range, symbol, mt5_tf, start, end)
        if rates is None or len(rates) == 0:
            return BAR_SCHEMA.empty_table()
        rows = []
        for r in rates:
            rows.append(
                {
                    "time": datetime.fromtimestamp(int(r["time"]), tz=UTC),
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "tick_volume": int(r["tick_volume"]),
                    "real_volume": int(r["real_volume"]),
                    "spread": int(r["spread"]),
                }
            )
        return pa.Table.from_pylist(rows, schema=BAR_SCHEMA)

    async def get_history_ticks(
        self, symbol: str, start: datetime, end: datetime
    ) -> pa.Table:
        from stinger_fx.data.parquet_store import TICK_SCHEMA

        self._require_connected()
        mt5 = self._mt5()
        ticks = await self._sdk(mt5.copy_ticks_range, symbol, start, end, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            return TICK_SCHEMA.empty_table()
        rows = []
        for t in ticks:
            rows.append(
                {
                    "time_ns": datetime.fromtimestamp(int(t["time"]), tz=UTC),
                    "bid": float(t["bid"]),
                    "ask": float(t["ask"]),
                    "last": float(t["last"]),
                    "volume": int(t["volume"]),
                    "flags": int(t["flags"]),
                }
            )
        return pa.Table.from_pylist(rows, schema=TICK_SCHEMA)

    # --- Orders -------------------------------------------------------------

    async def place_order(self, req: OrderRequest) -> OrderResult:
        self._require_connected()
        mt5 = self._mt5()
        action = mt5.TRADE_ACTION_DEAL if req.type == OrderType.MARKET else mt5.TRADE_ACTION_PENDING
        order_type = self._order_type_constant(mt5, req)

        # For market orders, fill the price from the current tick
        price = req.price
        if price is None:
            tick = await self._sdk(mt5.symbol_info_tick, req.symbol)
            if tick is None:
                return OrderResult(
                    ok=False, status=OrderStatus.REJECTED,
                    message="no current tick — cannot price market order",
                )
            price = tick.ask if req.side is Side.BUY else tick.bid

        request_dict: dict[str, Any] = {
            "action": action,
            "symbol": req.symbol,
            "volume": req.volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": req.magic,
            "comment": req.comment,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        if req.sl is not None:
            request_dict["sl"] = req.sl
        if req.tp is not None:
            request_dict["tp"] = req.tp

        # --- Retry loop (Phase 6.1.A) ---------------------------------------
        # Retryable retcodes (REQUOTE, REJECT, PRICE_OFF, TIMEOUT) get up to
        # ORDER_MAX_RETRIES attempts with exponential backoff. Permanent
        # errors (INVALID_VOLUME, NO_MONEY, MARKET_CLOSED) exit immediately.
        result = None
        last_attempt = 0
        for attempt in range(ORDER_MAX_RETRIES):
            last_attempt = attempt
            # On a retry for a MARKET order, re-fetch the live tick so we're
            # not chasing stale prices. Pending orders use the requested
            # price as-is — that's the whole point of a limit/stop.
            if attempt > 0 and req.type == OrderType.MARKET:
                tick = await self._sdk(mt5.symbol_info_tick, req.symbol)
                if tick is not None:
                    request_dict["price"] = (
                        tick.ask if req.side is Side.BUY else tick.bid
                    )
            result = await self._sdk(mt5.order_send, request_dict)
            if result is None:
                # SDK call failure (terminal dead?) — don't retry inside the
                # retry loop; surface as a single rejection. The health loop
                # will detect the disconnect separately.
                break
            retcode = int(result.retcode)
            if retcode == mt5.TRADE_RETCODE_DONE:
                break
            if retcode not in RETRYABLE_RETCODES:
                break
            if attempt + 1 < ORDER_MAX_RETRIES:
                backoff = self._order_retry_backoff[
                    min(attempt, len(self._order_retry_backoff) - 1)
                ]
                logger.info(
                    "order_send retryable retcode=%s attempt=%s sleeping=%.2fs",
                    retcode, attempt + 1, backoff,
                )
                await asyncio.sleep(backoff)

        if result is None:
            err = await self._sdk(mt5.last_error)
            return OrderResult(
                ok=False, status=OrderStatus.REJECTED,
                message=f"order_send() returned None after {last_attempt + 1} attempt(s): {err}",
            )
        ok = int(result.retcode) == mt5.TRADE_RETCODE_DONE
        status = OrderStatus.FILLED if ok else OrderStatus.REJECTED
        order = None
        if ok:
            order = Order(
                ticket=int(result.order),
                strategy_id=req.strategy_id,
                symbol=req.symbol,
                side=req.side,
                type=req.type,
                volume=req.volume,
                filled_volume=float(result.volume or req.volume),
                price=price,
                fill_price=float(result.price or price),
                sl=req.sl,
                tp=req.tp,
                status=status,
                comment=req.comment,
                magic=req.magic,
                client_order_id=req.client_order_id,
                requested_at=datetime.now(UTC),
                filled_at=datetime.now(UTC) if ok else None,
            )
        return OrderResult(
            ok=ok,
            ticket=int(result.order) if ok else None,
            status=status,
            message=str(result.comment or ""),
            raw_code=int(result.retcode),
            order=order,
        )

    @staticmethod
    def _order_type_constant(mt5, req: OrderRequest) -> int:
        if req.type == OrderType.MARKET:
            return mt5.ORDER_TYPE_BUY if req.side is Side.BUY else mt5.ORDER_TYPE_SELL
        if req.type == OrderType.LIMIT:
            return mt5.ORDER_TYPE_BUY_LIMIT if req.side is Side.BUY else mt5.ORDER_TYPE_SELL_LIMIT
        if req.type == OrderType.STOP:
            return mt5.ORDER_TYPE_BUY_STOP if req.side is Side.BUY else mt5.ORDER_TYPE_SELL_STOP
        if req.type == OrderType.STOP_LIMIT:
            return (
                mt5.ORDER_TYPE_BUY_STOP_LIMIT if req.side is Side.BUY else mt5.ORDER_TYPE_SELL_STOP_LIMIT
            )
        raise ValueError(f"unsupported order type: {req.type}")

    async def modify_order(
        self,
        ticket: int,
        *,
        sl: float | None = None,
        tp: float | None = None,
        price: float | None = None,
    ) -> OrderResult:
        self._require_connected()
        mt5 = self._mt5()
        positions = await self._sdk(mt5.positions_get, ticket=ticket)
        if positions:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": sl,
                "tp": tp,
            }
            result = await self._sdk(mt5.order_send, request)
        else:
            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "order": ticket,
                "price": price,
                "sl": sl,
                "tp": tp,
            }
            result = await self._sdk(mt5.order_send, request)
        if result is None:
            err = await self._sdk(mt5.last_error)
            return OrderResult(ok=False, status=OrderStatus.REJECTED, message=str(err))
        ok = int(result.retcode) == mt5.TRADE_RETCODE_DONE
        return OrderResult(
            ok=ok,
            ticket=ticket,
            status=OrderStatus.SUBMITTED if ok else OrderStatus.REJECTED,
            message=str(result.comment or ""),
            raw_code=int(result.retcode),
        )

    async def close_position(
        self, ticket: int, volume: float | None = None
    ) -> OrderResult:
        self._require_connected()
        mt5 = self._mt5()
        positions = await self._sdk(mt5.positions_get, ticket=ticket)
        if not positions:
            return OrderResult(
                ok=False, status=OrderStatus.REJECTED, message=f"position {ticket} not found"
            )
        pos = positions[0]
        close_side = Side.SELL if pos.type == mt5.ORDER_TYPE_BUY else Side.BUY
        tick = await self._sdk(mt5.symbol_info_tick, pos.symbol)
        price = tick.ask if close_side is Side.BUY else tick.bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": float(volume or pos.volume),
            "type": mt5.ORDER_TYPE_SELL if close_side is Side.SELL else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": int(pos.magic),
            "comment": "close",
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        result = await self._sdk(mt5.order_send, request)
        if result is None:
            err = await self._sdk(mt5.last_error)
            return OrderResult(ok=False, status=OrderStatus.REJECTED, message=str(err))
        ok = int(result.retcode) == mt5.TRADE_RETCODE_DONE
        return OrderResult(
            ok=ok,
            ticket=ticket,
            status=OrderStatus.FILLED if ok else OrderStatus.REJECTED,
            message=str(result.comment or ""),
            raw_code=int(result.retcode),
        )

    async def cancel_order(self, ticket: int) -> OrderResult:
        self._require_connected()
        mt5 = self._mt5()
        request = {"action": mt5.TRADE_ACTION_REMOVE, "order": ticket}
        result = await self._sdk(mt5.order_send, request)
        if result is None:
            err = await self._sdk(mt5.last_error)
            return OrderResult(ok=False, status=OrderStatus.REJECTED, message=str(err))
        ok = int(result.retcode) == mt5.TRADE_RETCODE_DONE
        return OrderResult(
            ok=ok,
            ticket=ticket,
            status=OrderStatus.CANCELLED if ok else OrderStatus.REJECTED,
            message=str(result.comment or ""),
            raw_code=int(result.retcode),
        )

    # --- State queries ------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        self._require_connected()
        mt5 = self._mt5()
        raw = await self._sdk(mt5.positions_get)
        out: list[Position] = []
        for p in raw or []:
            out.append(
                Position(
                    ticket=int(p.ticket),
                    symbol=p.symbol,
                    side=Side.BUY if p.type == mt5.ORDER_TYPE_BUY else Side.SELL,
                    volume=float(p.volume),
                    open_price=float(p.price_open),
                    open_time=datetime.fromtimestamp(int(p.time), tz=UTC),
                    sl=float(p.sl) if p.sl else None,
                    tp=float(p.tp) if p.tp else None,
                    swap=float(p.swap),
                    profit=float(p.profit),
                    comment=str(p.comment or ""),
                    magic=int(p.magic),
                )
            )
        return out

    async def get_open_orders(self) -> list[Order]:
        self._require_connected()
        mt5 = self._mt5()
        raw = await self._sdk(mt5.orders_get)
        out: list[Order] = []
        for o in raw or []:
            out.append(
                Order(
                    ticket=int(o.ticket),
                    strategy_id="",  # unknown — fill in from our DB by magic if you need it
                    symbol=o.symbol,
                    side=Side.BUY if o.type in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_BUY_LIMIT,
                                                 mt5.ORDER_TYPE_BUY_STOP) else Side.SELL,
                    type=self._order_type_from_mt5(mt5, int(o.type)),
                    volume=float(o.volume_initial),
                    price=float(o.price_open),
                    sl=float(o.sl) if o.sl else None,
                    tp=float(o.tp) if o.tp else None,
                    status=OrderStatus.SUBMITTED,
                    comment=str(o.comment or ""),
                    magic=int(o.magic),
                )
            )
        return out

    @staticmethod
    def _order_type_from_mt5(mt5, code: int) -> OrderType:
        if code in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL):
            return OrderType.MARKET
        if code in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
            return OrderType.LIMIT
        if code in (mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_SELL_STOP):
            return OrderType.STOP
        if code in (mt5.ORDER_TYPE_BUY_STOP_LIMIT, mt5.ORDER_TYPE_SELL_STOP_LIMIT):
            return OrderType.STOP_LIMIT
        return OrderType.MARKET
