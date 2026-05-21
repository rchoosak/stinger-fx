"""Bar aggregator — turns TickEvents into BarEvents for a (symbol, timeframe).

Used in live mode for sub-minute timeframes that MT5 doesn't natively emit
(M2, M3, M10, M45), and as a safety net for native timeframes too — emitting
a `BarEvent(is_closed=True)` exactly once per bar close.

The aggregator subscribes to TickEvent for its symbol and writes:
  • intra-bar updates with `is_closed=False`  (low-cost; strategies can opt out)
  • a final `is_closed=True` event at the bar boundary
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from stinger_fx.core.event_bus import AsyncEventBus
from stinger_fx.core.events import BarEvent, TickEvent
from stinger_fx.domain import Bar, Tick, Timeframe

logger = logging.getLogger("stinger.broker.aggregator")


def _bar_open_time(t: datetime, tf: Timeframe) -> datetime:
    """Truncate `t` to the bar boundary for the given timeframe (UTC)."""
    if tf is Timeframe.TICK:
        raise ValueError("TICK has no bar boundary")
    if tf is Timeframe.W1:
        # ISO week: Monday 00:00 UTC
        midnight = t.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight - timedelta(days=midnight.weekday())
    if tf is Timeframe.MN1:
        return t.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    secs = tf.seconds
    epoch = int(t.astimezone(UTC).timestamp())
    floored = epoch - (epoch % secs)
    return datetime.fromtimestamp(floored, tz=UTC)


class BarAggregator:
    """Stateful per-(symbol, tf) aggregator. One instance per (symbol, tf)."""

    def __init__(self, symbol: str, tf: Timeframe, bus: AsyncEventBus) -> None:
        if tf is Timeframe.TICK:
            raise ValueError("BarAggregator requires a non-TICK timeframe")
        self.symbol = symbol
        self.tf = tf
        self.bus = bus
        self._open_time: datetime | None = None
        self._o = self._h = self._l = self._c = 0.0
        self._tick_count = 0

    async def on_tick(self, evt: TickEvent) -> None:
        t = evt.tick
        if t.symbol != self.symbol:
            return
        boundary = _bar_open_time(t.time, self.tf)

        if self._open_time is None:
            self._reset(boundary, t)
            return

        if boundary > self._open_time:
            # Bar closed — emit the closed bar, then start a new one
            await self._emit(is_closed=True)
            self._reset(boundary, t)
            return

        # Same bar — update OHLC
        price = t.bid  # use bid for OHLC (convention; configurable later)
        self._h = max(self._h, price)
        self._l = min(self._l, price)
        self._c = price
        self._tick_count += 1
        # (intra-bar updates are deliberately not emitted to keep volume down;
        # turn this on if strategies need it.)

    def _reset(self, open_time: datetime, t: Tick) -> None:
        price = t.bid
        self._open_time = open_time
        self._o = self._h = self._l = self._c = price
        self._tick_count = 1

    async def _emit(self, *, is_closed: bool) -> None:
        assert self._open_time is not None
        bar = Bar(
            symbol=self.symbol,
            timeframe=self.tf,
            time=self._open_time,
            open=self._o,
            high=self._h,
            low=self._l,
            close=self._c,
            tick_volume=self._tick_count,
            is_closed=is_closed,
        )
        await self.bus.publish(BarEvent(bar=bar))
