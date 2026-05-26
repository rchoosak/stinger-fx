"""TrailingStopManager — ratchets SL toward the market as price moves favourably.

Behaviour:
  • For a BUY position, the SL tracks `bid - distance_pips * point`. It
    can only *advance* (move up), never retreat — once a stop is moved
    higher it stays there even if the market dips below the trigger
    again.
  • For a SELL position, mirror: SL tracks `ask + distance_pips * point`,
    only moves down.
  • Optional `activate_after_pips`: don't start trailing until the
    position has accrued that much unrealised profit. Useful for letting
    a trade breathe before clamping a stop on it.

The manager is symbol-aware — pass `symbol=` to restrict to one symbol;
omit it to manage every tagged position. Multi-symbol strategies usually
want one manager per symbol.
"""

from __future__ import annotations

import logging

from stinger_fx.domain import Side, Tick
from stinger_fx.strategies.context import StrategyContext

logger = logging.getLogger("stinger.strategy.trailing")


class TrailingStopManager:
    """One instance per (strategy, symbol-or-all).

    `point` defaults to 0.0001 — the standard pip for 4-digit majors. For
    JPY-quoted pairs pass `point=0.01`.
    """

    def __init__(
        self,
        ctx: StrategyContext,
        *,
        distance_pips: float,
        activate_after_pips: float = 0.0,
        symbol: str | None = None,
        point: float = 0.0001,
    ) -> None:
        if distance_pips <= 0:
            raise ValueError(f"distance_pips must be > 0, got {distance_pips!r}")
        if activate_after_pips < 0:
            raise ValueError(
                f"activate_after_pips must be ≥ 0, got {activate_after_pips!r}"
            )
        self._ctx = ctx
        self._distance = distance_pips * point
        self._activate = activate_after_pips * point
        self._symbol = symbol
        self._point = point
        # Best SL we've ever set for each ticket — keeps us from chasing the
        # market backwards across a refresh.
        self._best_sl: dict[int, float] = {}

    async def on_tick(self, ctx: StrategyContext, tick: Tick) -> None:
        if self._symbol is not None and tick.symbol != self._symbol:
            return
        # Only manage positions owned by this strategy. PositionView is
        # already magic-filtered, so a straight iteration is correct.
        for pos in ctx.position.for_symbol(tick.symbol):
            await self._maybe_advance(ctx, pos.ticket, pos.side, pos.open_price, pos.sl, tick)

    async def _maybe_advance(
        self,
        ctx: StrategyContext,
        ticket: int,
        side: Side,
        open_price: float,
        current_sl: float | None,
        tick: Tick,
    ) -> None:
        if side is Side.BUY:
            unrealised = tick.bid - open_price
            if unrealised < self._activate:
                return
            desired = tick.bid - self._distance
            # Only advance — don't retreat.
            prev_best = self._best_sl.get(ticket, current_sl if current_sl is not None else -float("inf"))
            if desired <= prev_best:
                return
            new_sl = desired
        else:  # SELL
            unrealised = open_price - tick.ask
            if unrealised < self._activate:
                return
            desired = tick.ask + self._distance
            prev_best = self._best_sl.get(ticket, current_sl if current_sl is not None else float("inf"))
            if desired >= prev_best:
                return
            new_sl = desired

        self._best_sl[ticket] = new_sl
        await ctx.move_stop(ticket, sl=new_sl, reason="trailing")
        logger.debug(
            "trailing.advance ticket=%s side=%s old_sl=%s new_sl=%s tick_bid=%s tick_ask=%s",
            ticket, side.value, current_sl, new_sl, tick.bid, tick.ask,
        )
