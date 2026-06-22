"""Aggregate open-risk computation for the OrderRouter's exposure cap.

Open risk = the money lost if every open position's stop-loss were hit at once.
Capping it bounds the simultaneous-stop loss across all strategies sharing one
account — the danger that per-position counts and the reactive equity gates
don't see (each strategy looks fine alone; together they're over-leveraged).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from stinger_fx.domain import Position


def position_risk(
    open_price: float,
    sl: float | None,
    volume: float,
    contract_size: float,
) -> float:
    """Money at risk if this position's stop is hit, in account currency.

    Returns 0.0 when there's no usable stop (``sl`` is None or <= 0): the loss
    is unbounded and can't be measured, so it's excluded from the cap rather
    than counted as zero. Callers that care about no-SL positions flag them
    separately.
    """
    if sl is None or sl <= 0:
        return 0.0
    return abs(open_price - sl) * volume * contract_size


def aggregate_open_risk(
    positions: Iterable[Position],
    contract_sizes: Mapping[str, float],
) -> float:
    """Sum of per-position risk over all open positions.

    A position whose symbol is absent from (or non-positive in)
    ``contract_sizes`` is skipped — its risk can't be measured without the
    contract size, so it's left out rather than guessed.
    """
    total = 0.0
    for p in positions:
        cs = contract_sizes.get(p.symbol)
        if cs is None or cs <= 0:
            continue
        total += position_risk(p.open_price, p.sl, p.volume, cs)
    return total
