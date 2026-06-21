"""Risk-based position sizing.

`size_by_risk` turns a risk budget (a % of account equity) plus a stop distance
into a lot size, using the same `price_move × volume × contract_size` P&L model
the engine settles trades with — so a trade sized to risk 1% loses ~1% of
equity if its stop is hit.

Returns ``0.0`` when it can't produce a tradable size (no stop distance, the
result rounds below the symbol's minimum lot, etc.); the caller falls back to
the strategy's fixed volume rather than placing a zero/oversize order.

Assumes the account currency equals the symbol's profit currency (e.g. a USD
account trading USD-quoted XAUUSD / EURUSD). Cross-currency conversion is out of
scope — the engine's P&L model makes the same assumption, so sizing stays
consistent with realized P&L.
"""

from __future__ import annotations

import math


def size_by_risk(
    *,
    equity: float,
    risk_pct: float,
    entry: float,
    sl: float,
    contract_size: float,
    volume_min: float,
    volume_max: float,
    volume_step: float,
) -> float:
    """Lot size risking ``risk_pct`` of ``equity`` at the stop. 0.0 if unsizable."""
    risk_amount = equity * risk_pct / 100.0
    stop_dist = abs(entry - sl)
    if (
        stop_dist <= 0
        or contract_size <= 0
        or risk_amount <= 0
        or volume_step <= 0
    ):
        return 0.0
    raw = risk_amount / (stop_dist * contract_size)
    # Round DOWN to the lot step — never round up into more risk than budgeted.
    stepped = math.floor(raw / volume_step) * volume_step
    if stepped < volume_min:
        return 0.0
    return min(stepped, volume_max)
