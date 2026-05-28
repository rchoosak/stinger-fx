"""Slippage models for the sim broker.

Three built-in models, each returned by a factory that produces a callable
conforming to `SlippageModel`:

* **fixed**       — adds/subtracts a constant number of pips regardless of
                    spread or volatility. Classic and fast. Default.
* **spread**      — fills BUY at ask price, SELL at bid price. Models the
                    pure bid/ask spread cost; zero extra slippage on top.
* **volatility**  — scales extra slippage by ``factor × recent_bid_range``
                    over a rolling window of ticks. Useful for liquid pairs
                    where market impact varies with intra-day volatility.

All models receive the current (bid, ask) pair so they can make an informed
fill-price decision regardless of which model is selected.

Backward-compat note: the original module-level ``fixed_pips_slippage``
function (single call, returns a price) is preserved for callers that haven't
migrated to the factory pattern.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol, runtime_checkable

from stinger_fx.domain import Side

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SlippageModel(Protocol):
    """Callable that converts market bid/ask into a fill price."""

    def __call__(self, side: Side, *, bid: float, ask: float) -> float: ...


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def fixed_pips_model(pips: float = 0.0, point: float = 0.0001) -> SlippageModel:
    """Return a model that adds/subtracts a constant ``pips`` to the mid-price.

    BUY fills at mid + pips×point (adverse for the buyer).
    SELL fills at mid − pips×point (adverse for the seller).
    When ``pips=0`` the fill is exactly at mid (no extra cost beyond spread).
    """
    delta = pips * point

    def _apply(side: Side, *, bid: float, ask: float) -> float:
        mid = (bid + ask) / 2
        return mid + delta if side is Side.BUY else mid - delta

    return _apply


def spread_model() -> SlippageModel:
    """Return a model that fills at ask (BUY) or bid (SELL).

    This is the most realistic zero-extra-slippage model for liquid markets.
    The entire cost is the broker spread; no additional pips are added.
    """

    def _apply(side: Side, *, bid: float, ask: float) -> float:
        return ask if side is Side.BUY else bid

    return _apply


def volatility_model(
    factor: float = 0.25,
    point: float = 0.0001,
    window: int = 20,
) -> SlippageModel:
    """Return a stateful model that scales slippage by recent bid range.

    Maintains a rolling window of the last ``window`` bid prices. Slippage =
    ``factor × (max_bid − min_bid)`` over that window, applied on top of the
    mid-price. As volatility increases, so does fill cost — mimicking real
    market impact.

    Parameters
    ----------
    factor:
        Fraction of the recent range to use as extra slippage (0.25 = 25%).
    point:
        One pip in price units. Used as a floor when the window is empty.
    window:
        Rolling window size (number of fills to track).
    """
    bids: deque[float] = deque(maxlen=window)

    def _apply(side: Side, *, bid: float, ask: float) -> float:
        bids.append(bid)
        spread = ask - bid
        recent_range = (max(bids) - min(bids)) if len(bids) > 1 else spread
        extra = factor * max(recent_range, point)
        mid = (bid + ask) / 2
        return mid + extra if side is Side.BUY else mid - extra

    return _apply


def build_slippage_model(
    model: str,
    *,
    pips: float = 0.0,
    point: float = 0.0001,
    volatility_factor: float = 0.25,
    volatility_window: int = 20,
) -> SlippageModel:
    """Build a `SlippageModel` from config-level string identifiers.

    Parameters
    ----------
    model:
        One of ``"fixed"``, ``"spread"``, ``"volatility"``.
    pips:
        Pip count for the ``"fixed"`` model (ignored by others).
    point:
        One pip in price units.
    volatility_factor:
        ``factor`` for the ``"volatility"`` model.
    volatility_window:
        Rolling window size for the ``"volatility"`` model.
    """
    if model == "fixed":
        return fixed_pips_model(pips=pips, point=point)
    if model == "spread":
        return spread_model()
    if model == "volatility":
        return volatility_model(factor=volatility_factor, point=point, window=volatility_window)
    raise ValueError(
        f"unknown slippage model {model!r}; expected 'fixed', 'spread', or 'volatility'"
    )


# ---------------------------------------------------------------------------
# Backward-compat helper (used by existing replay_broker internals)
# ---------------------------------------------------------------------------


def fixed_pips_slippage(price: float, side: Side, pips: float, point: float = 0.0001) -> float:
    """Adverse-slippage model: buyer pays more, seller receives less.

    .. deprecated::
        Use :func:`fixed_pips_model` factory instead.  This function is
        retained for callers that have not yet migrated.
    """
    delta = pips * point
    return price + delta if side is Side.BUY else price - delta
