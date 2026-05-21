"""Slippage models for the sim broker."""

from __future__ import annotations

from stinger_fx.domain import Side


def fixed_pips_slippage(price: float, side: Side, pips: float, point: float = 0.0001) -> float:
    """Adverse-slippage model: buyer pays more, seller receives less."""
    delta = pips * point
    return price + delta if side is Side.BUY else price - delta
