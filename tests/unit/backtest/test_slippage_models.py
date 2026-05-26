"""Slippage model factories — fixed, spread, volatility."""

from __future__ import annotations

import pytest

from stinger_fx.backtest.slippage import (
    build_slippage_model,
    fixed_pips_model,
    fixed_pips_slippage,
    spread_model,
    volatility_model,
)
from stinger_fx.domain import Side


# ---------------------------------------------------------------------------
# fixed_pips_model
# ---------------------------------------------------------------------------


def test_fixed_model_buy_worse_than_mid() -> None:
    """BUY fill must be above mid when pips > 0."""
    model = fixed_pips_model(pips=2, point=0.0001)
    mid = (1.1000 + 1.1002) / 2  # 1.1001
    fill = model(Side.BUY, bid=1.1000, ask=1.1002)
    assert fill > mid
    assert fill == pytest.approx(1.1001 + 2 * 0.0001)  # mid + 2pips


def test_fixed_model_sell_worse_than_mid() -> None:
    """SELL fill must be below mid when pips > 0."""
    model = fixed_pips_model(pips=2, point=0.0001)
    mid = (1.1000 + 1.1002) / 2
    fill = model(Side.SELL, bid=1.1000, ask=1.1002)
    assert fill < mid
    assert fill == pytest.approx(1.1001 - 2 * 0.0001)  # mid - 2pips


def test_fixed_model_zero_pips_fills_at_mid() -> None:
    """Zero pips → fill at mid regardless of side."""
    model = fixed_pips_model(pips=0)
    bid, ask = 1.1000, 1.1002
    mid = (bid + ask) / 2
    assert model(Side.BUY, bid=bid, ask=ask) == pytest.approx(mid)
    assert model(Side.SELL, bid=bid, ask=ask) == pytest.approx(mid)


# ---------------------------------------------------------------------------
# spread_model
# ---------------------------------------------------------------------------


def test_spread_model_buy_at_ask() -> None:
    """BUY must fill exactly at ask (no extra cost beyond spread)."""
    model = spread_model()
    fill = model(Side.BUY, bid=1.1000, ask=1.1002)
    assert fill == pytest.approx(1.1002)


def test_spread_model_sell_at_bid() -> None:
    """SELL must fill exactly at bid."""
    model = spread_model()
    fill = model(Side.SELL, bid=1.1000, ask=1.1002)
    assert fill == pytest.approx(1.1000)


# ---------------------------------------------------------------------------
# volatility_model
# ---------------------------------------------------------------------------


def test_volatility_model_scales_with_range() -> None:
    """After a large price range has built up, slippage should be larger."""
    model = volatility_model(factor=0.5, window=5)

    # Warm up with volatile bids (range = 0.0050)
    for bid in (1.1000, 1.1010, 1.0990, 1.1020, 1.0980):
        fill_volatile = model(Side.BUY, bid=bid, ask=bid + 0.0002)
    fill_volatile = model(Side.BUY, bid=1.1000, ask=1.1002)

    # Fresh instance with stable bids (range ≈ 0)
    model_calm = volatility_model(factor=0.5, window=5)
    for bid in (1.1000, 1.1001, 1.1000, 1.1001, 1.1000):
        _ = model_calm(Side.BUY, bid=bid, ask=bid + 0.0002)
    fill_calm = model_calm(Side.BUY, bid=1.1000, ask=1.1002)

    assert fill_volatile > fill_calm, (
        "volatile market should produce higher slippage than calm market"
    )


def test_volatility_model_stateful() -> None:
    """Same model instance should accumulate bid history across calls."""
    model = volatility_model(factor=1.0, window=3, point=0.0001)
    # Single call — window not full
    fill_1 = model(Side.BUY, bid=1.1000, ask=1.1002)
    # After 3 calls with spread bids, window fills and range grows
    model(Side.BUY, bid=1.1010, ask=1.1012)
    model(Side.BUY, bid=1.0990, ask=1.0992)
    fill_3 = model(Side.BUY, bid=1.1000, ask=1.1002)
    # The fill after more volatile history should be higher (or equal) than the first
    assert fill_3 >= fill_1


# ---------------------------------------------------------------------------
# build_slippage_model
# ---------------------------------------------------------------------------


def test_build_fixed() -> None:
    fn = build_slippage_model("fixed", pips=1)
    assert fn(Side.BUY, bid=1.1000, ask=1.1000) == pytest.approx(1.1001)


def test_build_spread() -> None:
    fn = build_slippage_model("spread")
    assert fn(Side.BUY, bid=1.1000, ask=1.1002) == pytest.approx(1.1002)


def test_build_volatility() -> None:
    fn = build_slippage_model("volatility", volatility_factor=0.1)
    # Just check it returns a float and doesn't raise
    result = fn(Side.BUY, bid=1.1000, ask=1.1002)
    assert isinstance(result, float)


def test_build_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="unknown slippage model"):
        build_slippage_model("magic")


# ---------------------------------------------------------------------------
# Backward-compat: fixed_pips_slippage (old signature)
# ---------------------------------------------------------------------------


def test_legacy_fixed_pips_slippage_buy() -> None:
    result = fixed_pips_slippage(1.1000, Side.BUY, pips=2)
    assert result == pytest.approx(1.1002)


def test_legacy_fixed_pips_slippage_sell() -> None:
    result = fixed_pips_slippage(1.1000, Side.SELL, pips=2)
    assert result == pytest.approx(1.0998)
