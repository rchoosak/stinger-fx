"""Pure aggregate open-risk helpers — per-position risk and the sum."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from stinger_fx.domain import Position, Side
from stinger_fx.execution.exposure import aggregate_open_risk, position_risk


def _pos(symbol: str, *, open_price: float, sl: float | None, volume: float) -> Position:
    return Position(
        ticket=1, symbol=symbol, side=Side.BUY, volume=volume,
        open_price=open_price, open_time=datetime.now(UTC), sl=sl,
    )


def test_position_risk_with_stop() -> None:
    # |2000 - 1990| x 0.1 x 100 = 100.0
    assert position_risk(2000.0, 1990.0, 0.1, 100.0) == pytest.approx(100.0)


def test_position_risk_no_stop_is_zero() -> None:
    assert position_risk(2000.0, None, 0.1, 100.0) == 0.0
    assert position_risk(2000.0, 0.0, 0.1, 100.0) == 0.0


def test_aggregate_sums_measurable_positions() -> None:
    positions = [
        _pos("XAUUSD", open_price=2000.0, sl=1990.0, volume=0.1),  # 100
        _pos("EURUSD", open_price=1.1000, sl=1.0950, volume=1.0),  # 0.005*1*100000=500
    ]
    sizes = {"XAUUSD": 100.0, "EURUSD": 100_000.0}
    assert aggregate_open_risk(positions, sizes) == pytest.approx(600.0)


def test_aggregate_skips_unknown_contract_and_stopless() -> None:
    positions = [
        _pos("XAUUSD", open_price=2000.0, sl=1990.0, volume=0.1),  # 100
        _pos("GBPUSD", open_price=1.30, sl=1.29, volume=1.0),      # no size → skip
        _pos("XAUUSD", open_price=2000.0, sl=None, volume=0.5),    # no SL → 0
    ]
    sizes = {"XAUUSD": 100.0}
    assert aggregate_open_risk(positions, sizes) == pytest.approx(100.0)
