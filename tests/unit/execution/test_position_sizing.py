"""size_by_risk — risk-based lot sizing math."""

from __future__ import annotations

import pytest

from stinger_fx.execution.position_sizing import size_by_risk

_BOUNDS = {"volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01}


def test_basic_one_percent_risk() -> None:
    # 1% of 10_000 = $100 risk; stop 10.0 price units × contract 100 = $1000/lot
    # → 0.10 lot.
    vol = size_by_risk(
        equity=10_000, risk_pct=1.0, entry=2000.0, sl=1990.0,
        contract_size=100.0, **_BOUNDS,
    )
    assert vol == pytest.approx(0.10)


def test_rounds_down_to_volume_step() -> None:
    # raw = 100 / (7 × 100) = 0.142857 → floored to 0.14 (never round up into
    # more risk than budgeted).
    vol = size_by_risk(
        equity=10_000, risk_pct=1.0, entry=2000.0, sl=1993.0,
        contract_size=100.0, **_BOUNDS,
    )
    assert vol == pytest.approx(0.14)


def test_below_min_lot_returns_zero() -> None:
    # Tiny account → sized below volume_min → 0.0 (caller falls back to fixed).
    vol = size_by_risk(
        equity=10.0, risk_pct=1.0, entry=2000.0, sl=1990.0,
        contract_size=100.0, **_BOUNDS,
    )
    assert vol == 0.0


def test_clamped_to_volume_max() -> None:
    vol = size_by_risk(
        equity=1_000_000_000, risk_pct=1.0, entry=2000.0, sl=1999.0,
        contract_size=100.0, **_BOUNDS,
    )
    assert vol == pytest.approx(100.0)


def test_zero_stop_distance_returns_zero() -> None:
    assert size_by_risk(
        equity=10_000, risk_pct=1.0, entry=2000.0, sl=2000.0,
        contract_size=100.0, **_BOUNDS,
    ) == 0.0


def test_non_positive_equity_returns_zero() -> None:
    assert size_by_risk(
        equity=0.0, risk_pct=1.0, entry=2000.0, sl=1990.0,
        contract_size=100.0, **_BOUNDS,
    ) == 0.0
    assert size_by_risk(
        equity=-5.0, risk_pct=1.0, entry=2000.0, sl=1990.0,
        contract_size=100.0, **_BOUNDS,
    ) == 0.0


def test_sl_above_entry_uses_absolute_distance() -> None:
    # SELL setup: sl above entry. |2000 − 2010| = 10 → same 0.10 lot.
    vol = size_by_risk(
        equity=10_000, risk_pct=1.0, entry=2000.0, sl=2010.0,
        contract_size=100.0, **_BOUNDS,
    )
    assert vol == pytest.approx(0.10)
