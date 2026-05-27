"""Portfolio aggregation — combined equity, correlations, contributions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from stinger_fx.backtest.portfolio import aggregate_portfolio
from stinger_fx.backtest.reports import BacktestReport, TradeRecord


def _make_report(
    run_id: str,
    *,
    strategy_id: str,
    initial: float = 10_000.0,
    pnls: list[float] | None = None,
    base_time: datetime | None = None,
) -> BacktestReport:
    """Build a BacktestReport with a synthetic equity curve from per-trade P&L."""
    if base_time is None:
        base_time = datetime(2024, 1, 1, tzinfo=UTC)
    pnls = pnls or [0.0]
    trades: list[TradeRecord] = []
    equity_curve: list[tuple[datetime, float]] = []
    balance = initial
    for i, p in enumerate(pnls):
        ts = base_time + timedelta(hours=i)
        balance += p
        trades.append(TradeRecord(
            open_ts=ts, close_ts=ts + timedelta(minutes=30),
            side="buy", open_price=1.0, close_price=1.0 + p / 100,
            volume=0.1, pnl=p,
        ))
        equity_curve.append((ts, balance))
    return BacktestReport(
        run_id=run_id,
        strategy_id=strategy_id,
        started_at=base_time,
        finished_at=base_time + timedelta(hours=len(pnls)),
        trades=trades,
        equity_curve=equity_curve,
        initial_balance=initial,
        final_balance=balance,
    )


# --- Basic aggregation ------------------------------------------------------


def test_aggregate_single_report_yields_same_metrics() -> None:
    """One-report portfolio should match that report's numbers exactly."""
    r = _make_report("a", strategy_id="strat_a", pnls=[10, -5, 8, -3])
    p = aggregate_portfolio([r])
    assert p.initial_balance == 10_000.0
    assert p.final_balance == 10_010.0  # 10 - 5 + 8 - 3
    assert p.net_pnl == pytest.approx(10.0)
    assert len(p.contributions) == 1
    assert p.contributions[0].share_pct == pytest.approx(100.0)


def test_aggregate_two_reports_combines_balances() -> None:
    a = _make_report("a", strategy_id="strat_a", pnls=[10, 20, 30], initial=10_000)
    b = _make_report("b", strategy_id="strat_b", pnls=[5, -5, 15], initial=5_000)
    p = aggregate_portfolio([a, b])
    assert p.initial_balance == 15_000.0
    # a: 10_000 + 60 = 10_060;  b: 5_000 + 15 = 5_015 → total 15_075
    assert p.final_balance == pytest.approx(15_075.0)
    assert p.net_pnl == pytest.approx(75.0)


def test_contribution_shares_sum_to_100() -> None:
    a = _make_report("a", strategy_id="s1", pnls=[10, 20])  # +30
    b = _make_report("b", strategy_id="s2", pnls=[5, 5])    # +10
    p = aggregate_portfolio([a, b])
    total = sum(c.share_pct for c in p.contributions)
    assert total == pytest.approx(100.0, abs=0.01)


def test_aggregate_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="at least one"):
        aggregate_portfolio([])


# --- Equity curve merge -----------------------------------------------------


def test_combined_equity_curve_uses_union_of_timestamps() -> None:
    """Each report has its own timeline — combined curve covers both."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    a = _make_report("a", strategy_id="s1", pnls=[10, 20], base_time=base)
    # b starts 3 hours later
    b = _make_report("b", strategy_id="s2", pnls=[5, 5], base_time=base + timedelta(hours=3))
    p = aggregate_portfolio([a, b])
    # 4 timestamps total (2 from a + 2 from b, no overlap)
    assert len(p.equity_curve) == 4
    # First timestamp is from a — b should be at its initial balance
    first_ts, first_eq = p.equity_curve[0]
    assert first_eq == pytest.approx(10_010 + 10_000)  # a after 1st trade + b's initial


# --- Correlation ------------------------------------------------------------


def test_correlation_perfectly_correlated_strategies() -> None:
    """Two identical strategies should have correlation 1.0 with themselves
    and with each other."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    a = _make_report("a", strategy_id="s1", pnls=[10, 20, -5, 15], base_time=base)
    b = _make_report("b", strategy_id="s2", pnls=[10, 20, -5, 15], base_time=base)
    p = aggregate_portfolio([a, b])
    assert p.correlation_matrix["a"]["a"] == pytest.approx(1.0)
    assert p.correlation_matrix["a"]["b"] == pytest.approx(1.0)


def test_correlation_inverse_strategies() -> None:
    """Two opposite strategies should be strongly anti-correlated."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    a = _make_report("a", strategy_id="s1", pnls=[10, 20, -5, 15], base_time=base)
    b = _make_report("b", strategy_id="s2", pnls=[-10, -20, 5, -15], base_time=base)
    p = aggregate_portfolio([a, b])
    assert p.correlation_matrix["a"]["b"] == pytest.approx(-1.0, abs=0.01)


def test_correlation_single_report_self_only() -> None:
    """One-report portfolio has only the self-correlation entry."""
    r = _make_report("a", strategy_id="s1", pnls=[10, 5])
    p = aggregate_portfolio([r])
    assert p.correlation_matrix == {"a": {"a": 1.0}}


# --- Portfolio metrics ------------------------------------------------------


def test_max_drawdown_computed_from_combined_curve() -> None:
    """Drawdown should be on the combined curve, not naive max of components."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    # Strategy that surges then crashes
    a = _make_report("a", strategy_id="s1", pnls=[100, 100, -150, -50], base_time=base)
    p = aggregate_portfolio([a])
    # Peak after 2nd trade: 10_200; trough after 3rd: 10_050; drawdown ~= 1.47%
    assert p.max_drawdown_pct > 1.0
    assert p.max_drawdown_pct < 3.0


def test_sharpe_is_a_real_number() -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    a = _make_report("a", strategy_id="s1", pnls=[10, 5, 8, -3, 12, -2], base_time=base)
    b = _make_report("b", strategy_id="s2", pnls=[5, -2, 8, 3, -5, 10], base_time=base)
    p = aggregate_portfolio([a, b])
    assert isinstance(p.sharpe, float)


# --- Serialisation ----------------------------------------------------------


def test_to_summary_is_json_serialisable() -> None:
    import json

    base = datetime(2024, 1, 1, tzinfo=UTC)
    a = _make_report("a", strategy_id="s1", pnls=[10, 5], base_time=base)
    b = _make_report("b", strategy_id="s2", pnls=[5, 8], base_time=base)
    p = aggregate_portfolio([a, b])
    payload = p.to_summary()
    encoded = json.dumps(payload, default=str)
    decoded = json.loads(encoded)
    assert "contributions" in decoded
    assert "correlation_matrix" in decoded
    assert "equity_curve" in decoded
    assert len(decoded["contributions"]) == 2
