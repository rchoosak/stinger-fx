from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from stinger_fx.backtest.reports import BacktestReport, TradeRecord


def _trade(pnl: float, *, side: str = "buy") -> TradeRecord:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    return TradeRecord(
        open_ts=t0, close_ts=t0 + timedelta(hours=1),
        side=side, open_price=1.10, close_price=1.11, volume=0.1, pnl=pnl,
    )


def _report(trades: list[TradeRecord]) -> BacktestReport:
    return BacktestReport(
        run_id="r1",
        strategy_id="s1",
        started_at=datetime(2024, 1, 1, tzinfo=UTC),
        finished_at=datetime(2024, 1, 2, tzinfo=UTC),
        trades=trades,
        equity_curve=[],
        initial_balance=10_000.0,
        final_balance=10_000.0 + sum(t.pnl for t in trades),
    )


def test_metrics_with_mixed_trades() -> None:
    r = _report([_trade(100), _trade(-30), _trade(70), _trade(-20)])
    assert r.net_pnl == pytest.approx(120.0)
    assert r.gross_profit == 170
    assert r.gross_loss == -50
    assert r.win_rate == 0.5
    assert r.profit_factor == pytest.approx(170 / 50)
    assert r.expectancy == pytest.approx(120 / 4)


def test_empty_trades_metrics() -> None:
    r = _report([])
    assert r.net_pnl == 0
    assert r.win_rate == 0.0
    assert r.expectancy == 0.0
    assert r.profit_factor == 0.0
    assert r.sharpe == 0.0


def test_max_drawdown_from_equity_curve() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    r = BacktestReport(
        run_id="r1",
        strategy_id="s1",
        started_at=t0,
        finished_at=t0,
        equity_curve=[
            (t0, 10_000),
            (t0 + timedelta(hours=1), 10_500),  # peak
            (t0 + timedelta(hours=2), 10_200),
            (t0 + timedelta(hours=3),  9_800),  # trough — DD = 700
            (t0 + timedelta(hours=4), 10_100),
        ],
        initial_balance=10_000.0,
        final_balance=10_100.0,
    )
    assert r.max_drawdown == pytest.approx(700.0)
