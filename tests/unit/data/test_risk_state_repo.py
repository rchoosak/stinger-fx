"""RiskStateRepo + TradeRepo.realized_since — persistence used by RiskMonitor
crash recovery."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from stinger_fx.data import RiskStateRepo, TradeRepo, in_memory_store
from stinger_fx.data.schemas import RiskStateRow


def _add_trade(repo: TradeRepo, *, symbol: str, pnl: float, close_ts: datetime) -> None:
    repo.add(
        position_id=1,
        strategy_id="s1",
        symbol=symbol,
        side="buy",
        open_ts=close_ts - timedelta(minutes=5),
        close_ts=close_ts,
        open_price=1.0,
        close_price=1.1,
        volume=0.1,
        pnl=pnl,
    )


def test_risk_state_save_then_load_round_trips() -> None:
    store = in_memory_store()
    repo = RiskStateRepo(store)
    assert repo.load() is None  # nothing persisted yet

    repo.save(peak_equity=12_345.0, kill_switch_tripped=True)
    row = repo.load()
    assert row is not None
    assert row.peak_equity == 12_345.0
    assert row.kill_switch_tripped is True


def test_risk_state_save_is_an_upsert_single_row() -> None:
    store = in_memory_store()
    repo = RiskStateRepo(store)
    repo.save(peak_equity=100.0, kill_switch_tripped=False)
    repo.save(peak_equity=200.0, kill_switch_tripped=True)
    repo.save(peak_equity=300.0, kill_switch_tripped=False)

    # Latest values win and there is exactly one row (id=1).
    row = repo.load()
    assert row is not None
    assert row.peak_equity == 300.0
    assert row.kill_switch_tripped is False
    with store.session() as s:
        from sqlmodel import select

        all_rows = list(s.exec(select(RiskStateRow)))
    assert len(all_rows) == 1
    assert all_rows[0].id == 1


def test_realized_since_sums_total_and_per_symbol() -> None:
    store = in_memory_store()
    repo = TradeRepo(store)
    now = datetime.now(UTC)
    _add_trade(repo, symbol="XAUUSD", pnl=10.0, close_ts=now)
    _add_trade(repo, symbol="XAUUSD", pnl=-4.0, close_ts=now)
    _add_trade(repo, symbol="EURUSD", pnl=3.0, close_ts=now)

    total, by_symbol = repo.realized_since(now - timedelta(hours=1))
    assert total == 9.0
    assert by_symbol == {"XAUUSD": 6.0, "EURUSD": 3.0}


def test_realized_since_excludes_trades_before_cutoff() -> None:
    store = in_memory_store()
    repo = TradeRepo(store)
    now = datetime.now(UTC)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # Yesterday's loss must NOT count toward today's realized P&L.
    _add_trade(repo, symbol="XAUUSD", pnl=-500.0, close_ts=midnight - timedelta(hours=2))
    _add_trade(repo, symbol="XAUUSD", pnl=7.0, close_ts=now)

    total, by_symbol = repo.realized_since(midnight)
    assert total == 7.0
    assert by_symbol == {"XAUUSD": 7.0}


def test_realized_since_empty_when_no_trades() -> None:
    store = in_memory_store()
    repo = TradeRepo(store)
    total, by_symbol = repo.realized_since(datetime.now(UTC))
    assert total == 0.0
    assert by_symbol == {}
