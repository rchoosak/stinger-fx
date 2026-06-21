"""TradeRepo.recent_pnls_for + BacktestRepo.latest_metrics_for — the drift
monitor's live + baseline data sources."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from stinger_fx.data import BacktestRepo, TradeRepo, in_memory_store

BASE = datetime(2024, 1, 1, tzinfo=UTC)


def _add(repo: TradeRepo, *, sid: str, pnl: float, i: int) -> None:
    repo.add(
        position_id=i,
        strategy_id=sid,
        symbol="XAUUSD",
        side="buy",
        open_ts=BASE + timedelta(minutes=i),
        close_ts=BASE + timedelta(minutes=i + 1),
        open_price=1.0,
        close_price=1.1,
        volume=0.1,
        pnl=pnl,
    )


def test_recent_pnls_for_returns_newest_first_for_strategy() -> None:
    store = in_memory_store()
    repo = TradeRepo(store)
    for i, pnl in enumerate([1.0, 2.0, 3.0, 4.0]):
        _add(repo, sid="s1", pnl=pnl, i=i)
    _add(repo, sid="s2", pnl=99.0, i=100)  # different strategy — ignored

    assert repo.recent_pnls_for("s1", limit=2) == [4.0, 3.0]  # newest first
    assert repo.recent_pnls_for("s1", limit=10) == [4.0, 3.0, 2.0, 1.0]
    assert repo.recent_pnls_for("s2", limit=10) == [99.0]
    assert repo.recent_pnls_for("nobody", limit=10) == []


def test_latest_metrics_for_returns_newest_finished_run() -> None:
    store = in_memory_store()
    repo = BacktestRepo(store)
    rid1 = repo.start_run("run1", "s1", {})
    repo.finish_run(rid1, {"win_rate": 0.4, "expectancy": 5.0}, "r1.json")
    rid2 = repo.start_run("run2", "s1", {})
    repo.finish_run(rid2, {"win_rate": 0.55, "expectancy": 8.0}, "r2.json")

    m = repo.latest_metrics_for("s1")
    assert m is not None
    assert m["win_rate"] == 0.55  # newest finished
    assert m["expectancy"] == 8.0


def test_latest_metrics_for_ignores_unfinished_and_missing() -> None:
    store = in_memory_store()
    repo = BacktestRepo(store)
    repo.start_run("run1", "s1", {})  # started, never finished
    assert repo.latest_metrics_for("s1") is None
    assert repo.latest_metrics_for("other") is None
