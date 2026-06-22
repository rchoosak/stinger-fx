"""Durable strategy position state: persist/restore + restart reconciliation."""

from __future__ import annotations

from datetime import UTC, datetime

from stinger_fx.domain import Position, Side
from stinger_fx.strategies.state_store import (
    InMemoryStateStore,
    JsonFileStateStore,
    PositionState,
    reconcile,
)

SID = "d1h4_xau"
SYMBOL = "XAUUSD"


def _state(*, side: str = Side.BUY.value, entry: float = 2000.0, ticket: int = 555,
           stop: float = 1990.0) -> PositionState:
    return PositionState(
        strategy_id=SID, symbol=SYMBOL, side=side,
        entry_price=entry, ticket=ticket, chandelier_stop=stop,
    )


def _pos(*, side: Side = Side.BUY, open_price: float = 2000.0,
         ticket: int = 555) -> Position:
    return Position(
        ticket=ticket, symbol=SYMBOL, side=side, volume=0.1,
        open_price=open_price, open_time=datetime.now(UTC), sl=1990.0,
    )


def test_in_memory_roundtrip() -> None:
    store = InMemoryStateStore()
    assert store.load(SID) is None
    store.save(_state(stop=1995.0))
    got = store.load(SID)
    assert got is not None and got.chandelier_stop == 1995.0
    store.clear(SID)
    assert store.load(SID) is None


def test_json_file_survives_restart(tmp_path) -> None:
    path = tmp_path / "state.json"
    JsonFileStateStore(path).save(_state(stop=1993.0, ticket=777))
    # A fresh store instance == a process restart reading the same file.
    restored = JsonFileStateStore(path).load(SID)
    assert restored is not None
    assert restored.ticket == 777 and restored.chandelier_stop == 1993.0


def test_json_file_clear(tmp_path) -> None:
    path = tmp_path / "state.json"
    store = JsonFileStateStore(path)
    store.save(_state())
    store.clear(SID)
    assert JsonFileStateStore(path).load(SID) is None


def test_json_file_corrupt_is_treated_as_empty(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{ not valid json", encoding="utf-8")
    assert JsonFileStateStore(path).load(SID) is None  # never blocks startup


def test_reconcile_restores_on_exact_match() -> None:
    st = _state()
    out = reconcile(st, [_pos()], strategy_id=SID, symbol=SYMBOL)
    assert out is st


def test_reconcile_rejects_side_mismatch() -> None:
    st = _state(side=Side.BUY.value)
    out = reconcile(st, [_pos(side=Side.SELL)], strategy_id=SID, symbol=SYMBOL)
    assert out is None


def test_reconcile_rejects_entry_mismatch() -> None:
    st = _state(entry=2000.0)
    out = reconcile(st, [_pos(open_price=2000.5)], strategy_id=SID, symbol=SYMBOL)
    assert out is None


def test_reconcile_rejects_ticket_mismatch() -> None:
    st = _state(ticket=555)
    out = reconcile(st, [_pos(ticket=999)], strategy_id=SID, symbol=SYMBOL)
    assert out is None


def test_reconcile_rejects_when_no_live_position() -> None:
    assert reconcile(_state(), [], strategy_id=SID, symbol=SYMBOL) is None


def test_reconcile_none_persisted_is_none() -> None:
    assert reconcile(None, [_pos()], strategy_id=SID, symbol=SYMBOL) is None


def test_reconcile_rejects_strategy_or_symbol_mismatch() -> None:
    st = _state()
    assert reconcile(st, [_pos()], strategy_id="other", symbol=SYMBOL) is None
    assert reconcile(st, [_pos()], strategy_id=SID, symbol="EURUSD") is None
