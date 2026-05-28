"""BrokerPool — keyed access + primary semantics."""

from __future__ import annotations

from typing import cast

import pytest

from stinger_fx.brokers import BrokerPool
from stinger_fx.brokers.base import BaseBroker
from stinger_fx.core.errors import ConfigError


class _FakeBroker(BaseBroker):
    name = "fake"
    tag: str = ""  # test sentinel for pool ordering assertions
    async def connect(self): ...
    async def disconnect(self): ...
    async def is_connected(self): return True
    async def get_account_info(self): raise NotImplementedError
    async def get_account_snapshot(self): raise NotImplementedError
    async def get_symbol_info(self, symbol): raise NotImplementedError
    async def list_symbols(self): return []
    async def subscribe_ticks(self, symbol): ...
    async def subscribe_bars(self, symbol, tf): ...
    async def unsubscribe(self, symbol, tf=None): ...
    async def get_history_bars(self, *a, **kw): raise NotImplementedError
    async def get_history_ticks(self, *a, **kw): raise NotImplementedError
    async def place_order(self, req): raise NotImplementedError
    async def modify_order(self, ticket, **kw): raise NotImplementedError
    async def close_position(self, ticket, volume=None): raise NotImplementedError
    async def cancel_order(self, ticket): raise NotImplementedError
    async def get_positions(self): return []
    async def get_open_orders(self): return []


def _broker(tag: str) -> _FakeBroker:
    b = _FakeBroker(None)  # type: ignore[arg-type]
    b.tag = tag
    return b


def test_pool_get_by_account_id() -> None:
    a, b = _broker("a"), _broker("b")
    pool = BrokerPool([("primary", a), ("secondary", b)])
    assert cast(_FakeBroker, pool.get("primary")).tag == "a"
    assert cast(_FakeBroker, pool.get("secondary")).tag == "b"
    assert pool.has("primary")
    assert not pool.has("nope")


def test_pool_primary_is_first_added() -> None:
    a, b = _broker("first"), _broker("second")
    pool = BrokerPool([("a", a), ("b", b)])
    assert cast(_FakeBroker, pool.primary()).tag == "first"
    assert pool.primary_id() == "a"


def test_pool_rejects_duplicate_ids() -> None:
    pool = BrokerPool([("x", _broker("a"))])
    with pytest.raises(ConfigError):
        pool.add("x", _broker("b"))


def test_pool_get_unknown_raises() -> None:
    pool = BrokerPool([("a", _broker("a"))])
    with pytest.raises(KeyError):
        pool.get("missing")


def test_pool_empty_primary_raises() -> None:
    with pytest.raises(KeyError):
        BrokerPool().primary()


def test_pool_all_and_items_preserve_insertion_order() -> None:
    a, b, c = _broker("a"), _broker("b"), _broker("c")
    pool = BrokerPool([("x", a), ("y", b), ("z", c)])
    assert [cast(_FakeBroker, t).tag for t in pool.all()] == ["a", "b", "c"]
    assert [k for k, _ in pool.items()] == ["x", "y", "z"]
    assert len(pool) == 3
