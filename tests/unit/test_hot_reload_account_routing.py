"""Regression tests for the hot-reload account-routing hotfix.

Pre-fix bugs
============

Two related defects in the hot-reload path (config/reload.py + runtime.py):

1. **``_diff_strategies`` ignored the ``account`` field.** It only
   compared ``class_path`` / ``enabled`` / ``params``. Changing a
   strategy's ``account:`` in ``strategies.yaml`` was detected
   (``o != n``) but no branch matched, so the reload was a silent
   no-op and the strategy kept trading on the previous account.

2. **The ``add_strategy`` callback ignored ``entry.account``.** It
   called ``_wire_broker_subscriptions(broker)`` with the broker
   captured at startup (the primary), and it never updated
   ``self._strategy_accounts`` or ``router.strategy_accounts``. So a
   new strategy hot-added with ``account: demo_b`` would:
     * route signals through ``demo_a`` (router falls back to primary
       when the strategy_id is missing from ``strategy_accounts``);
     * subscribe its symbols on ``demo_a`` instead of ``demo_b``.

Fix
===

* ``ReloadActions`` gains a ``change_account`` callback.
  ``_diff_strategies`` calls it when only ``account`` changes.
* ``runtime._build_reload_actions``:
    - ``add`` callback now updates ``self._strategy_accounts`` and
      ``router.strategy_accounts`` from ``entry.account``, and
      subscribes the entry's broker (not the captured primary) to the
      strategy's feeds.
    - ``remove`` callback drops the strategy_id from both maps so a
      later re-add starts clean.
    - ``change_account`` callback flips both maps live, subscribes the
      new broker, and rejects unknown account_ids loudly so a typo in
      YAML doesn't silently downgrade routing back to primary.

These tests exercise the runtime callbacks end-to-end against a stubbed
broker pool — they don't crack open the YAML watcher (covered in
``test_reload.py``) but they prove the wire-up itself is correct.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stinger_fx.brokers.base import BaseBroker
from stinger_fx.config.models import StrategyEntry
from stinger_fx.runtime import StingerApp


def _write_multi_account_app_yaml(path: Path, data_dir: Path) -> None:
    path.write_text(
        "mode: normal\n"
        f"data_dir: {data_dir}\n"
        "brokers:\n"
        "  - id: demo_a\n"
        "    type: mt5\n"
        "    mt5: {}\n"
        "  - id: demo_b\n"
        "    type: mt5\n"
        "    mt5: {}\n"
        "risk:\n"
        "  max_open_positions_per_strategy: 5\n"
    )


def _write_strategies_yaml(path: Path, *, account: str = "demo_a") -> None:
    path.write_text(
        "strategies:\n"
        "  - id: ma\n"
        "    class_path: stinger_fx.strategies.examples.ma_crossover:MACrossover\n"
        "    enabled: true\n"
        f"    account: {account}\n"
        "    params: {symbol: EURUSD, timeframe: M15}\n"
    )


class _StubBroker(BaseBroker):
    """Tracks subscribe_bars calls so tests can assert which broker the
    add / change_account paths actually wired."""

    name = "stub"

    def __init__(self, bus, account_id: str = ""):
        super().__init__(bus)
        self.account_id = account_id
        self.subscribed_bars: list[tuple[str, str]] = []

    async def connect(self): ...
    async def disconnect(self): ...
    async def is_connected(self): return False
    async def get_account_info(self): raise NotImplementedError
    async def get_account_snapshot(self): raise NotImplementedError
    async def get_symbol_info(self, symbol): raise NotImplementedError
    async def list_symbols(self): return []
    async def subscribe_ticks(self, symbol): ...

    async def subscribe_bars(self, symbol, tf):
        self.subscribed_bars.append((symbol, tf.value))

    async def unsubscribe(self, symbol, tf=None): ...

    async def get_history_bars(self, *a, **kw):
        from stinger_fx.data.parquet_store import BAR_SCHEMA
        return BAR_SCHEMA.empty_table()

    async def get_history_ticks(self, *a, **kw):
        from stinger_fx.data.parquet_store import TICK_SCHEMA
        return TICK_SCHEMA.empty_table()

    async def place_order(self, req): raise NotImplementedError
    async def modify_order(self, ticket, **kw): raise NotImplementedError
    async def close_position(self, ticket, volume=None): raise NotImplementedError
    async def cancel_order(self, ticket): raise NotImplementedError
    async def get_positions(self): return []
    async def get_open_orders(self): return []


@pytest.fixture
async def app(monkeypatch, tmp_path: Path):
    """Build a configured + set-up StingerApp on a multi-account config."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_multi_account_app_yaml(config_dir / "app.yaml", tmp_path / "data")
    _write_strategies_yaml(config_dir / "strategies.yaml", account="demo_a")
    (config_dir / "backtest.yaml").write_text("runs: []\n")

    # Each call to build_broker yields a fresh stub tagged with the cfg id.
    # That lets tests differentiate which broker received subscribe_bars.
    def fake_build_broker(cfg, bus):
        return _StubBroker(bus, account_id=cfg.id)

    monkeypatch.setattr("stinger_fx.runtime.build_broker", fake_build_broker)

    a = StingerApp(config_dir)
    await a.setup()
    yield a
    # cleanup
    for runner in list(a.runners.values()):
        await runner.stop()
    await a.engine.bus.close()  # type: ignore[union-attr]


# --- 1. change_account flips the routing maps -----------------------------


@pytest.mark.asyncio
async def test_change_account_updates_strategy_accounts_map(app) -> None:
    """The ``change_account`` callback must update
    ``self._strategy_accounts[sid]`` from the old account to the new one.
    Pre-fix this map was never updated on reload — the strategy kept
    routing to its startup account."""
    actions = app._build_reload_actions(app._pool.primary())

    # Strategy starts on demo_a per the YAML.
    assert app._strategy_accounts["ma"] == "demo_a"

    await actions.change_account("ma", "demo_b")

    assert app._strategy_accounts["ma"] == "demo_b", (
        f"_strategy_accounts not flipped — got {app._strategy_accounts}"
    )


@pytest.mark.asyncio
async def test_change_account_updates_router_strategy_accounts(app) -> None:
    """And it must update ``router.strategy_accounts`` too — that's what
    ``OrderRouter._broker_for(sid)`` actually consults to route an order."""
    actions = app._build_reload_actions(app._pool.primary())
    assert app._router is not None
    assert app._router.strategy_accounts["ma"] == "demo_a"

    await actions.change_account("ma", "demo_b")

    assert app._router.strategy_accounts["ma"] == "demo_b", (
        f"router.strategy_accounts not flipped — got "
        f"{app._router.strategy_accounts}"
    )


@pytest.mark.asyncio
async def test_change_account_subscribes_new_broker(app) -> None:
    """After re-routing, the new broker must be subscribed to the
    strategy's symbols so its tick pump actually wakes up for them."""
    actions = app._build_reload_actions(app._pool.primary())

    # Baseline: demo_b has no subscriptions yet (this strategy is on demo_a).
    demo_b = app._pool.get("demo_b")
    assert demo_b.subscribed_bars == []  # type: ignore[attr-defined]

    await actions.change_account("ma", "demo_b")

    assert ("EURUSD", "M15") in demo_b.subscribed_bars, (  # type: ignore[attr-defined]
        f"demo_b not subscribed to the strategy's feeds after re-route — "
        f"got {demo_b.subscribed_bars}"  # type: ignore[attr-defined]
    )


# --- 2. change_account guard rails ----------------------------------------


@pytest.mark.asyncio
async def test_change_account_unknown_account_raises(app) -> None:
    """A typo in YAML must not silently downgrade routing back to the
    primary broker (which is what would happen if we just stored the
    unknown id — the router's ``_broker_for`` falls back to primary).
    Reject loudly."""
    actions = app._build_reload_actions(app._pool.primary())

    with pytest.raises(ValueError, match="unknown account"):
        await actions.change_account("ma", "no_such_broker")

    # And the original routing is untouched.
    assert app._strategy_accounts["ma"] == "demo_a"


@pytest.mark.asyncio
async def test_change_account_unknown_strategy_is_noop(app) -> None:
    """A change_account for a strategy id that isn't currently running
    should NOT raise (it might race with a removal); it just logs and
    skips."""
    actions = app._build_reload_actions(app._pool.primary())
    # No exception — strategy 'nope' doesn't exist.
    await actions.change_account("nope", "demo_b")


# --- 3. add_strategy now honours entry.account ----------------------------


@pytest.mark.asyncio
async def test_add_strategy_respects_entry_account(app) -> None:
    """When a brand-new strategy is hot-added with ``account: demo_b``,
    the add callback must:
      * register `_strategy_accounts['new'] = 'demo_b'`
      * register `router.strategy_accounts['new'] = 'demo_b'`
      * subscribe demo_b (NOT the primary demo_a) to the strategy's feeds

    Pre-fix none of these happened — the new strategy fell back to
    primary in the router and its ticks were polled by demo_a only."""
    actions = app._build_reload_actions(app._pool.primary())

    new_entry = StrategyEntry(
        id="new_strat",
        class_path="stinger_fx.strategies.examples.ma_crossover:MACrossover",
        enabled=True,
        account="demo_b",
        params={"symbol": "GBPUSD", "timeframe": "M15"},
    )

    await actions.add_strategy(new_entry)

    assert app._strategy_accounts["new_strat"] == "demo_b", (
        f"new strategy account not registered — got "
        f"{app._strategy_accounts}"
    )
    assert app._router is not None
    assert app._router.strategy_accounts["new_strat"] == "demo_b", (
        f"router didn't get the new strategy's account — got "
        f"{app._router.strategy_accounts}"
    )

    demo_a = app._pool.get("demo_a")
    demo_b = app._pool.get("demo_b")
    assert ("GBPUSD", "M15") in demo_b.subscribed_bars, (  # type: ignore[attr-defined]
        f"demo_b should have been subscribed to GBPUSD/M15 — got "
        f"{demo_b.subscribed_bars}"  # type: ignore[attr-defined]
    )
    assert ("GBPUSD", "M15") not in demo_a.subscribed_bars, (  # type: ignore[attr-defined]
        f"demo_a should NOT have been subscribed to GBPUSD/M15 "
        f"(that's the pre-fix bug — captured primary broker). got "
        f"{demo_a.subscribed_bars}"  # type: ignore[attr-defined]
    )


# --- 4. remove_strategy cleans up routing maps ----------------------------


@pytest.mark.asyncio
async def test_remove_strategy_drops_account_mapping(app) -> None:
    """After remove, ``_strategy_accounts`` and ``router.strategy_accounts``
    must drop the entry so a later re-add with the same id starts clean
    (otherwise stale routes can leak)."""
    actions = app._build_reload_actions(app._pool.primary())
    assert "ma" in app._strategy_accounts
    assert app._router is not None
    assert "ma" in app._router.strategy_accounts

    await actions.remove_strategy("ma")

    assert "ma" not in app._strategy_accounts
    assert "ma" not in app._router.strategy_accounts
