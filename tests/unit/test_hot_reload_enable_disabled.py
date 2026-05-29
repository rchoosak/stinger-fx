"""Regression tests for the disabled-strategy hot-enable hotfix.

Pre-fix bug
===========

``StingerApp.setup`` (runtime.py:130-134) skips any strategy entry with
``enabled: false`` — no runner is created. Later, when the user edits
``strategies.yaml`` to flip ``enabled: false → true``, the reload diff
(reload.py:154) calls ``set_enabled(sid, True)``. The runtime callback
(runtime.py:431) then short-circuits because ``self.runners.get(sid)
is None``:

    async def set_enabled(sid: str, enabled: bool) -> None:
        runner = self.runners.get(sid)
        if runner is None:
            return       # ← bug: silent no-op
        ...

The reload is logged as "applied" but **the strategy never starts**.
Operators see the YAML change accepted, no error, no warning — and the
strategy is dead until restart.

Fix
===

When the runner is missing and ``enabled=True``, route through the
``add()`` path so the runner, magic, account mapping, and broker
subscriptions all get wired identically to a fresh add. The lookup
falls back to the current ``self.full_cfg`` (which is still the OLD
snapshot during reload — accurate for class_path/params/account
because if any of those had also changed, the diff layer would have
fired ``replace_strategy`` / ``update_params`` / ``change_account``
in addition to ``set_enabled``).

These tests pin:

  1. Disabled at startup → enabled via reload action creates a runner.
     (THE bug fix.)
  2. Enabled at startup → disabled → enabled still works (the
     pause/resume path is preserved when the runner exists).
  3. ``enabled: false`` flip on a non-existent runner is still a no-op
     (no spurious add).
  4. Account/magic mapping wired correctly after the hot-enable (so
     order routing works immediately, not only after the next reload).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stinger_fx.brokers.base import BaseBroker
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


def _write_strategies_yaml(
    path: Path, *, enabled: bool = True, account: str = "demo_a",
) -> None:
    path.write_text(
        "strategies:\n"
        "  - id: ma\n"
        "    class_path: stinger_fx.strategies.examples.ma_crossover:MACrossover\n"
        f"    enabled: {str(enabled).lower()}\n"
        f"    account: {account}\n"
        "    params: {symbol: EURUSD, timeframe: M15}\n"
    )


class _StubBroker(BaseBroker):
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
async def disabled_startup_app(monkeypatch, tmp_path: Path):
    """App that started with `ma` strategy `enabled: false` — no runner
    exists at setup time."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_multi_account_app_yaml(config_dir / "app.yaml", tmp_path / "data")
    _write_strategies_yaml(config_dir / "strategies.yaml", enabled=False)
    (config_dir / "backtest.yaml").write_text("runs: []\n")

    def fake_build_broker(cfg, bus):
        return _StubBroker(bus, account_id=cfg.id)

    monkeypatch.setattr("stinger_fx.runtime.build_broker", fake_build_broker)

    a = StingerApp(config_dir)
    await a.setup()
    yield a
    for runner in list(a.runners.values()):
        await runner.stop()
    await a.engine.bus.close()  # type: ignore[union-attr]


# --- 1. THE bug fix --------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_at_startup_can_be_enabled_via_reload(disabled_startup_app) -> None:
    """The headline regression: a strategy that started disabled must be
    startable via a hot-reload that flips `enabled: false → true`.

    Pre-fix this silently no-op'd because the runner didn't exist."""
    app = disabled_startup_app
    # Sanity: no runner at startup (was disabled).
    assert "ma" not in app.runners, (
        f"expected no runner for disabled-at-startup strategy, got "
        f"{list(app.runners)}"
    )

    # Build the reload actions exactly as the watcher would.
    actions = app._build_reload_actions(app._pool.primary())
    await actions.set_enabled("ma", True)

    # After the hot-enable, a runner must exist.
    assert "ma" in app.runners, (
        "set_enabled(True) on a disabled-at-startup strategy must create "
        f"the runner — got runners={list(app.runners)}. Pre-fix this "
        "callback short-circuited silently."
    )


@pytest.mark.asyncio
async def test_hot_enabled_strategy_gets_account_routing(disabled_startup_app) -> None:
    """The hot-enable must wire account routing too — not just create a
    runner with no broker mapping. Pre-fix wouldn't have run at all; the
    risk now is wiring it as a no-account runner that order-routes to
    primary by fallback."""
    app = disabled_startup_app
    actions = app._build_reload_actions(app._pool.primary())
    await actions.set_enabled("ma", True)

    # Strategy entry in strategies.yaml targets account=demo_a.
    assert app._strategy_accounts.get("ma") == "demo_a", (
        f"expected ma → demo_a in _strategy_accounts; got "
        f"{app._strategy_accounts}"
    )
    assert app._router is not None
    assert app._router.strategy_accounts.get("ma") == "demo_a", (
        f"router must see the new strategy's account too; got "
        f"{app._router.strategy_accounts}"
    )


@pytest.mark.asyncio
async def test_hot_enabled_strategy_subscribes_its_broker(disabled_startup_app) -> None:
    """The hot-enable must subscribe the correct broker to the strategy's
    feeds — otherwise the strategy is alive but blind."""
    app = disabled_startup_app
    actions = app._build_reload_actions(app._pool.primary())
    await actions.set_enabled("ma", True)

    demo_a = app._pool.get("demo_a")
    assert ("EURUSD", "M15") in demo_a.subscribed_bars, (  # type: ignore[attr-defined]
        f"demo_a should have been subscribed to EURUSD/M15 — got "
        f"{demo_a.subscribed_bars}"  # type: ignore[attr-defined]
    )


# --- 2. Negative case: enabled stays a no-op for missing runner ----------


@pytest.mark.asyncio
async def test_disabled_set_to_disabled_is_still_noop(disabled_startup_app) -> None:
    """If the runner doesn't exist and the call asks for `enabled=False`,
    the action must remain a no-op (don't spuriously add a runner and
    immediately pause it)."""
    app = disabled_startup_app
    actions = app._build_reload_actions(app._pool.primary())
    await actions.set_enabled("ma", False)

    assert "ma" not in app.runners, (
        "set_enabled(False) on a non-existent runner must be a no-op — "
        f"got runners={list(app.runners)}"
    )


# --- 3. Unknown strategy id ----------------------------------------------


@pytest.mark.asyncio
async def test_hot_enable_unknown_strategy_warns_and_returns(disabled_startup_app) -> None:
    """If the diff somehow asks to enable a strategy that's no longer in
    the YAML at all, the callback must skip + log rather than crash."""
    app = disabled_startup_app
    actions = app._build_reload_actions(app._pool.primary())
    # Not in config — set_enabled must skip without raising.
    await actions.set_enabled("not_in_yaml", True)
    assert "not_in_yaml" not in app.runners


# --- 4. Existing runner: pause/resume path preserved ---------------------


@pytest.fixture
async def enabled_startup_app(monkeypatch, tmp_path: Path):
    """Counterpart fixture — app with the strategy enabled at startup so
    a runner does exist. Used to assert the pause/resume path still
    works (regression guard for the fix)."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_multi_account_app_yaml(config_dir / "app.yaml", tmp_path / "data")
    _write_strategies_yaml(config_dir / "strategies.yaml", enabled=True)
    (config_dir / "backtest.yaml").write_text("runs: []\n")

    def fake_build_broker(cfg, bus):
        return _StubBroker(bus, account_id=cfg.id)

    monkeypatch.setattr("stinger_fx.runtime.build_broker", fake_build_broker)

    a = StingerApp(config_dir)
    await a.setup()
    yield a
    for runner in list(a.runners.values()):
        await runner.stop()
    await a.engine.bus.close()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_existing_runner_can_be_paused_and_resumed(enabled_startup_app) -> None:
    """Regression: when the runner exists, set_enabled must still flip
    pause/resume on it (and *not* spuriously call add)."""
    app = enabled_startup_app
    assert "ma" in app.runners
    runner = app.runners["ma"]

    actions = app._build_reload_actions(app._pool.primary())

    # Pause via set_enabled(False).
    await actions.set_enabled("ma", False)
    # Still one runner (we paused it, didn't remove).
    assert "ma" in app.runners
    assert app.runners["ma"] is runner, "set_enabled must NOT replace the runner"

    # Resume via set_enabled(True).
    await actions.set_enabled("ma", True)
    assert app.runners["ma"] is runner, "resume must not replace the runner either"
