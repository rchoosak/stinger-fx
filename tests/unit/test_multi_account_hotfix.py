"""Regression tests for the P1 multi-account startup-crash hotfix.

Background
==========
``AppConfig._normalise_brokers`` makes ``broker:`` (singular) optional when
``brokers:`` (multi-account list) is supplied. Several call sites in the
runtime + CLI previously did unconditional ``app_cfg.broker.X`` accesses,
which crashed with ``AttributeError: 'NoneType' object has no attribute X``
the moment a valid multi-account YAML was loaded.

These tests pin down the three call paths that were broken so future
refactors can't silently re-introduce the bug.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from stinger_fx.brokers.base import BaseBroker
from stinger_fx.cli import app as cli_app
from stinger_fx.config.models import AppConfig, BrokerConfig, MT5Config
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


def _write_strategies_yaml(path: Path) -> None:
    path.write_text(
        "strategies:\n"
        "  - id: ma\n"
        "    class_path: stinger_fx.strategies.examples.ma_crossover:MACrossover\n"
        "    enabled: true\n"
        "    account: demo_a\n"
        "    params: {symbol: EURUSD, timeframe: M15}\n"
    )


# --- 1. AppConfig.primary_broker_config -------------------------------------


def test_primary_broker_config_multi_account_returns_first() -> None:
    """When `brokers:` (list) is supplied, primary_broker_config returns the
    first entry — no AttributeError from `broker` being None."""
    cfg = AppConfig(
        brokers=[
            BrokerConfig(id="demo_a", type="mt5", mt5=MT5Config()),
            BrokerConfig(id="demo_b", type="mt5", mt5=MT5Config()),
        ],
    )
    assert cfg.broker is None  # multi-account → singular is None
    assert cfg.primary_broker_config.id == "demo_a"
    assert cfg.primary_broker_config.type == "mt5"


def test_primary_broker_config_single_account_returns_broker() -> None:
    """Backward compat: legacy `broker:` config still produces the same primary."""
    cfg = AppConfig(
        broker=BrokerConfig(id="default", type="mt5", mt5=MT5Config()),
    )
    assert cfg.primary_broker_config.id == "default"
    assert cfg.primary_broker_config.type == "mt5"


# --- 2. runtime.py — StingerApp.setup() with multi-account config ----------


class _StubBroker(BaseBroker):
    """Minimal broker stub — setup() doesn't connect, so we never hit the SDK."""
    name = "stub"
    async def connect(self): ...
    async def disconnect(self): ...
    async def is_connected(self): return False
    async def get_account_info(self): raise NotImplementedError
    async def get_account_snapshot(self): raise NotImplementedError
    async def get_symbol_info(self, symbol): raise NotImplementedError
    async def list_symbols(self): return []
    async def subscribe_ticks(self, symbol): ...
    async def subscribe_bars(self, symbol, tf): ...
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


@pytest.mark.asyncio
async def test_setup_runs_with_multi_account_config(monkeypatch, tmp_path: Path) -> None:
    """Regression: StingerApp.setup() must NOT crash when the config uses
    `brokers:` (multi-account). The pre-fix bug crashed on line 85 of
    runtime.py reading app_cfg.broker.type while broker was None."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_multi_account_app_yaml(config_dir / "app.yaml", tmp_path / "data")
    _write_strategies_yaml(config_dir / "strategies.yaml")
    (config_dir / "backtest.yaml").write_text("runs: []\n")

    # Stub the broker factory so we don't need the MT5 SDK
    monkeypatch.setattr("stinger_fx.runtime.build_broker", lambda cfg, bus: _StubBroker(bus))

    app = StingerApp(config_dir)
    # The bug was an AttributeError at setup-time. Just calling setup() is
    # the regression assertion — any exception fails the test.
    await app.setup()

    # Bonus assertion: the broker pool actually contains both configured accounts
    assert len(list(app._pool.items())) == 2

    # Cleanup
    for runner in list(app.runners.values()):
        await runner.stop()
    await app.engine.bus.close()  # type: ignore[union-attr]


# --- 3. CLI config validate with multi-account config ----------------------


def test_cli_config_validate_multi_account(tmp_path: Path) -> None:
    """Regression: `stinger-fx config validate` must accept a multi-account
    YAML and print the primary broker type — pre-fix it crashed on
    cli.py:150 reading cfg.app.broker.type."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_multi_account_app_yaml(config_dir / "app.yaml", tmp_path / "data")
    _write_strategies_yaml(config_dir / "strategies.yaml")
    (config_dir / "backtest.yaml").write_text("runs: []\n")

    runner = CliRunner()
    result = runner.invoke(cli_app, ["config", "validate", "--config-dir", str(config_dir)])
    assert result.exit_code == 0, f"validate failed: {result.output}"
    assert "OK:" in result.output
    assert "broker=mt5" in result.output  # primary broker's type


# --- 4. config reload diff with multi-account configs ----------------------


@pytest.mark.asyncio
async def test_reload_diff_multi_account_doesnt_crash() -> None:
    """Regression: config/reload.py:78 was an unconditional `.broker.type`
    access. Multi-account configs would crash during hot-reload diff."""
    from stinger_fx.config.reload import ConfigReloader, ReloadActions, ReloadResult

    old = AppConfig(
        brokers=[BrokerConfig(id="a", type="mt5", mt5=MT5Config())],
    )
    new = AppConfig(
        brokers=[BrokerConfig(id="a", type="mt5", mt5=MT5Config())],
    )
    # ReloadActions wires the apply-side callbacks; _diff_app doesn't call
    # any of them, so async no-op stubs are enough.
    async def _noop(*_a, **_kw): return None
    actions = ReloadActions(
        add_strategy=_noop,
        remove_strategy=_noop,
        replace_strategy=_noop,
        update_params=_noop,
        set_enabled=_noop,
        update_log_level=_noop,
        update_risk=_noop,
    )
    reloader = ConfigReloader(actions)
    result = ReloadResult()
    await reloader._diff_app(old, new, result)
    # No exception, no restart needed (configs are equal)
    assert "broker.type" not in " ".join(result.needs_restart)
