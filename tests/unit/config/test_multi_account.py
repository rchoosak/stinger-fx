"""AppConfig multi-account validation + backward-compat shim."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stinger_fx.config.models import (
    AppConfig,
    BrokerConfig,
    MT5Config,
    StrategyEntry,
)


def _app_with(**kwargs) -> AppConfig:
    """Build an AppConfig with the bare-minimum required fields."""
    return AppConfig(**kwargs)


def test_legacy_single_broker_keeps_working() -> None:
    cfg = _app_with(broker=BrokerConfig(type="mt5", mt5=MT5Config()))
    bl = cfg.broker_list
    assert len(bl) == 1
    assert bl[0].id == "default"
    assert bl[0].type == "mt5"


def test_new_multi_broker_list_resolves() -> None:
    cfg = _app_with(
        brokers=[
            BrokerConfig(id="primary", type="mt5", mt5=MT5Config()),
            BrokerConfig(id="secondary", type="mt5", mt5=MT5Config()),
        ]
    )
    bl = cfg.broker_list
    assert [b.id for b in bl] == ["primary", "secondary"]


def test_both_broker_and_brokers_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        _app_with(
            broker=BrokerConfig(type="mt5", mt5=MT5Config()),
            brokers=[BrokerConfig(id="x", type="mt5", mt5=MT5Config())],
        )
    assert "keep only one" in str(exc.value)


def test_neither_broker_nor_brokers_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        _app_with()
    assert "broker" in str(exc.value).lower()


def test_duplicate_broker_ids_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        _app_with(
            brokers=[
                BrokerConfig(id="x", type="mt5", mt5=MT5Config()),
                BrokerConfig(id="x", type="mt5", mt5=MT5Config()),
            ]
        )
    assert "duplicate broker ids" in str(exc.value)


def test_strategy_account_defaults_to_default() -> None:
    s = StrategyEntry(id="ma", class_path="a.b:C", params={})
    assert s.account == "default"


def test_strategy_account_field_accepts_id_pattern() -> None:
    s = StrategyEntry(id="ma", class_path="a.b:C", account="primary-acc_1", params={})
    assert s.account == "primary-acc_1"


def test_strategy_account_rejects_bad_pattern() -> None:
    with pytest.raises(ValidationError):
        StrategyEntry(id="ma", class_path="a.b:C", account="with space", params={})
