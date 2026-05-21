"""Broker factory — picks the concrete implementation from `BrokerConfig`."""

from __future__ import annotations

from stinger_fx.brokers.base import BaseBroker
from stinger_fx.config.models import BrokerConfig
from stinger_fx.core.errors import ConfigError
from stinger_fx.core.event_bus import AsyncEventBus


def build_broker(cfg: BrokerConfig, bus: AsyncEventBus) -> BaseBroker:
    if cfg.type == "mt5":
        # Lazy import — MetaTrader5 is Windows-only and we don't want a top-level
        # import to break unit tests on Mac/Linux.
        from stinger_fx.brokers.mt5.broker import MT5Broker

        if cfg.mt5 is None:
            raise ConfigError("broker.type=mt5 but broker.mt5 block is missing")
        return MT5Broker(bus=bus, cfg=cfg.mt5)
    if cfg.type == "mt4":
        raise ConfigError("MT4 broker is a Phase-2 feature; not yet implemented")
    raise ConfigError(f"unknown broker type: {cfg.type}")
