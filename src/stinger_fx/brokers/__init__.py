"""Broker abstraction + concrete implementations + multi-account pool."""

from stinger_fx.brokers.bar_aggregator import BarAggregator
from stinger_fx.brokers.base import BaseBroker
from stinger_fx.brokers.pool import BrokerPool
from stinger_fx.brokers.registry import build_broker

__all__ = ["BarAggregator", "BaseBroker", "BrokerPool", "build_broker"]
