"""Custom exception hierarchy for Stinger-Fx."""

from __future__ import annotations


class StingerError(Exception):
    """Base for all Stinger-Fx errors."""


class ConfigError(StingerError):
    """Raised when configuration is invalid or cannot be loaded."""


class BrokerError(StingerError):
    """Raised by broker adapters for connection / order failures."""


class BrokerNotConnectedError(BrokerError):
    """A broker call was made before connect() succeeded."""


class StrategyError(StingerError):
    """Raised by the strategy framework — user code failures bubble through here."""


class StrategyQuarantinedError(StrategyError):
    """A strategy was disabled by the runtime after exceeding error budget."""


class RiskCheckFailed(StingerError):
    """The order router refused a signal."""


class BacktestError(StingerError):
    """Raised by backtest engines."""
