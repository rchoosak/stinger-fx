"""Execution package import compatibility."""

from stinger_fx.backtest.order_router import OrderRouter as BacktestOrderRouter
from stinger_fx.execution import OrderRouter as ExecutionOrderRouter


def test_backtest_order_router_import_is_compat_shim() -> None:
    assert BacktestOrderRouter is ExecutionOrderRouter
