"""Safe expression evaluator for custom metrics."""

from __future__ import annotations

import pytest

from stinger_fx.backtest.metric_dsl import (
    MetricDSLError,
    SAFE_FUNCTIONS,
    compile_metric,
    evaluate_custom_metrics,
)


# --- Happy path -------------------------------------------------------------


def test_arithmetic_expression() -> None:
    expr = compile_metric("net_pnl + sharpe * 100")
    assert expr.evaluate({"net_pnl": 10.0, "sharpe": 0.5}) == pytest.approx(60.0)


def test_division_with_safe_denominator() -> None:
    expr = compile_metric("net_pnl / (max_drawdown + 1)")
    assert expr.evaluate({"net_pnl": 100.0, "max_drawdown": 9.0}) == pytest.approx(10.0)


def test_ternary_expression() -> None:
    expr = compile_metric("sharpe if max_drawdown < 15 else sharpe * 0.5")
    assert expr.evaluate({"sharpe": 2.0, "max_drawdown": 10.0}) == pytest.approx(2.0)
    assert expr.evaluate({"sharpe": 2.0, "max_drawdown": 20.0}) == pytest.approx(1.0)


def test_unary_negation() -> None:
    expr = compile_metric("-net_pnl")
    assert expr.evaluate({"net_pnl": 5.0}) == pytest.approx(-5.0)


def test_safe_function_call_min_max_abs() -> None:
    expr = compile_metric("max(net_pnl, 0) + abs(sharpe)")
    assert expr.evaluate({"net_pnl": -5.0, "sharpe": -2.0}) == pytest.approx(2.0)
    assert expr.evaluate({"net_pnl": 10.0, "sharpe": -2.0}) == pytest.approx(12.0)


def test_chained_comparison() -> None:
    expr = compile_metric("0 < sharpe < 5")
    assert expr.evaluate({"sharpe": 2.0}) is True
    assert expr.evaluate({"sharpe": -1.0}) is False
    assert expr.evaluate({"sharpe": 6.0}) is False


def test_boolean_combinators() -> None:
    expr = compile_metric("sharpe > 1 and max_drawdown < 20")
    assert expr.evaluate({"sharpe": 2.0, "max_drawdown": 10.0}) is True
    assert expr.evaluate({"sharpe": 0.5, "max_drawdown": 10.0}) is False


def test_power_operator() -> None:
    expr = compile_metric("net_pnl ** 2")
    assert expr.evaluate({"net_pnl": 4.0}) == pytest.approx(16.0)


def test_free_variables_collected() -> None:
    expr = compile_metric("net_pnl + sharpe - max(profit_factor, 1)")
    assert expr.free_variables == {"net_pnl", "sharpe", "profit_factor"}


# --- Compile-time errors ----------------------------------------------------


def test_rejects_attribute_access() -> None:
    # Attribute access shows up as a non-Name function target → rejected
    # by the "only direct function calls allowed" check.
    with pytest.raises(MetricDSLError, match="(disallowed|direct function calls)"):
        compile_metric("net_pnl.bit_length()")


def test_rejects_subscript() -> None:
    with pytest.raises(MetricDSLError, match="disallowed"):
        compile_metric("metrics[0]")


def test_rejects_undefined_function() -> None:
    with pytest.raises(MetricDSLError, match="safe whitelist"):
        compile_metric("eval('1+1')")


def test_rejects_keyword_args() -> None:
    with pytest.raises(MetricDSLError, match="keyword"):
        compile_metric("round(net_pnl, ndigits=2)")


def test_rejects_lambda() -> None:
    # Lambda call: the "function" target is a Lambda node, not a Name —
    # caught by the "direct function calls only" rule. (The Lambda body
    # itself is also disallowed, but we hit the call check first.)
    with pytest.raises(MetricDSLError, match="(disallowed|direct function calls)"):
        compile_metric("(lambda x: x * 2)(net_pnl)")


def test_rejects_string_literal() -> None:
    with pytest.raises(MetricDSLError, match="disallowed constant"):
        compile_metric("net_pnl + 'a'")


def test_rejects_syntax_error() -> None:
    with pytest.raises(MetricDSLError, match="syntax error"):
        compile_metric("net_pnl +")


def test_rejects_call_chain() -> None:
    with pytest.raises(MetricDSLError, match="direct function calls"):
        compile_metric("min(net_pnl)()")


# --- Runtime errors ---------------------------------------------------------


def test_undefined_metric_at_eval() -> None:
    expr = compile_metric("net_pnl + sharpe_v2")
    with pytest.raises(MetricDSLError, match="undefined metric 'sharpe_v2'"):
        expr.evaluate({"net_pnl": 1.0})


def test_division_by_zero_at_eval() -> None:
    expr = compile_metric("net_pnl / max_drawdown")
    with pytest.raises(MetricDSLError, match="division by zero"):
        expr.evaluate({"net_pnl": 1.0, "max_drawdown": 0.0})


def test_type_error_at_eval() -> None:
    """min() needs comparable values — mixing None breaks at runtime."""
    expr = compile_metric("min(net_pnl, max_drawdown)")
    with pytest.raises(MetricDSLError):
        expr.evaluate({"net_pnl": None, "max_drawdown": 5.0})  # type: ignore[dict-item]


# --- Whitelist coverage -----------------------------------------------------


def test_safe_functions_are_callable() -> None:
    """Every name in SAFE_FUNCTIONS should be callable as advertised."""
    for name, fn in SAFE_FUNCTIONS.items():
        # 1-arg call works for all of them except min/max which need 2
        if name in ("min", "max"):
            assert fn(1.0, 2.0) is not None
        elif name in ("log",):
            assert fn(2.0) > 0
        else:
            assert fn(1.0) is not None


# --- Batch helper -----------------------------------------------------------


def test_evaluate_custom_metrics_adds_all_to_dict() -> None:
    base = {"net_pnl": 100.0, "sharpe": 2.0, "max_drawdown": 10.0}
    custom = {
        "risk_adjusted": "sharpe - 0.5 * max_drawdown / 10",
        "pnl_per_dd": "net_pnl / (max_drawdown + 1)",
    }
    result = evaluate_custom_metrics(custom, base)
    assert result["risk_adjusted"] == pytest.approx(2.0 - 0.5 * 10.0 / 10)  # 1.5
    assert result["pnl_per_dd"] == pytest.approx(100.0 / 11.0)
    # Base metrics untouched
    assert result["net_pnl"] == 100.0
    assert result["sharpe"] == 2.0
