"""Safe expression evaluator for user-defined metrics.

Lets users compose custom metrics from the built-in ones via a YAML
expression:

    custom_metrics:
      risk_adjusted: "sharpe - 0.5 * max_drawdown / 10"
      pnl_per_dd:    "net_pnl / (max_drawdown + 1)"
      regime_score:  "sharpe if max_drawdown < 15 else sharpe * 0.5"

The expression strings are parsed via Python's ``ast`` module and
evaluated with a strict whitelist of node types — **no** access to
builtins, attributes, subscripts, lambdas, comprehensions, imports,
function definitions, or any I/O. Free variables are looked up in the
metric dict supplied at evaluation time.

Whitelisted constructs:

  * Binary arithmetic: + - * / // % **
  * Unary: + - not
  * Comparisons:  == != < <= > >=
  * Boolean: and or
  * Ternary: ``a if cond else b``
  * Constants: numbers (int/float), booleans, None
  * Names: looked up in the supplied metrics dict
  * Calls: only to the whitelisted functions in ``SAFE_FUNCTIONS``

Any other AST node raises :class:`MetricDSLError` at compile time.
Division by zero / unknown metric / type mismatch raises at evaluation
time so the calling code can surface a clear message.
"""

from __future__ import annotations

import ast
import math
import operator as op
from collections.abc import Callable
from typing import Any


class MetricDSLError(ValueError):
    """Raised for any compile-time or evaluation-time error in the DSL."""


# Allowed binary operators (Python AST → operator function)
_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    ast.Not: op.not_,
}

_CMP_OPS: dict[type, Callable[[Any, Any], bool]] = {
    ast.Eq: op.eq,
    ast.NotEq: op.ne,
    ast.Lt: op.lt,
    ast.LtE: op.le,
    ast.Gt: op.gt,
    ast.GtE: op.ge,
}

# Functions callable from inside an expression. Pure, side-effect-free,
# no I/O, no introspection.
SAFE_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}


class MetricExpression:
    """Compiled, evaluable form of a metric DSL expression."""

    def __init__(self, source: str, tree: ast.Expression) -> None:
        self.source = source
        self._tree = tree
        # Pre-collect free variable names (names that aren't function calls)
        self.free_variables = _collect_free_variables(tree)

    def evaluate(self, metrics: dict[str, Any]) -> Any:
        """Evaluate the expression with ``metrics`` as the variable scope.

        Raises :class:`MetricDSLError` on undefined variable, division by
        zero, or runtime type errors.
        """
        try:
            return _eval_node(self._tree.body, metrics)
        except KeyError as e:
            raise MetricDSLError(
                f"undefined metric {e.args[0]!r} in expression {self.source!r}"
            ) from None
        except ZeroDivisionError:
            raise MetricDSLError(
                f"division by zero in expression {self.source!r}"
            ) from None
        except (TypeError, ValueError) as e:
            raise MetricDSLError(
                f"runtime error in expression {self.source!r}: {e}"
            ) from e


def compile_metric(source: str) -> MetricExpression:
    """Parse + validate a DSL expression.

    Raises :class:`MetricDSLError` if any disallowed AST node is present.
    """
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as e:
        raise MetricDSLError(f"syntax error: {e.msg} in {source!r}") from e
    _validate(tree)
    return MetricExpression(source=source, tree=tree)


# --- Validator --------------------------------------------------------------


_ALLOWED_NODES: tuple[type, ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Call,
    ast.And,
    ast.Or,
    *_BIN_OPS,
    *_UNARY_OPS,
    *_CMP_OPS,
)


def _validate(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise MetricDSLError(
                f"disallowed node {type(node).__name__} in expression"
            )
        # Only allow numeric / bool / None constants — strings could be
        # used for format-string tricks, lists/tuples bring surface area.
        if (
            isinstance(node, ast.Constant)
            and not isinstance(node.value, (int, float, bool))
            and node.value is not None
        ):
            raise MetricDSLError(
                f"disallowed constant type {type(node.value).__name__}"
            )
        if isinstance(node, ast.Call):
            # Function must be a bare Name (not an attribute access or call
            # chain) AND must be in SAFE_FUNCTIONS.
            if not isinstance(node.func, ast.Name):
                raise MetricDSLError("only direct function calls allowed")
            if node.func.id not in SAFE_FUNCTIONS:
                raise MetricDSLError(
                    f"function {node.func.id!r} not in safe whitelist"
                )
            # No keyword args — keep the calling convention simple
            if node.keywords:
                raise MetricDSLError("keyword arguments not allowed in calls")


# --- Evaluator --------------------------------------------------------------


def _eval_node(node: ast.AST, scope: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id in SAFE_FUNCTIONS:
            return SAFE_FUNCTIONS[node.id]
        if node.id not in scope:
            raise KeyError(node.id)
        return scope[node.id]

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, scope)
        right = _eval_node(node.right, scope)
        fn = _BIN_OPS.get(type(node.op))
        if fn is None:
            raise MetricDSLError(f"disallowed binary op {type(node.op).__name__}")
        return fn(left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, scope)
        fn = _UNARY_OPS.get(type(node.op))
        if fn is None:
            raise MetricDSLError(f"disallowed unary op {type(node.op).__name__}")
        return fn(operand)

    if isinstance(node, ast.BoolOp):
        values = [_eval_node(v, scope) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        return any(values)

    if isinstance(node, ast.Compare):
        # Chain: left op1 right1 op2 right2 ...  → all must hold
        current = _eval_node(node.left, scope)
        for cmp_op, right_node in zip(node.ops, node.comparators, strict=True):
            right = _eval_node(right_node, scope)
            fn = _CMP_OPS.get(type(cmp_op))
            if fn is None:
                raise MetricDSLError(f"disallowed comparator {type(cmp_op).__name__}")
            if not fn(current, right):
                return False
            current = right
        return True

    if isinstance(node, ast.IfExp):
        test = _eval_node(node.test, scope)
        return _eval_node(node.body if test else node.orelse, scope)

    if isinstance(node, ast.Call):
        # Function whitelist already enforced at compile time
        assert isinstance(node.func, ast.Name)
        fn = SAFE_FUNCTIONS[node.func.id]
        args = [_eval_node(a, scope) for a in node.args]
        return fn(*args)

    raise MetricDSLError(f"unexpected node {type(node).__name__}")


def _collect_free_variables(tree: ast.AST) -> set[str]:
    """Collect Name nodes that aren't bound to whitelisted functions."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in SAFE_FUNCTIONS:
            names.add(node.id)
    return names


# --- Convenience batch evaluation -------------------------------------------


def evaluate_custom_metrics(
    custom_metrics: dict[str, str],
    base_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Compile + evaluate every custom-metric expression against a base
    metric dict. Returns ``base_metrics`` with custom metrics added.

    Custom metrics can't reference other custom metrics (avoid cycles) —
    expressions only resolve against ``base_metrics``.
    """
    out = dict(base_metrics)
    for name, expr in custom_metrics.items():
        compiled = compile_metric(expr)
        out[name] = compiled.evaluate(base_metrics)
    return out
