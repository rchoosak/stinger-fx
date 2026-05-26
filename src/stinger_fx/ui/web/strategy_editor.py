"""In-browser strategy code editor — safety helpers.

The actual FastAPI routes live in :mod:`stinger_fx.ui.web.server`; this
module hosts the validation / path-resolution / scaffold helpers so they
can be unit-tested without dragging in the FastAPI app.

Two threat vectors the helpers defend against:

  1. **Path traversal** — a malicious ``name`` like ``../../etc/passwd``
     could escape ``user_strategies_dir``. We accept only names that
     match ``[A-Za-z0-9_-]+`` (no slashes, dots, traversal), then
     resolve the candidate path and assert it's still under the root.

  2. **Syntactically broken code** — saving Python that won't parse
     would break the next config-reload. ``validate_source`` calls
     ``ast.parse`` and returns a structured error instead of writing.

A scaffold helper produces a minimal strategy template a user can fill in.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

# Names of strategy files: alphanumeric + underscore/hyphen. No dots, no
# slashes, no absolute paths.  Always lowercased before persistence to
# avoid case-insensitive filesystem surprises on macOS.
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class ValidationError:
    """Returned by validate_source when AST parse fails."""

    message: str
    line: int | None = None
    column: int | None = None


def is_safe_name(name: str) -> bool:
    """Accept only ``[a-z][a-z0-9_]*``. Lowercase Python-style module names."""
    return bool(_NAME_PATTERN.match(name))


def resolve_path(user_strategies_dir: Path, name: str) -> Path:
    """Resolve ``name`` to ``{dir}/{name}.py`` and verify it stays inside the root.

    Raises ``ValueError`` when:
      * ``name`` doesn't match the safe pattern
      * The resolved path escapes ``user_strategies_dir`` (path traversal)
    """
    if not is_safe_name(name):
        raise ValueError(f"invalid strategy name: {name!r}")
    root = user_strategies_dir.resolve()
    candidate = (root / f"{name}.py").resolve()
    # `is_relative_to` is the safe way to check containment after .resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise ValueError(
            f"strategy path escapes user_strategies_dir: {candidate}"
        ) from e
    return candidate


def validate_source(source: str) -> ValidationError | None:
    """Run ``ast.parse`` on ``source``. Returns None on success, or a
    structured ValidationError when the source has a syntax error."""
    try:
        ast.parse(source)
    except SyntaxError as e:
        return ValidationError(
            message=str(e.msg),
            line=e.lineno,
            column=e.offset,
        )
    return None


def list_strategies(user_strategies_dir: Path) -> list[str]:
    """Return the names (without ``.py``) of every strategy file in the
    user directory. Empty list when the directory doesn't exist."""
    if not user_strategies_dir.exists() or not user_strategies_dir.is_dir():
        return []
    out = []
    for path in sorted(user_strategies_dir.iterdir()):
        if path.is_file() and path.suffix == ".py":
            stem = path.stem
            if is_safe_name(stem):
                out.append(stem)
    return out


SCAFFOLD_TEMPLATE = '''"""Auto-generated strategy: {name}.

Replace the body of `on_bar` (and other hooks) with your trading logic.
See `stinger_fx.strategies.examples.ma_crossover` for a working reference.
"""

from __future__ import annotations

from pydantic import Field

from stinger_fx.domain import Bar, Subscription, Timeframe
from stinger_fx.strategies.base import BaseStrategy
from stinger_fx.strategies.context import StrategyContext
from stinger_fx.strategies.parameters import StrategyParams


class {class_name}Params(StrategyParams):
    symbol: str = "EURUSD"
    timeframe: Timeframe = Timeframe.M15
    volume: float = Field(0.01, gt=0)


class {class_name}(BaseStrategy):
    name = "{name}"
    Params = {class_name}Params

    @classmethod
    def subscriptions(cls, params: StrategyParams) -> list[Subscription]:
        assert isinstance(params, {class_name}Params)
        return [Subscription(symbol=params.symbol, timeframe=params.timeframe)]

    async def on_bar(self, ctx: StrategyContext, bar: Bar) -> None:
        # TODO: emit ctx.buy(...) / ctx.sell(...) based on `bar` and ctx.history
        return None
'''


def scaffold_source(name: str) -> str:
    """Generate a starter strategy source from the template.

    ``name`` must be a safe module name; the class name is derived as
    CamelCase from the module name (foo_bar → FooBar).
    """
    if not is_safe_name(name):
        raise ValueError(f"invalid strategy name: {name!r}")
    class_name = "".join(part.capitalize() for part in name.split("_"))
    return SCAFFOLD_TEMPLATE.format(name=name, class_name=class_name)
