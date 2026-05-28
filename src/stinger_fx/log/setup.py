"""structlog configuration — one entry point, called once on startup.

Categories use Python logger names rooted at `stinger.*` so they can be
addressed individually by stdlib config. Each category-specific JSONL file
is created on demand by `add_jsonl_handler`.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import orjson
import structlog


def _orjson_dumps(value: object, default: Callable[[Any], Any] | None = None) -> str:
    """Pass-through to orjson.dumps decoded to str — matches the signature
    JSONRenderer expects (it calls us as a json.dumps drop-in)."""
    return orjson.dumps(value, default=default).decode()


def configure(
    *,
    level: str = "INFO",
    log_dir: Path | None = None,
    console: bool = True,
) -> None:
    """Wire structlog and stdlib logging.

    - All loggers route through structlog processors (JSON for files, pretty
      console renderer for stdout).
    - Per-category JSONL handlers can be attached via `add_jsonl_handler`.
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    # The processor list mixes types whose precise structlog signatures aren't
    # exported. cast lets us hand it to ProcessorFormatter / structlog.configure
    # without per-item annotations.
    shared_pre = cast(
        "list[Any]",
        [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ],
    )

    structlog.configure(
        processors=[
            *shared_pre,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Console formatter (human-readable)
    console_fmt = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_pre,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()),
        ],
    )
    # JSON formatter (file sinks)
    json_fmt = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_pre,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(serializer=_orjson_dumps),
        ],
    )

    root = logging.getLogger()
    # Reset any prior config (re-configure on hot log_level change is fine).
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(console_fmt)
        root.addHandler(ch)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        # A single combined log captures everything; categories below add their own.
        fh = logging.FileHandler(log_dir / "engine.jsonl", encoding="utf-8")
        fh.setFormatter(json_fmt)
        root.addHandler(fh)

    # Stash the JSON formatter so add_jsonl_handler can reuse it
    root._stinger_json_formatter = json_fmt  # type: ignore[attr-defined]


def add_jsonl_handler(logger_name: str, log_dir: Path, filename: str | None = None) -> None:
    """Route a specific logger to its own JSONL file (in addition to root)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    name = filename or f"{logger_name.replace('.', '_')}.jsonl"
    handler = logging.FileHandler(log_dir / name, encoding="utf-8")
    root = logging.getLogger()
    fmt = getattr(root, "_stinger_json_formatter", None)
    if fmt is not None:
        handler.setFormatter(fmt)
    logging.getLogger(logger_name).addHandler(handler)
    # Don't disable propagation: the root combined log keeps the audit trail.


def set_level(level: str) -> None:
    """Apply a log-level change at runtime (used by hot reload)."""
    logging.getLogger().setLevel(level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return cast("structlog.stdlib.BoundLogger", structlog.get_logger(name))
