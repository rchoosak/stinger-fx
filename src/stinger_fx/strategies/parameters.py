"""StrategyParams base — Pydantic v2 model that strategies subclass.

Concrete strategies subclass this and add typed fields with `Field(...)`
constraints. The runner validates raw YAML params against the subclass at
load time and on every hot-reload.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrategyParams(BaseModel):
    """Base class for strategy parameter models."""

    model_config = ConfigDict(extra="forbid", frozen=True)
