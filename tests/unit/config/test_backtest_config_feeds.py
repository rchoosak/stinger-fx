"""BacktestRunConfig + SweepRunConfig feed-shape normalisation (Phase 4)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from stinger_fx.config.models import BacktestRunConfig, SweepRunConfig
from stinger_fx.domain import Subscription, Timeframe


def _common_run_kwargs() -> dict:
    return {
        "id": "r",
        "strategy_id": "s",
        "start": datetime(2024, 1, 1, tzinfo=UTC),
        "end": datetime(2024, 2, 1, tzinfo=UTC),
    }


def _common_sweep_kwargs() -> dict:
    return {
        "id": "sweep",
        "strategy_id": "s",
        "start": datetime(2024, 1, 1, tzinfo=UTC),
        "end": datetime(2024, 2, 1, tzinfo=UTC),
        "data_source": Path("/tmp"),
        "parameter_grid": {"x": [1, 2]},
    }


# --- BacktestRunConfig --------------------------------------------------------


def test_singular_feeds_resolve_to_one_entry() -> None:
    cfg = BacktestRunConfig(symbol="EURUSD", timeframe=Timeframe.M15, **_common_run_kwargs())
    assert cfg.feed_list == [Subscription(symbol="EURUSD", timeframe=Timeframe.M15)]
    # Primary attributes match the lone feed
    assert cfg.symbol == "EURUSD"
    assert cfg.timeframe == Timeframe.M15


def test_plural_feeds_are_cartesian_product_sorted() -> None:
    cfg = BacktestRunConfig(
        symbols=["EURUSD", "GBPUSD"],
        timeframes=[Timeframe.M15, Timeframe.H1],
        **_common_run_kwargs(),
    )
    assert len(cfg.feed_list) == 4
    # Sorted by (symbol, tf.value) for deterministic merge ties
    feed_tuples = [(f.symbol, f.timeframe.value) for f in cfg.feed_list]
    assert feed_tuples == [
        ("EURUSD", "H1"),
        ("EURUSD", "M15"),
        ("GBPUSD", "H1"),
        ("GBPUSD", "M15"),
    ]
    # Primary attributes get back-filled to the first feed
    assert cfg.symbol == "EURUSD"
    assert cfg.timeframe == Timeframe.H1


def test_explicit_feeds_list_passes_through() -> None:
    feeds = [
        Subscription(symbol="EURUSD", timeframe=Timeframe.M15),
        Subscription(symbol="GBPUSD", timeframe=Timeframe.H4),
    ]
    cfg = BacktestRunConfig(feeds=feeds, **_common_run_kwargs())
    assert {(f.symbol, f.timeframe.value) for f in cfg.feed_list} == {
        ("EURUSD", "M15"),
        ("GBPUSD", "H4"),
    }


def test_no_feeds_at_all_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        BacktestRunConfig(**_common_run_kwargs())
    assert "ONE of" in str(exc.value)


def test_mixing_singular_and_plural_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        BacktestRunConfig(
            symbol="EURUSD",
            timeframe=Timeframe.M15,
            symbols=["GBPUSD"],
            timeframes=[Timeframe.H1],
            **_common_run_kwargs(),
        )
    assert "mixes feed shapes" in str(exc.value)


def test_plural_requires_both_lists() -> None:
    with pytest.raises(ValidationError):
        BacktestRunConfig(
            symbols=["EURUSD"],
            **_common_run_kwargs(),  # missing timeframes
        )


def test_duplicate_feeds_are_deduped() -> None:
    feeds = [
        Subscription(symbol="EURUSD", timeframe=Timeframe.M15),
        Subscription(symbol="EURUSD", timeframe=Timeframe.M15),
    ]
    cfg = BacktestRunConfig(feeds=feeds, **_common_run_kwargs())
    assert len(cfg.feed_list) == 1


# --- SweepRunConfig (mirrors BacktestRunConfig) -------------------------------


def test_sweep_singular_still_works() -> None:
    cfg = SweepRunConfig(symbol="EURUSD", timeframe=Timeframe.M15, **_common_sweep_kwargs())
    assert cfg.feed_list == [Subscription(symbol="EURUSD", timeframe=Timeframe.M15)]


def test_sweep_plural_resolves_to_cartesian() -> None:
    cfg = SweepRunConfig(
        symbols=["EURUSD", "GBPUSD"],
        timeframes=[Timeframe.M15],
        **_common_sweep_kwargs(),
    )
    assert len(cfg.feed_list) == 2
