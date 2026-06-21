"""NewsBlackout + TradingFilterConfig validation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from stinger_fx.config.models import NewsBlackout, TradingFilterConfig


def test_news_blackout_requires_tz_aware() -> None:
    # Naive datetimes would raise TypeError at runtime (naive vs aware
    # signal.time) inside the trading filter — reject them at config load.
    with pytest.raises(ValidationError, match="timezone"):
        NewsBlackout(
            start=datetime(2024, 1, 1, 12, 0),  # naive
            end=datetime(2024, 1, 1, 13, 0, tzinfo=UTC),
        )
    with pytest.raises(ValidationError, match="timezone"):
        NewsBlackout(
            start=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            end=datetime(2024, 1, 1, 13, 0),  # naive
        )


def test_news_blackout_accepts_aware_and_orders() -> None:
    w = NewsBlackout(
        start=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        end=datetime(2024, 1, 1, 13, 0, tzinfo=UTC),
    )
    assert w.start < w.end
    with pytest.raises(ValidationError, match="end must be after start"):
        NewsBlackout(
            start=datetime(2024, 1, 1, 13, 0, tzinfo=UTC),
            end=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        )


def test_trading_filter_parses_news_blackouts_from_iso_strings() -> None:
    cfg = TradingFilterConfig(
        enabled=True,
        news_blackouts=[
            {"start": "2024-01-01T12:00:00Z", "end": "2024-01-01T12:30:00Z"}
        ],
    )
    assert cfg.news_blackouts[0].start.tzinfo is not None


def test_session_hours_must_be_set_together() -> None:
    with pytest.raises(ValidationError, match="together"):
        TradingFilterConfig(enabled=True, session_start_hour_utc=8)  # missing end
    with pytest.raises(ValidationError, match="together"):
        TradingFilterConfig(enabled=True, session_end_hour_utc=16)  # missing start
