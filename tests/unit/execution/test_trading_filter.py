"""evaluate_trading_filter — spread / session / rollover / news gating."""

from __future__ import annotations

from datetime import UTC, datetime

from stinger_fx.config.models import NewsBlackout, TradingFilterConfig
from stinger_fx.execution.trading_filter import evaluate_trading_filter

NOON = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)


def _cfg(**over) -> TradingFilterConfig:
    return TradingFilterConfig(enabled=True, **over)


def test_disabled_always_allows() -> None:
    cfg = TradingFilterConfig(enabled=False, max_spread_points=1)
    assert evaluate_trading_filter(cfg, now=NOON, spread_points=9999) is None


def test_all_off_allows() -> None:
    assert evaluate_trading_filter(_cfg(), now=NOON, spread_points=50) is None


# --- spread ---------------------------------------------------------------


def test_spread_over_cap_blocks() -> None:
    cfg = _cfg(max_spread_points=20)
    assert "spread" in (evaluate_trading_filter(cfg, now=NOON, spread_points=21) or "")


def test_spread_at_or_under_cap_allows() -> None:
    cfg = _cfg(max_spread_points=20)
    assert evaluate_trading_filter(cfg, now=NOON, spread_points=20) is None


def test_spread_unknown_skips_check() -> None:
    cfg = _cfg(max_spread_points=1)
    assert evaluate_trading_filter(cfg, now=NOON, spread_points=None) is None


# --- session window -------------------------------------------------------


def test_inside_session_allows() -> None:
    cfg = _cfg(session_start_hour_utc=8, session_end_hour_utc=16)
    assert evaluate_trading_filter(cfg, now=NOON, spread_points=None) is None


def test_outside_session_blocks() -> None:
    cfg = _cfg(session_start_hour_utc=8, session_end_hour_utc=16)
    night = NOON.replace(hour=18)
    assert "session" in (evaluate_trading_filter(cfg, now=night, spread_points=None) or "")


def test_session_wrap_around_midnight() -> None:
    cfg = _cfg(session_start_hour_utc=22, session_end_hour_utc=6)
    assert evaluate_trading_filter(cfg, now=NOON.replace(hour=23), spread_points=None) is None
    assert evaluate_trading_filter(cfg, now=NOON.replace(hour=3), spread_points=None) is None
    assert evaluate_trading_filter(cfg, now=NOON.replace(hour=10), spread_points=None) is not None


# --- rollover -------------------------------------------------------------


def test_inside_rollover_window_blocks() -> None:
    cfg = _cfg(block_rollover=True, rollover_hour_utc=21, rollover_block_minutes=5)
    at = datetime(2024, 1, 1, 21, 3, tzinfo=UTC)
    assert "rollover" in (evaluate_trading_filter(cfg, now=at, spread_points=None) or "")


def test_outside_rollover_window_allows() -> None:
    cfg = _cfg(block_rollover=True, rollover_hour_utc=21, rollover_block_minutes=5)
    at = datetime(2024, 1, 1, 21, 10, tzinfo=UTC)
    assert evaluate_trading_filter(cfg, now=at, spread_points=None) is None


def test_rollover_window_wraps_midnight() -> None:
    cfg = _cfg(block_rollover=True, rollover_hour_utc=0, rollover_block_minutes=5)
    at = datetime(2024, 1, 1, 23, 58, tzinfo=UTC)  # 2 min before next-day 00:00
    assert evaluate_trading_filter(cfg, now=at, spread_points=None) is not None


# --- news blackout --------------------------------------------------------


def test_news_blackout_blocks_inside_half_open_window() -> None:
    w = NewsBlackout(
        start=datetime(2024, 1, 1, 11, 55, tzinfo=UTC),
        end=datetime(2024, 1, 1, 12, 5, tzinfo=UTC),
    )
    cfg = _cfg(news_blackouts=[w])
    assert "news" in (evaluate_trading_filter(cfg, now=NOON, spread_points=None) or "")
    # end is exclusive
    assert evaluate_trading_filter(cfg, now=w.end, spread_points=None) is None


def test_first_matching_reason_wins() -> None:
    # spread checked before session — a wide spread reports spread, not session.
    cfg = _cfg(
        max_spread_points=10,
        session_start_hour_utc=8,
        session_end_hour_utc=16,
    )
    night_wide = NOON.replace(hour=20)
    assert "spread" in (evaluate_trading_filter(cfg, now=night_wide, spread_points=99) or "")
