"""Engine-level pre-trade filter.

`evaluate_trading_filter` returns a human-readable block reason when an order
should be refused for a market-condition reason — wide spread, outside the
trading session, near the daily rollover, or inside a news blackout — or
``None`` when the order may proceed. Pure and time-source agnostic: the caller
passes ``now`` (``signal.time`` — wall-clock live, sim time in backtests) and the
current ``spread_points``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from stinger_fx.config.models import TradingFilterConfig


def _hour_in_session(hour: int, start: int, end: int) -> bool:
    """True if `hour` is within [start, end) UTC hours, wrapping past midnight
    when start > end (e.g. 22→6)."""
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def _minutes_to_rollover(now: datetime, hour: int) -> float:
    """Smallest distance in minutes from `now` to the `hour:00` rollover,
    considering the rollover on the previous, current, and next day so the
    window wraps correctly around midnight."""
    base = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    return min(
        abs((now - (base + timedelta(days=d))).total_seconds()) for d in (-1, 0, 1)
    ) / 60.0


def evaluate_trading_filter(
    cfg: TradingFilterConfig,
    *,
    now: datetime,
    spread_points: int | None,
) -> str | None:
    """Return a block reason, or None when the order may proceed."""
    if not cfg.enabled:
        return None

    if (
        cfg.max_spread_points > 0
        and spread_points is not None
        and spread_points > cfg.max_spread_points
    ):
        return f"spread {spread_points} > max {cfg.max_spread_points} points"

    start, end = cfg.session_start_hour_utc, cfg.session_end_hour_utc
    if start is not None and end is not None and not _hour_in_session(now.hour, start, end):
        return (
            f"outside session {start:02d}:00-{end:02d}:00 UTC (now {now.hour:02d}:00)"
        )

    if cfg.block_rollover:
        mins = _minutes_to_rollover(now, cfg.rollover_hour_utc)
        if mins <= cfg.rollover_block_minutes:
            return (
                f"within {cfg.rollover_block_minutes}min of {cfg.rollover_hour_utc:02d}:00 "
                f"rollover ({mins:.1f}min away)"
            )

    for window in cfg.news_blackouts:
        if window.start <= now < window.end:
            return f"news blackout {window.start.isoformat()}-{window.end.isoformat()}"

    return None
