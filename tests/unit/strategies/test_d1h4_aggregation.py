"""Calendar-aware H1 → H4 / D1 aggregation.

Covers the swing-aggregation spec: anchor hour, Sunday-fragment → Monday D1,
missing-slot rejection vs scheduled breaks, Friday/DST late finalisation, and
the no-lookahead guarantee (a bucket is only emitted after a later bar).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from stinger_fx.domain import Bar, Timeframe
from stinger_fx.strategies.aggregation import (
    ForexWeekCalendar,
    MultiTimeframeAggregator,
    d1_bucket_key,
    h4_bucket_key,
)

SYMBOL = "XAUUSD"
# 2024-01-07 is a Sunday; 2024-01-08 a Monday; 2024-01-12 a Friday.
SUN = datetime(2024, 1, 7, tzinfo=UTC)
MON = datetime(2024, 1, 8, tzinfo=UTC)
TUE = datetime(2024, 1, 9, tzinfo=UTC)
FRI = datetime(2024, 1, 12, tzinfo=UTC)


def _h1(t: datetime, *, o: float = 100.0, h: float = 101.0,
        lo: float = 99.0, c: float = 100.5) -> Bar:
    return Bar(
        symbol=SYMBOL, timeframe=Timeframe.H1, time=t,
        open=o, high=h, low=lo, close=c, tick_volume=10, is_closed=True,
    )


def _feed_all(agg: MultiTimeframeAggregator, times: list[datetime]):
    h4: list[Bar] = []
    d1: list[Bar] = []
    for i, t in enumerate(times):
        res = agg.feed(_h1(t, o=100.0 + i, h=101.0 + i, lo=99.0 + i, c=100.5 + i))
        if res.h4:
            h4.append(res.h4)
        if res.d1:
            d1.append(res.d1)
    return h4, d1


# --------------------------------------------------------------------------- #
# H4 folding + OHLC + no-lookahead                                            #
# --------------------------------------------------------------------------- #

def test_h4_folds_four_slots_and_emits_on_next_bucket() -> None:
    agg = MultiTimeframeAggregator(SYMBOL, anchor_hour=0)
    slots = [MON + i * timedelta(hours=1) for i in range(4)]  # 00..03
    h4, _ = _feed_all(agg, slots)
    assert h4 == []  # bucket still open — NOT finalised by its own last slot
    # The first bar of the next bucket triggers emission (no lookahead).
    res = agg.feed(_h1(MON + timedelta(hours=4)))
    assert res.h4 is not None
    bar = res.h4
    assert bar.timeframe is Timeframe.H4
    assert bar.time == MON                       # bucket open time
    assert bar.open == 100.0                      # first slot's open
    assert bar.close == 100.5 + 3                 # 4th slot's close
    assert bar.high == 101.0 + 3 and bar.low == 99.0


def test_missing_open_slot_drops_bucket() -> None:
    agg = MultiTimeframeAggregator(SYMBOL, anchor_hour=0)
    # Skip the 02:00 slot — it's an open weekday hour, so the bucket is corrupt.
    times = [MON, MON + timedelta(hours=1), MON + timedelta(hours=3),
             MON + timedelta(hours=4)]
    h4, _ = _feed_all(agg, times)
    assert h4 == []  # 00:00 bucket dropped (02:00 missing); 04 starts new bucket


def test_scheduled_break_still_aggregates() -> None:
    # Start-of-week H4 bucket 20:00–24:00: 20/21 are weekend-closed (scheduled),
    # only 22/23 trade. Missing 20/21 must NOT drop the bucket.
    agg = MultiTimeframeAggregator(SYMBOL, anchor_hour=0)
    times = [SUN + timedelta(hours=22), SUN + timedelta(hours=23), MON]
    h4, _ = _feed_all(agg, times)
    assert len(h4) == 1
    assert h4[0].time == SUN + timedelta(hours=20)  # bucket anchored at 20:00


def test_daily_maintenance_break_still_emits_complete_d1() -> None:
    """XAU's maintenance hour shifts with US DST.

    Marking both possible UTC hours as scheduled means neither is required,
    while any bar that does arrive in one of those hours is still included.
    Without this calendar setting almost every real Dukascopy D1 was dropped
    as corrupt, so the strategy never accumulated enough daily bars to warm.
    """
    calendar = ForexWeekCalendar(
        daily_break_hours=frozenset({21, 22})
    )
    agg = MultiTimeframeAggregator(
        SYMBOL, anchor_hour=0, calendar=calendar
    )
    times = [SUN + timedelta(hours=23)]
    times += [
        MON + timedelta(hours=hour)
        for hour in range(24)
        if hour not in {21, 22}
    ]
    times.append(TUE)  # next bucket finalises Monday
    _, d1 = _feed_all(agg, times)
    assert len(d1) == 1
    assert d1[0].time == MON
    assert d1[0].tick_volume == 23 * 10


# --------------------------------------------------------------------------- #
# D1 trading-day keying                                                        #
# --------------------------------------------------------------------------- #

def test_sunday_fragment_merges_into_monday_d1() -> None:
    agg = MultiTimeframeAggregator(SYMBOL, anchor_hour=0)
    times = [SUN + timedelta(hours=22), SUN + timedelta(hours=23)]
    times += [MON + i * timedelta(hours=1) for i in range(24)]  # Mon 00..23
    times.append(TUE)  # finalises Monday's D1
    _, d1 = _feed_all(agg, times)
    # Exactly one D1, keyed Monday — never a stand-alone Sunday D1.
    assert len(d1) == 1
    assert d1[0].time == MON
    assert all(b.time != SUN for b in d1)


def test_nonzero_anchor_keeps_correct_trading_day() -> None:
    # anchor_hour = week open (22): the Mon session is [Sun22:00, Mon22:00).
    agg = MultiTimeframeAggregator(SYMBOL, anchor_hour=22)
    times = [SUN + timedelta(hours=22), SUN + timedelta(hours=23)]
    times += [MON + i * timedelta(hours=1) for i in range(22)]  # Mon 00..21
    times.append(MON + timedelta(hours=22))  # Mon 22:00 → next session, finalises
    _, d1 = _feed_all(agg, times)
    assert len(d1) == 1
    assert d1[0].time == SUN + timedelta(hours=22)  # session keyed at its open
    # Sanity: the key helpers agree.
    assert d1_bucket_key(MON + timedelta(hours=10), 22, ForexWeekCalendar()) == \
        SUN + timedelta(hours=22)
    assert d1_bucket_key(MON + timedelta(hours=22), 22, ForexWeekCalendar()) == \
        MON + timedelta(hours=22)


# --------------------------------------------------------------------------- #
# Friday / DST: no early finalisation                                         #
# --------------------------------------------------------------------------- #

def test_friday_partial_bucket_not_finalised_early() -> None:
    # Friday closes 22:00, so the 20:00–24:00 H4 only has 20/21. It must stay
    # open until the next session's bar arrives — never finalised early.
    agg = MultiTimeframeAggregator(SYMBOL, anchor_hour=0)
    # Warm a prior complete Friday bucket so the stream is realistic.
    pre = [FRI + timedelta(hours=16) + i * timedelta(hours=1) for i in range(4)]
    _feed_all(agg, pre)
    # Feeding 20:00 finalises the *prior* (16:00) complete bucket — expected.
    r1 = agg.feed(_h1(FRI + timedelta(hours=20)))
    assert r1.h4 is not None and r1.h4.time == FRI + timedelta(hours=16)
    # The partial Friday 20:00 bucket (only 20/21 trade) must NOT finalise here.
    r2 = agg.feed(_h1(FRI + timedelta(hours=21)))
    assert r2.h4 is None
    # Only the next session's bar (following Monday) can close it — no early
    # finalisation on the short Friday.
    next_mon = FRI + timedelta(days=3)  # Monday
    res = agg.feed(_h1(next_mon))
    assert res.h4 is not None
    assert res.h4.time == FRI + timedelta(hours=20)


def test_nonzero_anchor_drops_bucket_on_missing_early_slot() -> None:
    # anchor=10 with a 22:00 week open → the Monday trading day spans
    # [Sun22:00, Tue10:00), so its early slots sit far (~8h+) before the key.
    # A missing open slot there must still drop the D1 bucket (regression: the
    # expected-slot window must cover the whole merged trading day).
    agg = MultiTimeframeAggregator(SYMBOL, anchor_hour=10)
    drop = MON + timedelta(hours=2)              # an open Monday slot, ~8h < key
    end = TUE + timedelta(hours=10)              # Tue 10:00 finalises Monday
    t, d1 = SUN + timedelta(hours=22), []
    while t < end:
        if t != drop:
            res = agg.feed(_h1(t))
            if res.d1:
                d1.append(res.d1)
        t += timedelta(hours=1)
    res = agg.feed(_h1(end))
    if res.d1:
        d1.append(res.d1)
    assert d1 == [], "missing early-day slot must drop the merged D1 bucket"


def test_out_of_order_and_duplicate_h1_ignored() -> None:
    agg = MultiTimeframeAggregator(SYMBOL, anchor_hour=0)
    agg.feed(_h1(MON + timedelta(hours=2)))
    # Earlier + duplicate timestamps must not roll or corrupt buckets.
    assert agg.feed(_h1(MON + timedelta(hours=1))).h4 is None
    assert agg.feed(_h1(MON + timedelta(hours=2))).h4 is None


def test_h4_key_alignment() -> None:
    assert h4_bucket_key(MON + timedelta(hours=5), 0) == MON + timedelta(hours=4)
    assert h4_bucket_key(MON + timedelta(hours=3), 0) == MON
    # anchor 1 → boundaries at 01,05,09,...
    assert h4_bucket_key(MON + timedelta(hours=2), 1) == MON + timedelta(hours=1)
