"""Calendar-aware H1 → H4 / D1 bar aggregation.

The engine's :class:`~stinger_fx.brokers.bar_aggregator.BarAggregator` folds
*ticks* into bars on naive UTC epoch-floor boundaries. That's right for
intraday timeframes but wrong for swing timeframes on a 24×5 market:

  * a **daily** bar must be anchored at a configurable hour (broker midnight),
    not always 00:00 UTC;
  * the few hours the market trades on **Sunday evening** belong to *Monday's*
    daily bar — they must not form a stand-alone "Sunday" D1;
  * a bucket with an **unexpected** missing H1 slot (calendar says the market
    was open but no bar arrived) is corrupt and must be dropped, while a
    **scheduled** break (weekend / session close) is not data loss;
  * a bucket must never be finalised early — on a short Friday or a DST shift
    the session can end before the nominal boundary, so a bucket is only
    confirmed complete once a bar from the *next* bucket is seen.

This module folds already-*completed* H1 bars into completed H4 and D1 bars
honouring all of the above. It is pure and deterministic: it only ever emits a
bucket after observing a later bar, so it can never look ahead. Live and
backtest feed the identical H1 stream through it, so both see identical H4/D1.

The session calendar is pluggable (:class:`SessionCalendar`); the default
:class:`ForexWeekCalendar` models the standard Sun-open / Fri-close FX week with
configurable hours, so the aggregator is not tied to one data provider.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import NamedTuple, Protocol, runtime_checkable

from stinger_fx.domain import Bar, Timeframe

_HOUR = timedelta(hours=1)


def anchored_day_start(t: datetime, anchor_hour: int) -> datetime:
    """The most recent ``anchor_hour:00`` boundary at or before ``t`` (UTC)."""
    d = t.astimezone(UTC).replace(
        hour=anchor_hour, minute=0, second=0, microsecond=0
    )
    if t < d:
        d -= timedelta(days=1)
    return d


@runtime_checkable
class SessionCalendar(Protocol):
    """Decides which H1 slots the market is open for, and where the week opens.

    ``slot_start`` is always an H1 *open* time (UTC, minute 0)."""

    def is_open(self, slot_start: datetime) -> bool: ...

    def week_open_before(self, t: datetime) -> datetime: ...


@dataclass(frozen=True)
class ForexWeekCalendar:
    """Standard 24×5 FX week: continuously open from Sunday ``week_open_hour``
    UTC to Friday ``week_close_hour`` UTC. Hours are configurable so the same
    calendar adapts to different broker server offsets / DST conventions; a
    different provider can supply its own :class:`SessionCalendar`.

    ``daily_break_hours`` lists UTC hours the instrument is closed *every*
    trading day — e.g. gold's ~21:00 maintenance break. Those hours are then
    treated as scheduled (not data loss), so a day missing only its break hour
    still aggregates instead of being dropped. Default empty = no daily break."""

    week_open_hour: int = 22   # Sunday open (UTC) — typical FX
    week_close_hour: int = 22  # Friday close (UTC)
    daily_break_hours: frozenset[int] = frozenset()
    # weekday() values: Mon=0 … Sat=5, Sun=6
    _SUNDAY: int = 6
    _FRIDAY: int = 4
    _SATURDAY: int = 5

    def is_open(self, slot_start: datetime) -> bool:
        if slot_start.hour in self.daily_break_hours:
            return False
        wd = slot_start.weekday()
        if wd == self._SATURDAY:
            return False
        if wd == self._SUNDAY:
            return slot_start.hour >= self.week_open_hour
        if wd == self._FRIDAY:
            return slot_start.hour < self.week_close_hour
        return True  # Mon–Thu fully open

    def week_open_before(self, t: datetime) -> datetime:
        """Most recent Sunday ``week_open_hour:00`` at or before ``t``."""
        days_since_sunday = (t.weekday() - self._SUNDAY) % 7
        candidate = (t - timedelta(days=days_since_sunday)).replace(
            hour=self.week_open_hour, minute=0, second=0, microsecond=0
        )
        if candidate > t:
            candidate -= timedelta(days=7)
        return candidate


@dataclass(frozen=True)
class AlwaysOpenCalendar:
    """A 24×7 calendar — every hour trades, no weekend, no Sunday merge. Handy
    for synthetic/continuous data and tests; not for real FX data."""

    def is_open(self, slot_start: datetime) -> bool:
        return True

    def week_open_before(self, t: datetime) -> datetime:
        # Far in the past so the Sunday-fragment merge never triggers: every
        # anchored day start is >= this, so D1 buckets are clean 24h windows.
        return (t - timedelta(days=400)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )


def h4_bucket_key(t: datetime, anchor_hour: int) -> datetime:
    """Open time of the 4-hour bucket containing ``t`` (anchored at
    ``anchor_hour``). Boundaries are ``anchor_hour + 4k`` hours."""
    a = anchored_day_start(t, anchor_hour)
    hours_since = int((t - a) / _HOUR)
    return a + timedelta(hours=4 * (hours_since // 4))


def d1_bucket_key(
    t: datetime, anchor_hour: int, calendar: SessionCalendar
) -> datetime:
    """Trading-day key for ``t``: the anchored day start it belongs to, with
    the Sunday-evening open fragment merged forward into Monday.

    The merge only fires when the anchored boundary falls inside the weekend
    gap (``anchor_hour`` earlier than the week open), so a non-midnight anchor
    does not misattribute Monday's bars to the wrong day."""
    b = anchored_day_start(t, anchor_hour)
    week_open = calendar.week_open_before(t)
    if b < week_open:
        # ``b`` is a weekend stub before the market opened; the fragment
        # [week_open, next anchor) belongs to the following trading day.
        nb = anchored_day_start(week_open, anchor_hour)
        if nb < week_open:
            nb += timedelta(days=1)
        return nb
    return b


def _expected_open_slots(
    key: datetime,
    span_before: timedelta,
    span_after: timedelta,
    key_fn: Callable[[datetime], datetime],
    calendar: SessionCalendar,
) -> list[datetime]:
    """Open H1 slots that should belong to bucket ``key`` — every hourly slot
    in [key-span_before, key+span_after) whose own key is ``key`` and that the
    calendar marks open."""
    out: list[datetime] = []
    t = key - span_before
    end = key + span_after
    while t < end:
        if calendar.is_open(t) and key_fn(t) == key:
            out.append(t)
        t += _HOUR
    return out


class _Bucketer:
    """Accumulates H1 bars into one timeframe's buckets, emitting a completed
    Bar when a later bucket's first bar arrives (never earlier — no lookahead).

    A bucket is emitted only when every *expected open* slot was received;
    otherwise it is dropped (an unexpected gap means the bucket is corrupt).
    Scheduled-break slots are not expected, so they don't block emission.
    """

    def __init__(
        self,
        *,
        symbol: str,
        tf: Timeframe,
        key_fn: Callable[[datetime], datetime],
        expected_fn: Callable[[datetime], list[datetime]],
    ) -> None:
        self._symbol = symbol
        self._tf = tf
        self._key_fn = key_fn
        self._expected_fn = expected_fn
        self._key: datetime | None = None
        self._bars: list[Bar] = []

    def feed(self, bar: Bar) -> Bar | None:
        key = self._key_fn(bar.time)
        if self._key is None:
            self._key = key
            self._bars = [bar]
            return None
        if key == self._key:
            self._bars.append(bar)
            return None
        # A bar from a later bucket — finalise the current one, then start new.
        done = self._finalize()
        self._key = key
        self._bars = [bar]
        return done

    def _finalize(self) -> Bar | None:
        assert self._key is not None
        expected = self._expected_fn(self._key)
        received = {b.time for b in self._bars}
        if not expected or not set(expected).issubset(received):
            # No open slots, or an expected slot is missing → corrupt; drop.
            return None
        ordered = sorted(self._bars, key=lambda b: b.time)
        return Bar(
            symbol=self._symbol,
            timeframe=self._tf,
            time=self._key,
            open=ordered[0].open,
            high=max(b.high for b in ordered),
            low=min(b.low for b in ordered),
            close=ordered[-1].close,
            tick_volume=sum(b.tick_volume for b in ordered),
            is_closed=True,
        )


class AggregationResult(NamedTuple):
    """Newly-completed higher-timeframe bars produced by one H1 ``feed`` call.
    Either field is ``None`` when that timeframe's bucket did not roll."""

    h4: Bar | None
    d1: Bar | None


class MultiTimeframeAggregator:
    """Folds a per-symbol stream of completed H1 bars into completed H4 and D1
    bars, honouring the anchor hour, the Sunday→Monday merge, missing-slot
    rejection, and emit-on-next-bucket finalisation.

    Feed completed H1 bars in time order via :meth:`feed`; it returns any
    higher-timeframe bars that *closed* as a result. The in-progress bucket is
    never emitted (matching live), so there is no end-of-data flush.
    """

    def __init__(
        self,
        symbol: str,
        *,
        anchor_hour: int = 0,
        calendar: SessionCalendar | None = None,
    ) -> None:
        if not (0 <= anchor_hour <= 23):
            raise ValueError("anchor_hour must be in [0, 23]")
        self.symbol = symbol
        self.anchor_hour = anchor_hour
        self.calendar: SessionCalendar = calendar or ForexWeekCalendar()
        self._last_time: datetime | None = None

        cal = self.calendar
        ah = anchor_hour

        def _h4_key(t: datetime) -> datetime:
            return h4_bucket_key(t, ah)

        def _d1_key(t: datetime) -> datetime:
            return d1_bucket_key(t, ah, cal)

        self._h4 = _Bucketer(
            symbol=symbol,
            tf=Timeframe.H4,
            key_fn=_h4_key,
            expected_fn=lambda k: _expected_open_slots(
                k, timedelta(0), timedelta(hours=4), _h4_key, cal
            ),
        )
        self._d1 = _Bucketer(
            symbol=symbol,
            tf=Timeframe.D1,
            key_fn=_d1_key,
            # A trading day spans < 48h (a normal 24h day plus, for the first
            # day of the week, the Sunday-open fragment which can sit up to ~24h
            # before the key when the anchor is far from the week open). Enumerate
            # +/-26h so no in-day open slot is missed regardless of anchor_hour.
            expected_fn=lambda k: _expected_open_slots(
                k, timedelta(hours=26), timedelta(hours=26), _d1_key, cal
            ),
        )

    def feed(self, bar: Bar) -> AggregationResult:
        if bar.symbol != self.symbol or bar.timeframe is not Timeframe.H1:
            return AggregationResult(None, None)
        if not bar.is_closed:
            return AggregationResult(None, None)
        if self._last_time is not None and bar.time <= self._last_time:
            # Out-of-order / duplicate H1 — ignore (never look back).
            return AggregationResult(None, None)
        self._last_time = bar.time
        return AggregationResult(h4=self._h4.feed(bar), d1=self._d1.feed(bar))


def fold_h1(
    bars: Sequence[Bar],
    *,
    anchor_hour: int = 0,
    calendar: SessionCalendar | None = None,
) -> tuple[list[Bar], list[Bar]]:
    """Convenience: fold a finished sequence of H1 bars into (h4_bars, d1_bars).
    Mirrors live semantics — the last in-progress bucket is not emitted."""
    agg = MultiTimeframeAggregator(
        bars[0].symbol if bars else "",
        anchor_hour=anchor_hour,
        calendar=calendar,
    )
    h4: list[Bar] = []
    d1: list[Bar] = []
    for b in bars:
        res = agg.feed(b)
        if res.h4 is not None:
            h4.append(res.h4)
        if res.d1 is not None:
            d1.append(res.d1)
    return h4, d1
