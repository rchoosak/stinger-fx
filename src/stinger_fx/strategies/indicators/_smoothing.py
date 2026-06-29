"""Shared helper for Wilder-smoothed indicators.

Wilder's smoothing is a recursive EMA with ``alpha = 1/period``:

    avg_t = avg_{t-1} + (x_t - avg_{t-1}) / period

so the influence of a value ``k`` steps in the past decays geometrically as
``(1 - 1/period)**k``. Beyond roughly ``37 * period`` steps that weight drops
below a float64 ULP (``2**-53``), i.e. the leading inputs can no longer change
the result at full double precision. Truncating a long input to its last
``FACTOR * period`` elements therefore yields a value **bit-identical** to the
full pass, while doing far less work when ``period`` is small relative to the
window (the hot case: RSI/ATR-14 fed a 2000-bar history every bar).

``FACTOR = 50`` keeps a comfortable margin over the ~37 needed.

**EMA** (``alpha = 2/(period+1)``) decays ~2x faster than Wilder, so a smaller
factor is already bit-identical; ``EMA_TAIL_FACTOR`` is set conservatively and
pinned bit-identical by property tests over random series.

NB: **ADX is intentionally not capped** — its ``tr_smooth == 0 → None`` seed
guard depends on which window seeds it, so truncating the input is *not*
bit-identical for a degenerate all-flat seed window. Use a streaming indicator
if ADX ever needs the speedup.
"""

from __future__ import annotations

from collections.abc import Sequence

WILDER_TAIL_FACTOR = 50
EMA_TAIL_FACTOR = 30


def wilder_tail[T](
    seq: Sequence[T], period: int, factor: int = WILDER_TAIL_FACTOR
) -> Sequence[T]:
    """Return the trailing slice of ``seq`` long enough for a bit-identical
    smoothed result at ``period`` (the whole sequence when already short enough).
    ``factor`` widens the window for slower-decaying smoothers (see module doc)."""
    cap = period * factor
    return seq[-cap:] if len(seq) > cap else seq
