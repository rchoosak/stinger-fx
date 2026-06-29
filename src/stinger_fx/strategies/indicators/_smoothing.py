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

Other smoothers reuse this with their own factor:
- **EMA** (``alpha = 2/(period+1)``) decays ~2x faster than Wilder, so a smaller
  factor is already bit-identical; ``EMA_TAIL_FACTOR`` is set conservatively.
- **ADX** cascades two Wilder stages (DI then DX), so the seed influence decays
  slower; ``ADX_TAIL_FACTOR`` is set higher. Both are pinned bit-identical by
  property tests over random series.
"""

from __future__ import annotations

from collections.abc import Sequence

WILDER_TAIL_FACTOR = 50
EMA_TAIL_FACTOR = 30
ADX_TAIL_FACTOR = 80


def wilder_tail[T](
    seq: Sequence[T], period: int, factor: int = WILDER_TAIL_FACTOR
) -> Sequence[T]:
    """Return the trailing slice of ``seq`` long enough for a bit-identical
    smoothed result at ``period`` (the whole sequence when already short enough).
    ``factor`` widens the window for slower-decaying smoothers (see module doc)."""
    cap = period * factor
    return seq[-cap:] if len(seq) > cap else seq
