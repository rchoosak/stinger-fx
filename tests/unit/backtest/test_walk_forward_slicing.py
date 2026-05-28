"""slice_folds — splits a date range into rolling / expanding folds."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta

import pytest

from stinger_fx.backtest.walk_forward import slice_folds


def _ts(days_from_start: int) -> datetime:
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=days_from_start)


def test_rolling_folds_split_evenly() -> None:
    """5 folds over 50 days → each fold 10 days wide; rolling = each fold's
    in-sample is the first 70% of its own slice."""
    folds = slice_folds(
        _ts(0), _ts(50), n_folds=5, in_sample_pct=0.7, scheme="rolling"
    )
    assert len(folds) == 5
    # Each fold is 10 days; in-sample = first 7 days; OOS = last 3 days
    for i, f in enumerate(folds):
        assert f.index == i
        # in_sample starts at this fold's window start (rolling)
        expected_win_start = _ts(i * 10)
        assert f.in_sample[0] == expected_win_start
        # in_sample ends at 7 days into this fold
        expected_is_end = _ts(i * 10 + 7)
        assert abs((f.in_sample[1] - expected_is_end).total_seconds()) < 1
        # OOS = [is_end, win_end]
        expected_win_end = _ts((i + 1) * 10)
        assert f.out_of_sample[0] == f.in_sample[1]
        assert abs((f.out_of_sample[1] - expected_win_end).total_seconds()) < 1


def test_expanding_folds_in_sample_grows_from_start() -> None:
    """In expanding scheme, every fold's in-sample STARTS at the global
    start; only the end advances."""
    folds = slice_folds(
        _ts(0), _ts(40), n_folds=4, in_sample_pct=0.5, scheme="expanding"
    )
    for f in folds:
        assert f.in_sample[0] == _ts(0)  # every fold starts at global start
    # in-sample END moves forward each fold
    is_ends = [f.in_sample[1] for f in folds]
    assert is_ends == sorted(is_ends)  # monotonically increasing
    assert is_ends[0] != is_ends[-1]   # not all identical


def test_oos_window_is_contiguous_with_in_sample() -> None:
    """OOS starts exactly where in-sample ends — no gaps, no overlap."""
    folds = slice_folds(_ts(0), _ts(100), n_folds=10, in_sample_pct=0.6)
    for f in folds:
        assert f.in_sample[1] == f.out_of_sample[0]


def test_folds_cover_full_range_back_to_back() -> None:
    """The end of fold N's OOS must equal the start of fold N+1's window
    (no gaps between adjacent folds in rolling scheme)."""
    folds = slice_folds(_ts(0), _ts(40), n_folds=4, scheme="rolling", in_sample_pct=0.5)
    for prev, curr in itertools.pairwise(folds):
        # Curr's in_sample[0] (window start in rolling) must equal prev's oos end
        assert curr.in_sample[0] == prev.out_of_sample[1]


def test_slice_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="n_folds"):
        slice_folds(_ts(0), _ts(10), n_folds=0)
    with pytest.raises(ValueError, match="in_sample_pct"):
        slice_folds(_ts(0), _ts(10), n_folds=3, in_sample_pct=1.0)
    with pytest.raises(ValueError, match="in_sample_pct"):
        slice_folds(_ts(0), _ts(10), n_folds=3, in_sample_pct=0.0)
    with pytest.raises(ValueError, match="scheme"):
        slice_folds(_ts(0), _ts(10), n_folds=3, scheme="random")
    with pytest.raises(ValueError, match="end"):
        slice_folds(_ts(10), _ts(0), n_folds=3)
