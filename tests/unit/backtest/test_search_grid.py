"""GridSearch — exhaustive cartesian enumeration via SearchStrategy Protocol."""

from __future__ import annotations

from stinger_fx.backtest.search import GridSearch, SearchStrategy


def test_grid_yields_all_cells_then_none() -> None:
    grid: dict[str, list] = {"a": [1, 2, 3], "b": ["x", "y"]}
    gs = GridSearch(grid)
    assert gs.total_trials == 6
    seen: list[dict] = []
    while True:
        s = gs.suggest()
        if s is None:
            break
        seen.append(s)
    # 3 × 2 = 6 unique cells
    assert len(seen) == 6
    # Set of tuples to verify the cartesian product
    pairs = {(d["a"], d["b"]) for d in seen}
    assert pairs == {(1, "x"), (1, "y"), (2, "x"), (2, "y"), (3, "x"), (3, "y")}


def test_grid_empty_yields_no_cells() -> None:
    gs = GridSearch({})
    assert gs.total_trials == 0
    assert gs.suggest() is None


def test_grid_satisfies_protocol() -> None:
    """GridSearch must satisfy the SearchStrategy Protocol via isinstance check."""
    gs = GridSearch({"x": [1]})
    assert isinstance(gs, SearchStrategy)


def test_grid_report_is_noop() -> None:
    """Grid is non-adaptive — report() must accept any score without raising."""
    gs = GridSearch({"a": [1, 2]})
    s1 = gs.suggest()
    assert s1 is not None
    gs.report(s1, 100.0)  # no error
    s2 = gs.suggest()
    assert s2 is not None
    gs.report(s2, -50.0)
    # Sequence didn't change due to reports
    assert s1 != s2


def test_grid_deterministic_order() -> None:
    """Same grid produces the same suggestion sequence on every construction."""
    grid = {"fast": [5, 10, 15], "slow": [20, 30]}
    seqs = []
    for _ in range(2):
        gs = GridSearch(grid)
        run = []
        while (s := gs.suggest()) is not None:
            run.append(s)
        seqs.append(run)
    assert seqs[0] == seqs[1]
