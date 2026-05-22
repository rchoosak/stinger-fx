"""SQLite engine + session factory.

A single engine instance per process is enough for our workloads. We use the
synchronous SQLModel API and call it from asyncio via `run_in_executor` only
where contention would matter — most engine writes are infrequent.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool


class SqliteStore:
    def __init__(self, db_path: Path, *, echo: bool = False) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so we can call from the executor.
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=echo,
            connect_args={"check_same_thread": False},
        )

    def create_all(self) -> None:
        """Create tables — call once at startup or via `stinger-fx db migrate`."""
        # SQLModel registers tables via metaclass; importing schemas suffices.
        from stinger_fx.data import schemas  # noqa: F401 — registers tables

        SQLModel.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with Session(self.engine) as s:
            yield s


def in_memory_store() -> SqliteStore:
    """Helper used by tests."""
    from stinger_fx.data import schemas  # noqa: F401

    store = SqliteStore.__new__(SqliteStore)
    store.db_path = Path(":memory:")
    store.engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(store.engine)
    return store
