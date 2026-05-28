"""In-browser strategy code editor — safety + endpoint behaviour."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stinger_fx.brokers import BrokerPool
from stinger_fx.brokers.base import BaseBroker
from stinger_fx.core import AsyncEventBus
from stinger_fx.domain import (
    AccountInfo,
    AccountSnapshot,
    Order,
    OrderRequest,
    OrderResult,
    OrderStatus,
    Position,
    SymbolInfo,
)
from stinger_fx.ui.handle import EngineHandle
from stinger_fx.ui.web import create_app
from stinger_fx.ui.web.strategy_editor import (
    is_safe_name,
    resolve_path,
    scaffold_source,
    validate_source,
)


class _StubBroker(BaseBroker):
    name = "stub"
    async def connect(self): ...
    async def disconnect(self): ...
    async def is_connected(self): return True
    async def get_account_info(self):
        return AccountInfo(account_id="x", broker="Demo", server="D",
                           currency="USD", leverage=100)
    async def get_account_snapshot(self):
        return AccountSnapshot(account_id="x", time=datetime.now(UTC),
                               balance=10_000, equity=10_000, margin=0, free_margin=10_000)
    async def get_symbol_info(self, symbol):
        return SymbolInfo(symbol="EURUSD", digits=5, point=0.00001,
                          contract_size=100_000, volume_min=0.01, volume_max=100,
                          volume_step=0.01, currency_base="EUR",
                          currency_profit="USD", currency_margin="USD")
    async def list_symbols(self): return ["EURUSD"]
    async def subscribe_ticks(self, symbol): ...
    async def subscribe_bars(self, symbol, tf): ...
    async def unsubscribe(self, symbol, tf=None): ...
    async def get_history_bars(self, *a, **kw):
        from stinger_fx.data.parquet_store import BAR_SCHEMA
        return BAR_SCHEMA.empty_table()
    async def get_history_ticks(self, *a, **kw):
        from stinger_fx.data.parquet_store import TICK_SCHEMA
        return TICK_SCHEMA.empty_table()
    async def place_order(self, req: OrderRequest) -> OrderResult:
        return OrderResult(ok=False, status=OrderStatus.REJECTED)
    async def modify_order(self, ticket, **kw): raise NotImplementedError
    async def close_position(self, ticket, volume=None): raise NotImplementedError
    async def cancel_order(self, ticket): raise NotImplementedError
    async def get_positions(self) -> list[Position]: return []
    async def get_open_orders(self) -> list[Order]: return []


@pytest.fixture
def client_with_editor(tmp_path: Path):
    user_dir = tmp_path / "user_strategies"
    bus = AsyncEventBus()
    broker = _StubBroker(bus)
    handle = EngineHandle(bus=bus, brokers=BrokerPool([("default", broker)]), runners={})
    app = create_app(handle, user_strategies_dir=user_dir)
    return TestClient(app), user_dir


# --- Unit tests for the helpers --------------------------------------------


def test_is_safe_name_accepts_valid() -> None:
    assert is_safe_name("my_strategy")
    assert is_safe_name("a")
    assert is_safe_name("ma_v2")
    assert is_safe_name("a1b2c3")


def test_is_safe_name_rejects_dangerous() -> None:
    assert not is_safe_name("")
    assert not is_safe_name("../etc/passwd")
    assert not is_safe_name("foo/bar")
    assert not is_safe_name("foo.bar")
    assert not is_safe_name("Foo")     # capitals
    assert not is_safe_name("1foo")    # leading digit
    assert not is_safe_name(".hidden")
    assert not is_safe_name("foo-bar") # dash


def test_resolve_path_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        resolve_path(tmp_path, "../escape")


def test_resolve_path_returns_under_root(tmp_path: Path) -> None:
    p = resolve_path(tmp_path, "my_strategy")
    assert p.parent == tmp_path.resolve()
    assert p.name == "my_strategy.py"


def test_validate_source_accepts_valid_python() -> None:
    assert validate_source("x = 1 + 2") is None
    assert validate_source("def foo():\n    return 42\n") is None


def test_validate_source_reports_syntax_error() -> None:
    err = validate_source("def broken(:")
    assert err is not None
    assert err.line is not None


def test_scaffold_generates_parseable_module() -> None:
    src = scaffold_source("my_test_strat")
    assert "class MyTestStrat" in src
    assert validate_source(src) is None


def test_scaffold_rejects_bad_name() -> None:
    with pytest.raises(ValueError):
        scaffold_source("../escape")


# --- Endpoint tests --------------------------------------------------------


def test_editor_index_renders(client_with_editor) -> None:
    client, _ = client_with_editor
    r = client.get("/editor")
    assert r.status_code == 200
    assert "Strategy editor" in r.text


def test_editor_index_lists_existing_files(client_with_editor) -> None:
    client, user_dir = client_with_editor
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "alpha.py").write_text("x = 1\n")
    (user_dir / "beta.py").write_text("x = 2\n")
    r = client.get("/editor")
    assert r.status_code == 200
    assert "alpha" in r.text
    assert "beta" in r.text


def test_editor_get_source_returns_file_content(client_with_editor) -> None:
    client, user_dir = client_with_editor
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "gamma.py").write_text("# hello\nx = 42\n")
    r = client.get("/editor/gamma/source")
    assert r.status_code == 200
    assert "hello" in r.text
    assert "x = 42" in r.text


def test_editor_view_404_for_missing(client_with_editor) -> None:
    client, _ = client_with_editor
    r = client.get("/editor/no_such")
    assert r.status_code == 404


def test_editor_put_valid_source_writes_file(client_with_editor) -> None:
    client, user_dir = client_with_editor
    user_dir.mkdir(parents=True, exist_ok=True)
    r = client.put(
        "/editor/saved/source",
        content=b"x = 1 + 2\n",
        headers={"Content-Type": "text/plain"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    written = (user_dir / "saved.py").read_text()
    assert written == "x = 1 + 2\n"


def test_editor_put_invalid_python_rejected(client_with_editor) -> None:
    """Syntax error → 400 with structured error info, no file written."""
    client, user_dir = client_with_editor
    r = client.put(
        "/editor/bad/source",
        content=b"def broken(:\n",
        headers={"Content-Type": "text/plain"},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["ok"] is False
    assert body["line"] is not None
    assert not (user_dir / "bad.py").exists()


def test_editor_put_path_traversal_rejected(client_with_editor) -> None:
    """A traversal-style name returns 400 (caught by is_safe_name)."""
    client, _ = client_with_editor
    r = client.put(
        "/editor/..%2Fescape/source",
        content=b"x = 1",
        headers={"Content-Type": "text/plain"},
    )
    assert r.status_code in (400, 404)


def test_editor_new_creates_scaffold(client_with_editor) -> None:
    client, user_dir = client_with_editor
    r = client.post("/editor/new", data={"name": "fresh_strat"})
    assert r.status_code == 200
    assert (user_dir / "fresh_strat.py").exists()
    written = (user_dir / "fresh_strat.py").read_text()
    assert "class FreshStrat" in written
    # Round-trip: AST-parses
    assert validate_source(written) is None


def test_editor_new_rejects_duplicate(client_with_editor) -> None:
    client, user_dir = client_with_editor
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "existing.py").write_text("x = 1\n")
    r = client.post("/editor/new", data={"name": "existing"})
    assert r.status_code == 409


def test_editor_new_rejects_invalid_name(client_with_editor) -> None:
    client, _ = client_with_editor
    r = client.post("/editor/new", data={"name": "Bad-Name"})
    assert r.status_code == 400


def test_editor_disabled_when_user_dir_not_configured(tmp_path: Path) -> None:
    """Without user_strategies_dir, every editor route returns 503."""
    bus = AsyncEventBus()
    broker = _StubBroker(bus)
    handle = EngineHandle(bus=bus, brokers=BrokerPool([("default", broker)]), runners={})
    app = create_app(handle)  # no user_strategies_dir
    client = TestClient(app)
    r = client.get("/editor")
    assert r.status_code == 503
    r = client.post("/editor/new", data={"name": "any"})
    assert r.status_code == 503
