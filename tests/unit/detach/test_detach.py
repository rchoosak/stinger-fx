"""detach.py — pidfile lifecycle + liveness + status (no real engine spawn)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from stinger_fx.detach import (
    DetachError,
    detach_status,
    detach_stop,
    pidfile_path,
    process_is_alive,
    read_pidfile,
    remove_pidfile,
    write_pidfile,
)


def test_pidfile_round_trip(tmp_path: Path) -> None:
    write_pidfile(tmp_path, pid=12345, web_url="http://127.0.0.1:8765")
    info = read_pidfile(tmp_path)
    assert info is not None
    assert info["pid"] == 12345
    assert info["web_url"] == "http://127.0.0.1:8765"
    assert "started" in info


def test_pidfile_missing_returns_none(tmp_path: Path) -> None:
    assert read_pidfile(tmp_path) is None


def test_pidfile_remove_is_noop_when_absent(tmp_path: Path) -> None:
    remove_pidfile(tmp_path)  # must not raise
    write_pidfile(tmp_path, pid=1, web_url="x")
    assert pidfile_path(tmp_path).exists()
    remove_pidfile(tmp_path)
    assert not pidfile_path(tmp_path).exists()


def test_pidfile_corrupt_returns_none(tmp_path: Path) -> None:
    pidfile_path(tmp_path).write_text("{not json")
    assert read_pidfile(tmp_path) is None


def test_process_is_alive_negative_pid_is_dead() -> None:
    assert process_is_alive(0) is False
    assert process_is_alive(-1) is False


def test_process_is_alive_self_is_true() -> None:
    assert process_is_alive(os.getpid()) is True


def test_process_is_alive_known_dead_pid_is_false() -> None:
    # PID 1 always exists on POSIX (init); use a deliberately huge pid that
    # won't be in use. Windows has no clean equivalent — skip.
    if os.name != "posix":
        pytest.skip("POSIX-only sentinel pid behavior")
    assert process_is_alive(2_999_999) is False


def test_status_returns_not_running_without_pidfile(tmp_path: Path) -> None:
    assert detach_status(tmp_path) == {"status": "not_running"}


def test_status_marks_dead_pid_as_stale(tmp_path: Path) -> None:
    # PID 0 is universally invalid; write_pidfile happily accepts it for tests.
    write_pidfile(tmp_path, pid=2_999_999, web_url="http://127.0.0.1:65535")
    status = detach_status(tmp_path)
    assert status["status"] == "stale_pidfile"
    assert status["pid"] == 2_999_999


def test_status_with_live_pid_reports_running(tmp_path: Path) -> None:
    # Use this test process's own PID for "live".
    write_pidfile(tmp_path, pid=os.getpid(), web_url="http://127.0.0.1:65535")
    status = detach_status(tmp_path)
    assert status["status"] == "running"
    assert status["pid"] == os.getpid()
    # health ping fails (no server listening) → key just absent, no error.
    assert "started" in status


def test_stop_without_pidfile_raises(tmp_path: Path) -> None:
    with pytest.raises(DetachError):
        detach_stop(tmp_path)


def test_stop_with_stale_pidfile_cleans_up(tmp_path: Path) -> None:
    write_pidfile(tmp_path, pid=2_999_999, web_url="http://127.0.0.1:65535")
    info = detach_stop(tmp_path, timeout_seconds=0.5)
    assert info["status"] == "already_stopped"
    assert not pidfile_path(tmp_path).exists()
