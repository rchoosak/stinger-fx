"""Detach mode — engine runs as a background process; CLI returns immediately.

Design choices:

  • The detached process is just `stinger-fx run --mode web` spawned with
    detach-from-terminal flags. The existing web UI doubles as the control
    plane — no separate IPC protocol.

  • A pidfile under `<data_dir>/engine.pid` holds {pid, web_url, started}.
    `stinger-fx status` checks if the pid is alive and pings the web
    UI's `/health`. `stinger-fx stop` POSTs `/control/shutdown` (which
    raises SIGINT inside the engine, going through the normal cleanup
    path) and falls back to SIGTERM after a timeout.

  • Works on both POSIX (`start_new_session=True`) and Windows
    (`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`).

  • Stale pidfiles (process gone) are detected with `os.kill(pid, 0)` /
    Windows `OpenProcess` and replaced silently — so a hard-killed
    previous run doesn't block the next `--detach`.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("stinger.detach")


PIDFILE_NAME = "engine.pid"


class DetachError(RuntimeError):
    """Raised when detach/status/stop hit an unrecoverable issue."""


def pidfile_path(data_dir: Path) -> Path:
    return data_dir / PIDFILE_NAME


def write_pidfile(
    data_dir: Path,
    *,
    pid: int,
    web_url: str,
) -> Path:
    path = pidfile_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": pid,
        "web_url": web_url,
        "started": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def read_pidfile(data_dir: Path) -> dict[str, Any] | None:
    path = pidfile_path(data_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def remove_pidfile(data_dir: Path) -> None:
    path = pidfile_path(data_dir)
    if path.exists():
        path.unlink()


def process_is_alive(pid: int) -> bool:
    """Cross-platform liveness check. Returns False for dead/never-existed pids."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
        except ImportError:
            return False
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        exit_code = wintypes.DWORD()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return bool(ok) and exit_code.value == STILL_ACTIVE
    # POSIX
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't own it
        return True
    return True


def detach_spawn(
    config_dir: Path,
    data_dir: Path,
    *,
    log_dir: Path | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Spawn `stinger-fx run --mode web` as a detached background process.

    Returns the pidfile dict that was just written.
    Raises DetachError if a live engine is already registered.
    """
    existing = read_pidfile(data_dir)
    if existing and process_is_alive(existing.get("pid", -1)):
        raise DetachError(
            f"detached engine already running pid={existing['pid']} "
            f"(stop it first with `stinger-fx stop` or remove "
            f"{pidfile_path(data_dir)} manually)"
        )

    # Discover the web bind from the running config so the pidfile records
    # the right URL.
    from stinger_fx.config import load_all

    cfg = load_all(config_dir)
    web_url = f"http://{cfg.app.web.host}:{cfg.app.web.port}"

    log_dir = log_dir or (data_dir / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "engine.detached.log"

    cmd: list[str] = [
        sys.executable, "-m", "stinger_fx", "run",
        "--mode", "web",
        "--config-dir", str(config_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)

    popen_kwargs: dict[str, Any] = {
        "stdout": open(log_path, "ab"),  # noqa: SIM115 — kept open for the child
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }
    if sys.platform == "win32":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        popen_kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    write_pidfile(data_dir, pid=proc.pid, web_url=web_url)
    return {"pid": proc.pid, "web_url": web_url, "log": str(log_path)}


def detach_stop(
    data_dir: Path,
    *,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    """Politely shut down the detached engine, fall back to SIGTERM."""
    info = read_pidfile(data_dir)
    if not info:
        raise DetachError(f"no pidfile at {pidfile_path(data_dir)}")
    pid = int(info.get("pid", 0))
    if not process_is_alive(pid):
        remove_pidfile(data_dir)
        return {"status": "already_stopped", "pid": pid}

    # Best effort: ask the web UI to shut itself down cleanly. If that
    # endpoint doesn't exist (very old build) or the request fails, fall
    # straight through to signals.
    web_url = info.get("web_url")
    if web_url:
        try:
            import httpx

            with httpx.Client(timeout=3.0) as client:
                client.post(f"{web_url}/control/shutdown")
        except Exception:  # noqa: BLE001
            logger.debug("/control/shutdown POST failed; falling back to signal")

    # Wait for the process to actually exit, polling periodically.
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not process_is_alive(pid):
            remove_pidfile(data_dir)
            return {"status": "stopped", "pid": pid}
        time.sleep(0.25)

    # Polite shutdown timed out — send the OS-appropriate kill signal.
    try:
        if sys.platform == "win32":
            os.kill(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass

    # Give it a brief grace period after the signal.
    extra_deadline = time.monotonic() + 5.0
    while time.monotonic() < extra_deadline:
        if not process_is_alive(pid):
            remove_pidfile(data_dir)
            return {"status": "terminated", "pid": pid}
        time.sleep(0.25)

    raise DetachError(
        f"failed to stop engine pid={pid} after {timeout_seconds + 5}s — "
        f"check manually with `ps`/`tasklist` and the pidfile at "
        f"{pidfile_path(data_dir)}"
    )


def detach_status(data_dir: Path) -> dict[str, Any]:
    """Read pidfile, check liveness, and ping /health if available."""
    info = read_pidfile(data_dir)
    if not info:
        return {"status": "not_running"}
    pid = int(info.get("pid", 0))
    if not process_is_alive(pid):
        return {"status": "stale_pidfile", "pid": pid, "web_url": info.get("web_url")}
    out: dict[str, Any] = {
        "status": "running",
        "pid": pid,
        "web_url": info.get("web_url"),
        "started": info.get("started"),
    }
    # Best-effort /health ping.
    web_url = info.get("web_url")
    if web_url:
        try:
            import httpx

            with httpx.Client(timeout=2.0) as client:
                r = client.get(f"{web_url}/health")
                if r.status_code == 200:
                    out["health"] = r.json()
        except Exception:  # noqa: BLE001
            pass
    return out
