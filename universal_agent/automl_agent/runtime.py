"""Process supervision and wall-clock budgets for the AutoML harness.

The supervisor manages one process group and its resource use. It does not
create an operating-system security sandbox; the host/container owns that
boundary.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import BinaryIO

import psutil

from .contracts import ProcessResult


class Budget:
    """A monotonic wall-clock budget that can resume from saved elapsed time."""

    def __init__(
        self,
        total_seconds: float,
        reserve_seconds: float = 60,
        spent_seconds: float = 0,
    ) -> None:
        values = (total_seconds, reserve_seconds, spent_seconds)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("budget values must be finite")
        if total_seconds < 0 or reserve_seconds < 0 or spent_seconds < 0:
            raise ValueError("budget values cannot be negative")
        self.total_seconds = float(total_seconds)
        self.reserve_seconds = float(reserve_seconds)
        self.spent_seconds = float(spent_seconds)
        self._started_at = time.monotonic()

    def elapsed(self) -> float:
        return self.spent_seconds + max(0.0, time.monotonic() - self._started_at)

    def remaining(self) -> float:
        return max(0.0, self.total_seconds - self.elapsed())

    def research_remaining(self) -> float:
        return max(0.0, self.total_seconds - self.elapsed() - self.reserve_seconds)


def _is_sensitive_env_name(name: str) -> bool:
    upper = name.upper()
    return any(part in upper for part in ("_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH"))


def _child_environment(n_jobs: int, explicit: dict[str, str] | None) -> dict[str, str]:
    """Build a small child environment rather than inheriting the host's."""
    parent_path = os.environ.get("PATH", os.defpath)
    result = {
        "PATH": parent_path,
        "PYTHONIOENCODING": "utf-8",
        "OMP_NUM_THREADS": str(n_jobs),
        "OPENBLAS_NUM_THREADS": str(n_jobs),
        "MKL_NUM_THREADS": str(n_jobs),
        "NUMEXPR_NUM_THREADS": str(n_jobs),
        "VECLIB_MAXIMUM_THREADS": str(n_jobs),
        "BLIS_NUM_THREADS": str(n_jobs),
        "RAYON_NUM_THREADS": str(n_jobs),
        "JOBLIB_MULTIPROCESSING": "0",
    }
    if os.name == "nt" and "SYSTEMROOT" in os.environ:
        result["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    for name, value in (explicit or {}).items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise TypeError("environment variable names and values must be strings")
        if _is_sensitive_env_name(name):
            continue
        result[name] = value
    return result


def _process_tree_rss_mb(pid: int) -> float:
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return total / (1024 * 1024)


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    """Stop the owned process group, escalating after a short grace period."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            pass
        grace_deadline = time.monotonic() + 0.2
        while time.monotonic() < grace_deadline:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            except OSError:
                break
            time.sleep(0.01)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            pass
    else:
        try:
            root = psutil.Process(process.pid)
            children = root.children(recursive=True)
            for child in reversed(children):
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            try:
                root.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass


def _read_stream(
    stream: BinaryIO,
    destination: bytearray,
    byte_limit: int,
    byte_count: list[int],
    lock: threading.Lock,
    overflow: threading.Event,
) -> None:
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                return
            with lock:
                remaining = byte_limit - byte_count[0]
                if remaining <= 0:
                    overflow.set()
                    continue
                kept = chunk[:remaining]
                destination.extend(kept)
                byte_count[0] += len(kept)
                if len(chunk) > remaining:
                    overflow.set()
    finally:
        try:
            stream.close()
        except OSError:
            pass


def run_process(
    argv: list[str],
    *,
    cwd: Path,
    timeout: float,
    n_jobs: int = 4,
    max_output_bytes: int = 131072,
    memory_limit_mb: int | None = None,
    env: dict[str, str] | None = None,
    cancel_event: threading.Event | None = None,
) -> ProcessResult:
    """Run an argv directly with capped output, wall timeout, and tree RSS cap."""
    if not argv or any(not isinstance(arg, str) for arg in argv):
        raise ValueError("argv must be a non-empty list of strings")
    if not math.isfinite(float(timeout)) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number")
    if not isinstance(n_jobs, int) or n_jobs < 1 or n_jobs > 8:
        raise ValueError("n_jobs must be between 1 and 8")
    if not isinstance(max_output_bytes, int) or max_output_bytes < 0:
        raise ValueError("max_output_bytes must be a non-negative integer")
    if memory_limit_mb is not None and (not isinstance(memory_limit_mb, int) or memory_limit_mb <= 0):
        raise ValueError("memory_limit_mb must be a positive integer")

    started = time.monotonic()
    if cancel_event is not None and cancel_event.is_set():
        return ProcessResult(
            status="cancelled",
            exit_code=None,
            stdout="",
            stderr="",
            seconds=0.0,
            peak_memory_mb=0.0,
            n_jobs=n_jobs,
        )
    stdout_data = bytearray()
    stderr_data = bytearray()
    byte_count = [0]
    lock = threading.Lock()
    output_overflow = threading.Event()
    status: str | None = None
    peak_memory_mb = 0.0
    process: subprocess.Popen[bytes] | None = None
    threads: list[threading.Thread] = []
    started_at: float | None = None
    finished_at: float | None = None

    try:
        popen_options: dict[str, object] = {}
        if os.name == "posix":
            popen_options["start_new_session"] = True
        elif os.name == "nt":
            popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        started_at = time.time()
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=_child_environment(n_jobs, env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            **popen_options,
        )
    except (OSError, ValueError, TypeError) as exc:
        return ProcessResult(
            status="error",
            exit_code=None,
            stdout="",
            stderr=f"could not start process: {type(exc).__name__}: {exc}",
            seconds=max(0.0, time.monotonic() - started),
            peak_memory_mb=0.0,
            started_at=started_at,
            finished_at=time.time(),
            n_jobs=n_jobs,
        )

    assert process.stdout is not None and process.stderr is not None
    for stream, destination in ((process.stdout, stdout_data), (process.stderr, stderr_data)):
        thread = threading.Thread(
            target=_read_stream,
            args=(stream, destination, max_output_bytes, byte_count, lock, output_overflow),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    deadline = started + timeout
    try:
        while process.poll() is None or any(thread.is_alive() for thread in threads):
            current_rss = _process_tree_rss_mb(process.pid)
            peak_memory_mb = max(peak_memory_mb, current_rss)
            if cancel_event is not None and cancel_event.is_set():
                status = "cancelled"
                _kill_process_group(process)
                break
            if output_overflow.is_set():
                status = "output_limit"
                _kill_process_group(process)
                break
            if memory_limit_mb is not None and current_rss > memory_limit_mb:
                status = "memory_limit"
                _kill_process_group(process)
                break
            if time.monotonic() >= deadline:
                status = "timeout"
                _kill_process_group(process)
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        status = "cancelled"
        _kill_process_group(process)

    if status in {"timeout", "output_limit", "memory_limit", "cancelled"} and process.poll() is None:
        _kill_process_group(process)
    for thread in threads:
        thread.join(timeout=1)
    if any(thread.is_alive() for thread in threads):
        # A pipe held open by a process outside the owned group must not hang the host.
        for stream in (process.stdout, process.stderr):
            try:
                stream.close()
            except OSError:
                pass
        for thread in threads:
            thread.join(timeout=0.1)

    exit_code = process.poll()
    if status is None:
        status = "ok" if exit_code == 0 else "error"
    finished_at = time.time()
    return ProcessResult(
        status=status,
        exit_code=exit_code,
        stdout=stdout_data.decode("utf-8", errors="replace"),
        stderr=stderr_data.decode("utf-8", errors="replace"),
        seconds=max(0.0, time.monotonic() - started),
        peak_memory_mb=peak_memory_mb,
        started_at=started_at,
        finished_at=finished_at,
        pid=process.pid,
        n_jobs=n_jobs,
    )
