"""Bounded orchestration for independent evaluator processes."""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

from .contracts import ProcessResult
from .runtime import run_process

_MAX_WORKERS = 2
_MAX_CONCURRENT_JOBS = 8
_MAX_CONCURRENT_MEMORY_MB = 28000
_DEADLINE_MESSAGE = "research deadline reached"


def _validate_jobs(jobs: list[dict], max_workers: int) -> list[tuple[int, int]]:
    if (
        not isinstance(max_workers, int)
        or isinstance(max_workers, bool)
        or not 1 <= max_workers <= _MAX_WORKERS
    ):
        raise ValueError("max_workers must be between 1 and 2")
    if not isinstance(jobs, list):
        raise TypeError("jobs must be a list of dictionaries")

    normalized: list[tuple[int, int]] = []
    required = {"argv", "cwd", "timeout", "n_jobs", "memory_limit_mb", "env"}
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise TypeError(f"job {index} must be a dictionary")
        missing = required.difference(job)
        if missing:
            raise ValueError(f"job {index} is missing required fields: {', '.join(sorted(missing))}")
        argv = job["argv"]
        if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) for arg in argv):
            raise ValueError(f"job {index} argv must be a non-empty list of strings")
        if not isinstance(job["cwd"], (str, os.PathLike)):
            raise ValueError(f"job {index} cwd must be a path")
        timeout = job["timeout"]
        try:
            timeout_value = float(timeout)
        except (OverflowError, TypeError, ValueError):
            timeout_value = math.nan
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout_value)
            or timeout_value <= 0
        ):
            raise ValueError(f"job {index} timeout must be a finite positive number")
        n_jobs = job["n_jobs"]
        if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or not 1 <= n_jobs <= _MAX_CONCURRENT_JOBS:
            raise ValueError(f"job {index} n_jobs must be between 1 and 8")
        memory = job["memory_limit_mb"]
        if memory is not None and (
            not isinstance(memory, int) or isinstance(memory, bool) or memory <= 0
        ):
            raise ValueError(f"job {index} memory_limit_mb must be a positive integer or None")
        if memory is not None and memory > _MAX_CONCURRENT_MEMORY_MB:
            raise ValueError(f"job {index} memory_limit_mb cannot exceed 28000")
        if not isinstance(job["env"], dict) or any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in job["env"].items()
        ):
            raise ValueError(f"job {index} env must map strings to strings")
        normalized.append((n_jobs, memory or 0))
    return normalized


def _cancelled_result(n_jobs: int, *, message: str = _DEADLINE_MESSAGE) -> ProcessResult:
    return ProcessResult(
        status="cancelled",
        exit_code=None,
        stdout="",
        stderr=message,
        seconds=0.0,
        peak_memory_mb=0.0,
        n_jobs=n_jobs,
    )


def _with_deadline_message(result: ProcessResult) -> ProcessResult:
    if result.status != "cancelled" or result.stderr == _DEADLINE_MESSAGE:
        return result
    return ProcessResult(
        status=result.status,
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=_DEADLINE_MESSAGE,
        seconds=result.seconds,
        peak_memory_mb=result.peak_memory_mb,
        started_at=result.started_at,
        finished_at=result.finished_at,
        pid=result.pid,
        n_jobs=result.n_jobs,
    )


def _deadline_reached(callback: Callable[[], float]) -> bool:
    remaining = callback()
    try:
        remaining_value = float(remaining)
    except (OverflowError, TypeError, ValueError):
        remaining_value = math.nan
    if (
        isinstance(remaining, bool)
        or not isinstance(remaining, (int, float))
        or not math.isfinite(remaining_value)
    ):
        raise ValueError("research_remaining must return a finite number")
    return remaining_value <= 0


def run_parallel_processes(
    jobs: list[dict],
    *,
    max_workers: int = 2,
    research_remaining: Callable[[], float] | None = None,
) -> Iterator[tuple[int, ProcessResult]]:
    """Run bounded subprocesses and yield each result once as it completes.

    All admissions, deadline checks, and yields happen on the caller thread;
    executor threads call only :func:`run_process`.
    """
    resources = _validate_jobs(jobs, max_workers)
    if research_remaining is not None and not callable(research_remaining):
        raise TypeError("research_remaining must be callable or None")

    cancellation = threading.Event()
    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="automl-process")
    pending = list(range(len(jobs)))
    active: dict[Future[ProcessResult], tuple[int, int, int]] = {}
    deadline_cancelled = False

    try:
        while pending or active:
            if research_remaining is not None and not deadline_cancelled:
                if _deadline_reached(research_remaining):
                    deadline_cancelled = True
                    cancellation.set()

            if not deadline_cancelled:
                while pending and len(active) < max_workers:
                    if research_remaining is not None and _deadline_reached(research_remaining):
                        deadline_cancelled = True
                        cancellation.set()
                        break

                    running_n_jobs = sum(item[1] for item in active.values())
                    running_memory = sum(item[2] for item in active.values())
                    selected_position = next(
                        (
                            position
                            for position, index in enumerate(pending)
                            if running_n_jobs + resources[index][0] <= _MAX_CONCURRENT_JOBS
                            and running_memory + resources[index][1] <= _MAX_CONCURRENT_MEMORY_MB
                        ),
                        None,
                    )
                    if selected_position is None:
                        break
                    index = pending.pop(selected_position)
                    job = jobs[index]
                    future = executor.submit(
                        run_process,
                        job["argv"],
                        cwd=Path(job["cwd"]),
                        timeout=job["timeout"],
                        n_jobs=job["n_jobs"],
                        memory_limit_mb=job["memory_limit_mb"],
                        env=job["env"],
                        cancel_event=cancellation,
                    )
                    active[future] = (index, *resources[index])

            if not active:
                if deadline_cancelled:
                    for index in pending:
                        yield index, _cancelled_result(resources[index][0])
                    pending.clear()
                    break
                if pending:
                    # Valid per-job values guarantee at least one queued job fits an empty slot.
                    raise RuntimeError("scheduler could not admit a valid queued job")
                break

            completed, _ = wait(tuple(active), timeout=0.02, return_when=FIRST_COMPLETED)
            for future in completed:
                index, _, _ = active.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = ProcessResult(
                        status="error",
                        exit_code=None,
                        stdout="",
                        stderr=f"process runner failed: {type(exc).__name__}: {exc}",
                        seconds=0.0,
                        n_jobs=jobs[index]["n_jobs"],
                    )
                if deadline_cancelled:
                    result = _with_deadline_message(result)
                yield index, result
    finally:
        if active:
            cancellation.set()
        # close(), callback failures, and KeyboardInterrupt all wait for process-tree cleanup.
        executor.shutdown(wait=True, cancel_futures=True)
