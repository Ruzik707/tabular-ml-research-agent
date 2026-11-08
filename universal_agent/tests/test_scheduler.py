import json
import sys
import time
from pathlib import Path

import pytest

import automl_agent.scheduler as scheduler_module
from automl_agent.scheduler import run_parallel_processes


def _job(code: str, cwd: Path, *, n_jobs: int = 4, memory: int = 12000, timeout: float = 5):
    return {
        "argv": [sys.executable, "-c", code],
        "cwd": cwd,
        "timeout": timeout,
        "n_jobs": n_jobs,
        "memory_limit_mb": memory,
        "env": {},
    }


def test_two_four_cpu_processes_overlap_and_report_real_intervals(tmp_path):
    jobs = []
    for index in range(2):
        output = tmp_path / f"interval-{index}.json"
        code = (
            "import json, os, time; "
            "start=time.time(); time.sleep(.35); finish=time.time(); "
            f"open({str(output)!r}, 'w').write(json.dumps([start, finish, "
            "os.environ['OMP_NUM_THREADS'], os.environ['OPENBLAS_NUM_THREADS']]))"
        )
        jobs.append(_job(code, tmp_path, n_jobs=4))

    results = dict(run_parallel_processes(jobs))
    assert set(results) == {0, 1}
    intervals = [json.loads((tmp_path / f"interval-{i}.json").read_text()) for i in range(2)]
    assert max(row[0] for row in intervals) < min(row[1] for row in intervals)
    for index, result in results.items():
        assert result.status == "ok"
        assert result.started_at is not None and result.finished_at is not None
        assert result.started_at <= result.finished_at
        assert result.pid and result.n_jobs == 4
        assert result.n_jobs == int(intervals[index][2]) == int(intervals[index][3])


def test_failure_timeout_and_success_are_returned_once_independently(tmp_path):
    jobs = [
        _job("import sys; print('failed', file=sys.stderr); sys.exit(7)", tmp_path),
        _job("import time; time.sleep(30)", tmp_path, timeout=0.2),
        _job("print('survived')", tmp_path),
    ]
    yielded = list(run_parallel_processes(jobs))
    assert sorted(index for index, _ in yielded) == [0, 1, 2]
    assert len({index for index, _ in yielded}) == 3
    results = dict(yielded)
    assert (results[0].status, results[0].exit_code) == ("error", 7)
    assert "failed" in results[0].stderr
    assert results[1].status == "timeout"
    assert results[2].status == "ok"
    assert "survived" in results[2].stdout


def test_admission_never_exceeds_worker_cpu_or_memory_caps(tmp_path):
    jobs = []
    for index, (n_jobs, memory) in enumerate(((6, 10000), (3, 18000), (3, 18000))):
        output = tmp_path / f"resource-{index}.json"
        code = (
            "import json, os, time; start=time.time(); time.sleep(.3); finish=time.time(); "
            f"open({str(output)!r}, 'w').write(json.dumps([start, finish, "
            f"{n_jobs}, {memory}]))"
        )
        jobs.append(_job(code, tmp_path, n_jobs=n_jobs, memory=memory))
    assert len(list(run_parallel_processes(jobs, max_workers=2))) == 3

    intervals = [json.loads((tmp_path / f"resource-{i}.json").read_text()) for i in range(3)]
    for point in {row[0] for row in intervals}:
        active = [row for row in intervals if row[0] <= point < row[1]]
        assert len(active) <= 2
        assert sum(row[2] for row in active) <= 8
        assert sum(row[3] for row in active) <= 28000


@pytest.mark.parametrize(
    "jobs,max_workers",
    [
        ([_job("pass", Path("."), n_jobs=9)], 2),
        ([_job("pass", Path("."), memory=28001)], 2),
        ([_job("pass", Path("."))], 3),
    ],
)
def test_invalid_resource_configuration_is_rejected(jobs, max_workers):
    with pytest.raises(ValueError):
        list(run_parallel_processes(jobs, max_workers=max_workers))


def test_dynamic_research_deadline_cancels_active_and_skips_queued_jobs(tmp_path):
    parent_marker = tmp_path / "active-started"
    queued_marker = tmp_path / "queued-started"
    jobs = [
        _job(
            f"import time; open({str(parent_marker)!r}, 'w').write('yes'); time.sleep(30)",
            tmp_path,
        ),
        _job("import time; time.sleep(30)", tmp_path),
        _job(f"open({str(queued_marker)!r}, 'w').write('yes')", tmp_path),
    ]
    first_started = None

    def remaining():
        nonlocal first_started
        if parent_marker.exists():
            first_started = first_started or time.monotonic()
            return 0 if time.monotonic() - first_started > 0.2 else 1
        return 1

    results = dict(run_parallel_processes(jobs, research_remaining=remaining))
    assert set(results) == {0, 1, 2}
    assert results[0].status == "cancelled"
    assert results[0].stderr == "research deadline reached"
    assert all(results[index].status == "cancelled" for index in (1, 2))
    assert all(results[index].stderr == "research deadline reached" for index in (1, 2))
    assert not queued_marker.exists()


def test_deadline_cancellation_kills_parent_and_descendant(tmp_path):
    parent_pid = tmp_path / "parent.pid"
    child_pid = tmp_path / "child.pid"
    child_marker = tmp_path / "child-survived"
    child_code = (
        f"import os, time; open({str(child_pid)!r}, 'w').write(str(os.getpid())); "
        f"time.sleep(.75); open({str(child_marker)!r}, 'w').write('alive')"
    )
    parent_code = (
        "import os, subprocess, sys, time; "
        f"open({str(parent_pid)!r}, 'w').write(str(os.getpid())); "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); time.sleep(30)"
    )
    jobs = [_job(parent_code, tmp_path)]
    started = None

    def remaining():
        nonlocal started
        if parent_pid.exists():
            started = started or time.monotonic()
            return 0 if time.monotonic() - started > 0.2 else 1
        return 1

    result = dict(run_parallel_processes(jobs, research_remaining=remaining))[0]
    assert result.status == "cancelled"
    time.sleep(0.85)
    assert parent_pid.exists() and child_pid.exists()
    assert not child_marker.exists(), "a descendant survived deadline cancellation"


def test_generator_close_cancels_remaining_process_tree(tmp_path):
    first_marker = tmp_path / "first-done"
    descendant_marker = tmp_path / "late-descendant"
    child_code = (
        f"import time; time.sleep(.7); open({str(descendant_marker)!r}, 'w').write('alive')"
    )
    long_code = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {child_code!r}]); time.sleep(30)"
    jobs = [
        _job(f"import time; time.sleep(.15); open({str(first_marker)!r}, 'w').write('done')", tmp_path),
        _job(long_code, tmp_path),
    ]
    generator = run_parallel_processes(jobs)
    first_index, first_result = next(generator)
    assert first_index == 0 and first_result.status == "ok"
    generator.close()
    time.sleep(0.85)
    assert first_marker.exists()
    assert not descendant_marker.exists(), "generator close left an active descendant"


def test_callback_error_cancels_active_process_before_reraising(tmp_path):
    parent_marker = tmp_path / "error-parent-started"
    child_marker = tmp_path / "error-descendant-survived"
    child_code = (
        f"import time; time.sleep(.7); open({str(child_marker)!r}, 'w').write('alive')"
    )
    parent_code = (
        "import subprocess, sys, time; "
        f"open({str(parent_marker)!r}, 'w').write('yes'); "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); time.sleep(30)"
    )
    calls = 0

    def broken_remaining():
        nonlocal calls
        calls += 1
        if parent_marker.exists() and calls > 2:
            raise RuntimeError("budget callback failed")
        return 1

    with pytest.raises(RuntimeError, match="budget callback failed"):
        list(run_parallel_processes([_job(parent_code, tmp_path)], research_remaining=broken_remaining))
    time.sleep(0.85)
    assert not child_marker.exists(), "callback error left a descendant running"


def test_keyboard_interrupt_cancels_active_process_tree(tmp_path, monkeypatch):
    parent_marker = tmp_path / "interrupt-parent-started"
    child_pid = tmp_path / "interrupt-child.pid"
    child_marker = tmp_path / "interrupt-descendant-survived"
    child_code = (
        f"import os, time; open({str(child_pid)!r}, 'w').write(str(os.getpid())); "
        f"time.sleep(.7); open({str(child_marker)!r}, 'w').write('alive')"
    )
    parent_code = (
        "import os, subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        f"while not os.path.exists({str(child_pid)!r}):\n"
        "    time.sleep(.005)\n"
        f"open({str(parent_marker)!r}, 'w').write('yes')\n"
        "time.sleep(30)"
    )

    def interrupt_after_launch(futures, *, timeout, return_when):
        deadline = time.monotonic() + 2
        while not parent_marker.exists() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert parent_marker.exists(), "worker did not launch before wait"
        raise KeyboardInterrupt

    monkeypatch.setattr(scheduler_module, "wait", interrupt_after_launch)
    with pytest.raises(KeyboardInterrupt):
        list(run_parallel_processes([_job(parent_code, tmp_path)]))
    assert child_pid.exists()
    time.sleep(0.85)
    assert not child_marker.exists(), "keyboard interrupt left a descendant running"
