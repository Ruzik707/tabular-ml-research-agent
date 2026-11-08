import json
import os
import sys
import time
from pathlib import Path

import pytest

from automl_agent.runtime import Budget, run_process


def test_budget_preserves_resume_time_and_final_reserve():
    budget = Budget(total_seconds=10, reserve_seconds=3, spent_seconds=2)
    assert budget.elapsed() >= 2
    assert budget.remaining() == pytest.approx(8, abs=0.01)
    assert budget.research_remaining() == pytest.approx(5, abs=0.01)
    time.sleep(0.02)
    assert 2.01 <= budget.elapsed() < 3
    assert budget.research_remaining() < 5


def test_budget_never_spends_final_reserve():
    budget = Budget(total_seconds=5, reserve_seconds=6, spent_seconds=1)
    assert budget.research_remaining() == 0


def test_run_process_passes_only_explicit_environment_and_limits_threads(tmp_path):
    code = (
        "import json, os; print(json.dumps({k: os.environ.get(k) for k in "
        "['PATH','PYTHONPATH','OPENAI_API_KEY','AUTOML_TEST','OMP_NUM_THREADS',"
        "'OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}))"
    )
    result = run_process(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        timeout=5,
        n_jobs=3,
        env={"PYTHONPATH": os.environ.get("PYTHONPATH", ""), "OPENAI_API_KEY": "never-pass", "AUTOML_TEST": "yes"},
    )
    assert result.status == "ok"
    child_env = json.loads(result.stdout)
    assert child_env["PATH"]
    assert child_env["PYTHONPATH"] == os.environ.get("PYTHONPATH", "")
    assert child_env["OPENAI_API_KEY"] is None
    assert child_env["AUTOML_TEST"] == "yes"
    assert child_env["OMP_NUM_THREADS"] == "3"
    assert child_env["OPENBLAS_NUM_THREADS"] == "3"
    assert child_env["MKL_NUM_THREADS"] == "3"


def test_run_process_caps_output_while_streaming(tmp_path):
    code = "import os, sys;\nwhile True:\n os.write(1, b'x' * 4096); os.write(2, b'y' * 4096)"
    result = run_process(
        [sys.executable, "-c", code], cwd=tmp_path, timeout=5, max_output_bytes=2048
    )
    assert result.status == "output_limit"
    assert len(result.stdout.encode()) + len(result.stderr.encode()) <= 2048


def test_run_process_timeout_kills_process_group_descendants(tmp_path):
    marker = tmp_path / "descendant-finished"
    child_code = f"import time; time.sleep(0.7); open({str(marker)!r}, 'w').write('alive')"
    parent_code = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "time.sleep(30)"
    )
    result = run_process(
        [sys.executable, "-c", parent_code], cwd=tmp_path, timeout=0.35
    )
    assert result.status == "timeout"
    time.sleep(0.85)
    assert not marker.exists(), "a grandchild survived the supervisor timeout"


def test_run_process_enforces_process_tree_rss_cap(tmp_path):
    code = "import time; blob = bytearray(128 * 1024 * 1024); time.sleep(2)"
    result = run_process(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        timeout=5,
        memory_limit_mb=48,
    )
    assert result.status == "memory_limit"
    assert result.peak_memory_mb >= 48


def test_run_process_reports_nonzero_exit(tmp_path):
    result = run_process(
        [sys.executable, "-c", "import sys; print('bad', file=sys.stderr); sys.exit(7)"],
        cwd=tmp_path,
        timeout=5,
    )
    assert result.status == "error"
    assert result.exit_code == 7
    assert "bad" in result.stderr
