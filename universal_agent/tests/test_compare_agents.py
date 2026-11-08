import os
import sys
from pathlib import Path

import psutil
import pytest

from scripts.compare_agents import agent_command, capture_process, run_stats


def test_restaging_sources_removes_unhashed_stale_modules(tmp_path, monkeypatch):
    import subprocess
    from scripts import compare_agents

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "solution.py").write_text("import pkg.current\n")
    (repo / "pkg").mkdir()
    (repo / "pkg/current.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "solution.py", "pkg/current.py"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Benchmark Test", "-c", "user.email=benchmark@example.invalid",
                    "commit", "-qm", "fixture"], cwd=repo, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    monkeypatch.setattr(compare_agents, "PINS", {"tiny": (revision, "", "pkg")})
    first = compare_agents.stage_sources(repo, tmp_path / "sources", ["tiny"])[0]
    target = Path(first["root"])
    (target / "pkg/stale.py").write_text("VALUE = 'not in pinned source'\n")
    second = compare_agents.stage_sources(repo, tmp_path / "sources", ["tiny"])[0]
    assert not (target / "pkg/stale.py").exists()
    assert second["source_sha256"] == first["source_sha256"]
    assert set(second["runtime_files"]) == {p.relative_to(target).as_posix() for p in target.rglob("*.py")}


@pytest.mark.parametrize("kind", ["universal", "aprelskiy", "oleg", "rakovskiy", "ruzik"])
def test_commands_use_common_time_threads_seed_and_native_fold_options(tmp_path, kind):
    argv = agent_command({"kind": kind, "root": str(tmp_path)}, tmp_path / "task.md",
                         tmp_path / "run", 180, 2, 42)
    assert argv[0] == sys.executable
    assert argv[argv.index("--time-budget") + 1] == "180"
    assert argv[argv.index("--seed") + 1] == "42"
    thread_flag = "--threads" if kind == "ruzik" else "--n-jobs"
    assert argv[argv.index(thread_flag) + 1] == "2"
    fold_flag = "--cv-folds" if kind in {"universal", "aprelskiy"} else "--folds"
    assert argv[argv.index(fold_flag) + 1] == "2"
    if kind in {"universal", "aprelskiy"}:
        assert argv[argv.index("--parallel-trials") + 1] == "1"


def test_successful_process_captures_final_csv_and_nonzero_exit_is_not_hidden(tmp_path):
    script = tmp_path / "producer.py"
    script.write_text("import sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text('id,y\\n001,2.5\\n')\nsys.exit(3)\n")
    submission = tmp_path / "submission.csv"
    snapshot = tmp_path / "captured.csv"
    result = capture_process([sys.executable, str(script), str(submission)], tmp_path, os.environ.copy(),
                             submission, snapshot, 5, tmp_path / "logs")
    assert result["snapshot_exists"] is True
    assert result["returncode"] == 3
    assert result["timed_out"] is False
    assert snapshot.read_text() == "id,y\n001,2.5\n"


def test_deadline_captures_checkpoint_before_cleanup_and_kills_detached_descendant(tmp_path):
    script = tmp_path / "slow.py"
    script.write_text('''import signal, subprocess, sys, time
from pathlib import Path
p = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
Path(sys.argv[2]).write_text(str(p.pid))
Path(sys.argv[1]).write_text('id,y\\n001,1.0\\n')
def terminate(*args):
    Path(sys.argv[1]).write_text('id,y\\n001,999.0\\n')
    raise SystemExit(0)
signal.signal(signal.SIGTERM, terminate)
time.sleep(60)
''')
    submission, snapshot, child_pid = (tmp_path / name for name in ("submission.csv", "captured.csv", "child.pid"))
    result = capture_process([sys.executable, str(script), str(submission), str(child_pid)], tmp_path,
                             os.environ.copy(), submission, snapshot, .8, tmp_path / "logs")
    assert result["timed_out"] is True
    assert result["seconds"] < 1.3
    assert snapshot.read_text() == "id,y\n001,1.0\n"
    pid = int(child_pid.read_text())
    if psutil.pid_exists(pid):
        assert psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_absent_submission_stays_missing_instead_of_becoming_dummy_score(tmp_path):
    result = capture_process([sys.executable, "-c", "raise SystemExit(2)"], tmp_path, os.environ.copy(),
                             tmp_path / "missing.csv", tmp_path / "captured.csv", 5, tmp_path / "logs")
    assert result["snapshot_exists"] is False
    assert result["returncode"] == 2


def test_ruzik_successful_llm_trials_are_read_from_nested_report(tmp_path):
    import json
    (tmp_path / 'report.json').write_text(json.dumps({
        'experiments': [{'status': 'ok'}], 'llm': {'successful_experiments': 3}}))
    assert run_stats('ruzik', tmp_path, tmp_path / 'missing.jsonl')['successful_llm_trials'] == 3


def test_explicit_source_pin_is_used_for_before_after_comparison(tmp_path,monkeypatch):
    import subprocess
    from scripts import compare_agents
    repo=tmp_path/'repo';repo.mkdir()
    (repo/'solution.py').write_text('print("default runtime")\n')
    (repo/'pkg').mkdir();(repo/'pkg/__init__.py').write_text('')
    subprocess.run(['git','init','-q'],cwd=repo,check=True)
    def commit(message):
        subprocess.run(['git','add','solution.py','pkg/__init__.py'],cwd=repo,check=True)
        subprocess.run(['git','-c','user.name=Benchmark Test','-c','user.email=benchmark@example.invalid','commit','-qm',message],cwd=repo,check=True)
        return subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    default=commit('default')
    (repo/'solution.py').write_text('print("explicit frozen runtime")\n')
    revision=commit('explicit override')
    monkeypatch.setattr(compare_agents,'PINS',{'tiny':(default,'','pkg')})
    import inspect
    override={'pins':{'tiny':(revision,'','pkg')}} if 'pins' in inspect.signature(compare_agents.stage_sources).parameters else {}
    result=compare_agents.stage_sources(repo,tmp_path/'staged',['tiny'],**override)[0]
    assert result['revision']==revision,'explicit comparison revision must override the default pin'
    assert (Path(result['root'])/'solution.py').read_text()=='print("explicit frozen runtime")\n'
