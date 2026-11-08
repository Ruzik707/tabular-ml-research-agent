from __future__ import annotations

import subprocess
import sys
import zipfile
import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGER = REPOSITORY / "scripts" / "package_solution.py"


def _make_fake_project(root: Path) -> set[str]:
    files = {
        "solution.py": "print('fixture')\n",
        "automl_agent/__init__.py": "\"\"\"Fixture package.\"\"\"\n",
        "automl_agent/contracts.py": "TASK_CONTRACT = True\n",
        "automl_agent/nested/worker.py": "def main(): pass\n",
        "README.md": "Fixture documentation\n",
        ".env.example": "OPENAI_MODEL=example\n",
        "requirements-mvp.txt": "numpy==1.26.4\n",
        "tests/test_hidden.py": "assert False\n",
        "docs/research/private-notes.md": "research log\n",
        "data/train.csv": "secret,dataset\n",
        ".env": "OPENAI_API_KEY=not-for-archive\n",
        ".venv/pyvenv.cfg": "home = /private\n",
    }
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    return {
        "solution.py",
        "automl_agent/__init__.py",
        "automl_agent/contracts.py",
        "automl_agent/nested/worker.py",
        "README.md",
        ".env.example",
        "requirements-mvp.txt",
    }


def test_packager_includes_only_runtime_allowlist_and_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source"
    expected = _make_fake_project(source)
    first = tmp_path / "one.zip"
    second = tmp_path / "two.zip"

    for output in (first, second):
        completed = subprocess.run(
            [sys.executable, str(PACKAGER), "--root", str(source), "--output", str(output)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert output.is_file()

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
        assert names == expected
        assert "solution.py" in names
        assert not any(
            name == "train.csv"
            or name == ".env"
            or name.startswith((".venv/", "tests/", "research/", "docs/research/", "data/"))
            or "private" in name
            for name in names
        )
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
        assert archive.read("solution.py") == b"print('fixture')\n"


def test_packager_fails_clearly_when_solution_entrypoint_is_missing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_fake_project(source)
    (source / "solution.py").unlink()

    completed = subprocess.run(
        [sys.executable, str(PACKAGER), "--root", str(source), "--output", str(tmp_path / "out.zip")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "solution.py" in (completed.stdout + completed.stderr)
    assert not (tmp_path / "out.zip").exists()


def test_actual_archive_solves_a_task_from_an_unrelated_working_directory(tmp_path):
    output = tmp_path / "runtime.zip"
    subprocess.run([sys.executable, str(PACKAGER), "--output", str(output)], check=True, capture_output=True)
    source = tmp_path / "unpacked"
    with zipfile.ZipFile(output) as archive:
        archive.extractall(source)
    task_dir = tmp_path / "данные с пробелом"
    task_dir.mkdir()
    (task_dir / "train.csv").write_text("record,x,target\n" + "".join(f"{i},{i},{i * 2}\n" for i in range(30)))
    (task_dir / "test.csv").write_text("record,x\n91,4\n92,8\n")
    (task_dir / "sample_submission.csv").write_text("record,target\n91,0\n92,0\n")
    task = task_dir / "task.md"
    task.write_text("Regression; target=target; metric RMSE. train.csv test.csv sample_submission.csv")
    result = subprocess.run([sys.executable, str(source / "solution.py"), str(task), "--offline", "--time-budget", "30", "--n-jobs", "1", "--run-dir", str(tmp_path / "artifacts")],
                            cwd=tmp_path, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "offline" and report["best_metric"] is not None
    assert (task_dir / "submission.csv").read_text().splitlines()[0] == "record,target"
