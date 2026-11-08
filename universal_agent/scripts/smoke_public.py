#!/usr/bin/env python3
"""Run explicitly offline CLI smoke checks for supplied public task.md files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-zА-Яа-я0-9._-]+", "-", value.strip()).strip("-._")
    return normalized or "task"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="+", type=Path, help="путь к одному или нескольким task.md")
    parser.add_argument(
        "--solution",
        type=Path,
        default=PROJECT_ROOT / "solution.py",
        help="путь к solution.py",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "public-smoke",
        help="каталог для изолированных артефактов каждого запуска",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=300.0,
        help="лимит каждого smoke-запуска в секундах (по умолчанию: 300)",
    )
    args = parser.parse_args(argv)
    solution = args.solution.resolve()
    if not solution.is_file():
        parser.error(f"Не найден CLI entrypoint: {solution}")
    if args.time_budget <= 0:
        parser.error("--time-budget должен быть положительным")

    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith("OPENAI_") or key.startswith("OLLAMA_"):
            environment.pop(key, None)

    failed = False
    for index, task_arg in enumerate(args.tasks, start=1):
        task = task_arg.resolve()
        if not task.is_file():
            print(f"[ошибка] task.md не найден: {task}", file=sys.stderr)
            failed = True
            continue

        base_run_dir = run_root / f"{index:02d}-{_slug(task.parent.name)}"
        run_dir = base_run_dir
        suffix = 2
        while run_dir.exists():
            run_dir = base_run_dir.with_name(f"{base_run_dir.name}-{suffix:02d}")
            suffix += 1
        command = [
            sys.executable,
            str(solution),
            str(task),
            "--offline",
            "--cv-folds", "1",
            "--family-trials", "1",
            "--time-budget",
            str(args.time_budget),
            "--max-trials",
            "1",
            "--n-jobs",
            "1",
            "--run-dir",
            str(run_dir),
        ]
        completed = subprocess.run(
            command,
            cwd=task.parent,
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            failed = True
            print(f"[ошибка] smoke завершился с кодом {completed.returncode}: {task}", file=sys.stderr)
        else:
            print(f"[готово] {task} -> {run_dir}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
