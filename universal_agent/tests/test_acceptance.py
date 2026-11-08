from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
SOLUTION = REPOSITORY / "solution.py"
BROKEN_HYPOTHESIS = "delivery_acceptance_broken_code_71f3"
RECOVERED_HYPOTHESIS = "delivery_acceptance_recovered_model_71f3"


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class _FakeOpenAIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        self.requests_seen: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        super().__init__(("127.0.0.1", 0), _FakeOpenAIHandler)


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    server: _FakeOpenAIServer

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        with self.server._lock:
            index = len(self.server.requests_seen)
            self.server.requests_seen.append(request)

        if index == 0:
            call = _tool_call(
                "broken-run",
                "run_experiment",
                {
                    "hypothesis": BROKEN_HYPOTHESIS,
                    "code": "def build_model(context):\n    return (\n",
                },
            )
        elif index == 1:
            messages = request.get("messages", [])
            tool_results = [item for item in messages if item.get("role") == "tool"]
            if not tool_results or not any(
                any(word in str(item.get("content", "")).lower() for word in ("error", "invalid", "syntax", "failed"))
                for item in tool_results
            ):
                self._send_error(500, "expected failed experiment outcome in request history")
                return
            call = _tool_call(
                "recovered-run",
                "run_experiment",
                {
                    "hypothesis": RECOVERED_HYPOTHESIS,
                    "code": (
                        "from sklearn.dummy import DummyRegressor\n"
                        "def build_model(context):\n"
                        "    return DummyRegressor(strategy='mean')\n"
                    ),
                },
            )
        else:
            call = _tool_call(
                "finish-research",
                "finish_research",
                {"reason": "The corrected model completed after the first code error."},
            )

        body = {
            "id": f"chatcmpl-delivery-{index}",
            "object": "chat.completion",
            "created": 1,
            "model": request.get("model", "acceptance-fake"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None, "tool_calls": [call]},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_error(self, status: int, message: str) -> None:
        encoded = json.dumps({"error": {"message": message}}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _write_synthetic_task(task_dir: Path) -> Path:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "train.csv").write_text(
        "feature,target\n0,0.0\n1,1.0\n2,2.0\n3,3.0\n4,4.0\n5,5.0\n6,6.0\n7,7.0\n",
        encoding="utf-8",
    )
    (task_dir / "test.csv").write_text("feature\n8\n9\n", encoding="utf-8")
    (task_dir / "sample_submission.csv").write_text("target\n0.0\n0.0\n", encoding="utf-8")
    task = task_dir / "task.md"
    task.write_text(
        "# Synthetic regression acceptance task\n\n"
        "Predict the numeric target from the feature column.\n\n"
        "Task type: regression\nMetric: RMSE\nTarget: target\n\n"
        "Input files: train.csv, test.csv, sample_submission.csv.\n",
        encoding="utf-8",
    )
    return task


def _jsonl_events(run_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(run_dir.rglob("*.jsonl"))
    assert paths, f"No JSONL event journal found in run directory {run_dir}"
    events: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                event = json.loads(line)
                events.append({**event, **event.get("payload", {})})
    return events


def test_fake_openai_cli_recovers_from_bad_experiment_and_finishes(tmp_path: Path) -> None:
    task_path = _write_synthetic_task(tmp_path / "synthetic")
    run_dir = tmp_path / "run-artifacts"
    fake = _FakeOpenAIServer()
    server_thread = threading.Thread(target=fake.serve_forever, daemon=True)
    server_thread.start()
    host, port = fake.server_address
    environment = os.environ.copy()
    environment.update(
        {
            "OPENAI_API_KEY": "acceptance-test-key",
            "OPENAI_MODEL": "acceptance-fake",
            "OPENAI_BASE_URL": f"http://{host}:{port}/v1",
            "AUTOML_PROVIDER": "openai",
        }
    )

    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(SOLUTION),
                "--cv-folds", "1", "--family-trials", "3",
                str(task_path),
                "--experiment-mode",
                "code",
                "--run-dir",
                str(run_dir),
                "--time-budget",
                "90",
                "--trial-timeout",
                "15",
                "--max-trials",
                "4",
                "--max-steps",
                "6",
                "--max-tokens",
                "16000",
                "--response-tokens",
                "256",
                "--llm-timeout",
                "10",
                "--n-jobs",
                "1",
                "--seed",
                "19",
            ],
            cwd=task_path.parent,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    finally:
        fake.shutdown()
        server_thread.join(timeout=5)
        fake.server_close()

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert len(fake.requests_seen) >= 3, "expected failed run, repair, and finish model calls"

    submission = task_path.parent / "submission.csv"
    assert submission.is_file()
    with submission.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert list(rows[0]) == ["target"]
    assert all(math.isfinite(float(row["target"])) for row in rows)

    events = _jsonl_events(run_dir)
    calls = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("event") == "tool_call"
    ]
    broken_call = next(
        (
            (index, event)
            for index, event in calls
            if event.get("name") == "run_experiment"
            and event.get("arguments", {}).get("hypothesis") == BROKEN_HYPOTHESIS
        ),
        None,
    )
    recovered_call = next(
        (
            (index, event)
            for index, event in calls
            if event.get("name") == "run_experiment"
            and event.get("arguments", {}).get("hypothesis") == RECOVERED_HYPOTHESIS
        ),
        None,
    )
    finish_call = next(
        (
            (index, event)
            for index, event in reversed(calls)
            if event.get("name") == "finish_research"
        ),
        None,
    )
    assert broken_call is not None, "journal must retain the failed hypothesis/tool call"
    assert recovered_call is not None, "journal must retain the repaired hypothesis/tool call"
    assert finish_call is not None and finish_call[1].get("arguments", {}).get("reason"), (
        "journal must retain the finish_research call and its reason"
    )

    trials = [
        (index, event.get("result", {}))
        for index, event in enumerate(events)
        if event.get("event") == "trial_completed"
    ]
    broken_trial = next(
        ((index, trial) for index, trial in trials if trial.get("hypothesis") == BROKEN_HYPOTHESIS),
        None,
    )
    recovered_trial = next(
        (
            (index, trial)
            for index, trial in trials
            if trial.get("hypothesis") == RECOVERED_HYPOTHESIS
        ),
        None,
    )
    assert broken_trial is not None and broken_trial[1].get("status") in {
        "error",
        "invalid_code",
        "invalid_output",
    }
    assert recovered_trial is not None and recovered_trial[1].get("status") == "ok"
    assert broken_call[0] < broken_trial[0] < recovered_call[0] < recovered_trial[0] < finish_call[0], (
        "journal must preserve failure, corrected trial, and finish in order"
    )

    report_path = run_dir / "report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report.get("status") == "completed"
    assert report.get("metric") == "rmse"
    assert report.get("submission_path")
