import json
import time

import numpy as np
import pandas as pd
import pytest

from automl_agent.agent import run, select_best
from automl_agent.config import parse_config
from automl_agent.contracts import ExperimentResult
from automl_agent.agent import _run_lock


def trial(name, metric, status="ok"):
    return ExperimentResult(name, status, "/tmp/trial", name, "hypothesis", metric=metric)


def test_best_selection_ignores_claimed_success_and_non_finite_scores():
    candidates = [trial("bad", 10000, "error"), trial("nan", float("nan")), trial("a", 0.6), trial("b", 0.8)]
    assert select_best(candidates, True).trial_id == "b"
    assert select_best(candidates, False).trial_id == "a"
    assert select_best([trial("failed", 0, "error")], False) is None


def make_task(path):
    path.mkdir()
    values = np.linspace(0, 10, 60)
    pd.DataFrame({"row": range(60), "feature": values, "outcome": values * 2 + 1}).to_csv(path / "train.csv", index=False)
    pd.DataFrame({"row": range(100, 110), "feature": np.linspace(1, 4, 10)}).to_csv(path / "test.csv", index=False)
    pd.DataFrame({"row": range(100, 110), "outcome": 0.0}).to_csv(path / "sample_submission.csv", index=False)
    (path / "task.md").write_text("Regression. Predict outcome. Metric RMSE. Files: train.csv test.csv sample_submission.csv. Save submission.csv.")
    return path / "task.md"


def test_offline_run_and_resume_preserve_valid_result(tmp_path):
    task = make_task(tmp_path / "dataset")
    run_dir = tmp_path / "run"
    args = ["--cv-folds", "1", "--family-trials", "3", str(task), "--offline", "--run-dir", str(run_dir), "--time-budget", "90", "--trial-timeout", "30", "--n-jobs", "1"]
    report = run(parse_config(args))
    assert report["status"] == "offline"
    assert report["llm_calls"] == 0
    assert report["best_metric"] is not None
    output = pd.read_csv(task.parent / "submission.csv")
    assert list(output.columns) == ["row", "outcome"]
    assert list(output["row"]) == list(range(100, 110))
    assert np.isfinite(output["outcome"]).all()
    resumed = run(parse_config(["--cv-folds", "1", "--family-trials", "3", *args, "--resume"]))
    assert resumed["selected_trial"] == report["selected_trial"]
    assert len(resumed["trials"]) == len(report["trials"])
    frame = pd.read_csv(task.parent / "train.csv")
    frame.loc[0, "outcome"] += 10
    frame.to_csv(task.parent / "train.csv", index=False)
    with pytest.raises(ValueError, match="fingerprint|changed|измен"):
        run(parse_config(["--cv-folds", "1", "--family-trials", "3", *args, "--resume"]))


def test_concurrent_resume_cannot_settle_another_process_pending_call(tmp_path):
    from automl_agent.journal import EventJournal, atomic_write_json

    task = make_task(tmp_path / "dataset")
    directory = tmp_path / "run"
    args = ["--cv-folds", "1", "--family-trials", "3", str(task), "--offline", "--run-dir", str(directory), "--time-budget", "90", "--n-jobs", "1"]
    run(parse_config(args))
    state = json.loads((directory / "state.json").read_text())
    state["active"] = True
    atomic_write_json(directory / "state.json", state)
    journal = EventJournal(directory / "events.jsonl")
    journal.append("tool_call", call_id="owned-by-running-process", name="run_experiment", arguments={"hypothesis": "pending"})
    before = (directory / "events.jsonl").read_bytes()
    with _run_lock(directory / "run.lock"):
        with pytest.raises(ValueError, match="Another process"):
            run(parse_config(["--cv-folds", "1", "--family-trials", "3", *args, "--resume"]))
    assert (directory / "events.jsonl").read_bytes() == before


def test_hung_generated_fit_is_killed_and_preserves_submission(tmp_path, monkeypatch):
    from automl_agent.contracts import ModelResponse
    from automl_agent.providers import OpenAICompatibleProvider

    task = make_task(tmp_path / "dataset")
    code = """from sklearn.base import BaseEstimator
class HungEstimator(BaseEstimator):
    def fit(self, X, y):
        while True:
            pass
    def predict(self, X):
        return [0.0] * len(X)
def build_model(context):
    return HungEstimator()
"""
    def complete(*args, **kwargs):
        return ModelResponse({"role": "assistant", "tool_calls": [{"id": "hang", "type": "function", "function": {
            "name": "run_experiment", "arguments": json.dumps({"hypothesis": "simulate a hung fit", "code": code})}}]}, total_tokens=20)
    monkeypatch.setattr(OpenAICompatibleProvider, "complete", complete)
    monkeypatch.setenv("OPENAI_MODEL", "fake")
    monkeypatch.setenv("AUTOML_PROVIDER", "openai")
    config = parse_config(["--cv-folds", "1", "--family-trials", "3", str(task), "--experiment-mode", "code", "--run-dir", str(tmp_path / "run"), "--time-budget", "30", "--trial-timeout", "0.75", "--max-trials", "2", "--n-jobs", "1"])
    report = run(config)
    assert report["trials"][-1]["status"] == "timeout"
    assert report["elapsed_seconds"] < 15
    output = pd.read_csv(task.parent / "submission.csv")
    assert len(output) == 10 and np.isfinite(output.outcome).all()
    assert report["status"] == "degraded"


def test_resume_uses_durable_response_and_usage_without_another_api_call(tmp_path, monkeypatch):
    from automl_agent.contracts import ModelResponse
    from automl_agent.journal import EventJournal, atomic_write_json
    from automl_agent.providers import OpenAICompatibleProvider

    task = make_task(tmp_path / "dataset")
    directory = tmp_path / "run"
    args = ["--cv-folds", "1", "--family-trials", "3", str(task), "--run-dir", str(directory), "--time-budget", "90", "--max-trials", "2", "--n-jobs", "1"]
    run(parse_config(["--cv-folds", "1", "--family-trials", "3", *args, "--offline", "--family-trials", "1"]))
    state = json.loads((directory / "state.json").read_text())
    state.update(active=True, total_tokens=0, llm_calls=0, checkpoint_epoch=time.time())
    atomic_write_json(directory / "state.json", state)
    journal = EventJournal(directory / "events.jsonl")
    code = "from sklearn.dummy import DummyRegressor\ndef build_model(context):\n    return DummyRegressor()\n"
    response = ModelResponse({"role": "assistant", "tool_calls": [{"id": "paid-call", "type": "function", "function": {
        "name": "run_experiment", "arguments": json.dumps({"hypothesis": "durable paid hypothesis", "code": code})}}]}, total_tokens=42)
    journal.append("llm_request", request_id="request_0001", messages=[], tools=[], max_tokens=256, input_token_bound=100)
    journal.append("llm_response", request_id="request_0001", response=response.to_dict())
    monkeypatch.setenv("OPENAI_MODEL", "fake")
    monkeypatch.setenv("AUTOML_PROVIDER", "openai")
    calls = []
    def forbidden_call(*args, **kwargs):
        calls.append(True)
        raise AssertionError("the response was already paid for and persisted")
    monkeypatch.setattr(OpenAICompatibleProvider, "complete", forbidden_call)
    report = run(parse_config(["--cv-folds", "1", "--family-trials", "3", *args, "--resume"]))
    assert not calls
    assert report["total_tokens"] >= 42
    assert report["llm_calls"] == 1
    assert any(t["hypothesis"] == "durable paid hypothesis" and t["status"] == "ok" for t in report["trials"])


def test_resume_restores_best_predictions_when_refit_budget_is_exhausted(tmp_path):
    from automl_agent.data import discover_task
    from automl_agent.evaluation import publish_submission
    from automl_agent.journal import atomic_write_json

    task = make_task(tmp_path / "dataset")
    directory = tmp_path / "run"
    args = ["--cv-folds", "1", "--family-trials", "3", str(task), "--offline", "--run-dir", str(directory), "--time-budget", "90", "--n-jobs", "1"]
    first = run(parse_config(args))
    candidate = next(t for t in first["trials"] if t["trial_id"] == first["selected_trial"])
    expected = pd.read_csv(candidate["prediction_path"])["outcome"]
    publish_submission(discover_task(task), directory / "fallback_predictions.csv", task.parent / "submission.csv")
    state = json.loads((directory / "state.json").read_text())
    state.update(active=True, elapsed_seconds=90, checkpoint_epoch=time.time())
    atomic_write_json(directory / "state.json", state)
    resumed = run(parse_config(["--cv-folds", "1", "--family-trials", "3", *args, "--resume"]))
    assert resumed["final_fit"]["status"] == "skipped"
    assert np.allclose(pd.read_csv(task.parent / "submission.csv")["outcome"], expected)


def test_usage_above_budget_prevents_dispatch_of_the_returned_tool(tmp_path, monkeypatch):
    from automl_agent.contracts import ModelResponse
    from automl_agent.providers import OpenAICompatibleProvider

    task = make_task(tmp_path / "dataset")
    monkeypatch.setenv("OPENAI_MODEL", "fake")
    monkeypatch.setenv("AUTOML_PROVIDER", "openai")
    def complete(*args, **kwargs):
        return ModelResponse({"role": "assistant", "tool_calls": [{"id": "over-budget", "type": "function", "function": {
            "name": "run_experiment", "arguments": json.dumps({"hypothesis": "must not execute", "code": "from sklearn.dummy import DummyRegressor\ndef build_model(context):\n return DummyRegressor()"})}}]}, total_tokens=100000)
    monkeypatch.setattr(OpenAICompatibleProvider, "complete", complete)
    report = run(parse_config(["--cv-folds", "1", "--family-trials", "3", str(task), "--run-dir", str(tmp_path / "run"), "--time-budget", "90", "--max-trials", "2", "--max-tokens", "30000", "--n-jobs", "1"]))
    assert report["stop_reason"] == "token_budget"
    assert not any(t["hypothesis"] == "must not execute" for t in report["trials"])


def test_finalize_only_refreshes_completed_run_without_research(tmp_path, monkeypatch):
    from automl_agent.agent import ResearchRun

    task = make_task(tmp_path / 'dataset')
    directory = tmp_path / 'run'
    args = [str(task), '--offline', '--run-dir', str(directory), '--time-budget', '90',
            '--cv-folds', '1', '--family-trials', '1', '--n-jobs', '1']
    first = run(parse_config(args))
    before = (directory / 'events.jsonl').read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError('finalize-only must not run experiments or contact LLM')
    monkeypatch.setattr(ResearchRun, '_family_comparison', forbidden)
    monkeypatch.setattr(ResearchRun, 'research', forbidden)
    config = parse_config([*args, '--resume'])
    config.offline = False
    config.finalize_only = True
    refreshed = run(config)
    assert refreshed['final_fit']['status'] == 'ok'
    assert refreshed['final_fit']['artifact_dir'] != first['final_fit']['artifact_dir']
    assert refreshed['stop_reason'] == 'finalize_only'
    assert refreshed['llm_calls'] == first['llm_calls']
    assert refreshed['trials'] == first['trials']
    assert (directory / 'events.jsonl').read_bytes().startswith(before)
