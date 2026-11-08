import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from universal_agent.scripts.benchmark_reference import REFERENCE_SPEC, run_workspace


def _write_case(workspace: Path, *, name: str, task_type: str, target: str, metric: str):
    task_dir = workspace / "inputs" / name
    evaluator_dir = workspace / "evaluator" / name
    task_dir.mkdir(parents=True)
    evaluator_dir.mkdir(parents=True)
    ids = np.asarray([f"{name}-{index:04d}" for index in range(28)], dtype=object)
    x = np.arange(28, dtype=float)
    category = np.where(x.astype(int) % 2, "odd", "even")
    if task_type == "regression":
        target_values = 2.5 * x + np.where(category == "odd", 3.0, -2.0)
        train_index, test_index = np.arange(20), np.arange(20, 28)
        classes = []
        positive_class = None
    else:
        target_values = np.where(x < 14, "alert", "normal")
        train_index, test_index = np.r_[0:10, 14:24], np.r_[10:14, 24:28]
        classes = ["alert", "normal"]
        positive_class = "alert"

    frame = pd.DataFrame({"row_id": ids, "numeric": x, "category": category, target: target_values})
    train_frame = frame.iloc[train_index].copy()
    if task_type == "regression":
        train_frame = train_frame[["numeric", "category", target, "row_id"]]
    train_frame.to_csv(task_dir / "train.csv", index=False)
    frame.iloc[test_index].drop(columns=[target]).to_csv(task_dir / "test.csv", index=False)
    sample = frame.iloc[test_index][["row_id"]].copy()
    sample[target] = 0.0 if task_type == "regression" else "alert"
    sample.to_csv(task_dir / "sample_submission.csv", index=False)
    pd.DataFrame({"row_id": ids[test_index], target: target_values[test_index]}).to_csv(
        evaluator_dir / "truth.csv", index=False
    )
    (task_dir / "data_dictionary.csv").write_text("feature,role,dtype\n", encoding="utf-8")
    (task_dir / "task.md").write_text("tiny reference task\n", encoding="utf-8")
    hashes = {
        filename: hashlib.sha256((task_dir / filename).read_bytes()).hexdigest()
        for filename in ["train.csv", "test.csv", "sample_submission.csv", "data_dictionary.csv", "task.md"]
    }
    return {
        "name": name,
        "target": target,
        "task_type": task_type,
        "metric": metric,
        "maximize": task_type == "classification",
        "positive_class": positive_class,
        "train_rows": len(train_index),
        "test_rows": len(test_index),
        "feature_columns": ["numeric", "category"],
        "id_columns": ["row_id"],
        "classes": classes,
        "split_protocol": "test fixture deterministic split",
        "task_dir": str(task_dir),
        "truth_path": str(evaluator_dir / "truth.csv"),
        "input_hashes": hashes,
        "split_fingerprint": hashlib.sha256(name.encode()).hexdigest(),
    }


def _workspace(tmp_path):
    cases = [
        _write_case(tmp_path, name="tiny-regression", task_type="regression", target="value", metric="rmse"),
        _write_case(tmp_path, name="tiny-binary", task_type="classification", target="label", metric="f1"),
    ]
    (tmp_path / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    return tmp_path


def test_reference_specs_are_fixed_and_explicit():
    assert set(REFERENCE_SPEC) == {"dummy", "linear"}
    assert REFERENCE_SPEC["dummy"]["regression"] == {"estimator": "DummyRegressor", "strategy": "mean"}
    assert REFERENCE_SPEC["dummy"]["classification"] == {"estimator": "DummyClassifier", "strategy": "most_frequent"}
    assert REFERENCE_SPEC["linear"]["regression"]["estimator"] == "Pipeline(StandardScaler+Ridge)"
    assert REFERENCE_SPEC["linear"]["regression"]["ridge_alpha"] == 1.0
    assert REFERENCE_SPEC["linear"]["classification"]["estimator"] == "Pipeline(StandardScaler+LogisticRegression)"
    assert REFERENCE_SPEC["linear"]["classification"]["logistic_C"] == 1.0
    assert REFERENCE_SPEC["linear"]["classification"]["class_weight"] is None


def test_run_workspace_scores_fixed_references_and_records_hashes_without_truth(tmp_path):
    workspace = _workspace(tmp_path)
    result = run_workspace(workspace, threads=2)
    assert result["run_status"] == "completed"
    assert result["threads"] == 2
    assert result["source_sha256"]
    assert result["data_module_sha256"]
    assert {case["name"] for case in result["cases"]} == {"tiny-regression", "tiny-binary"}
    for case_result in result["cases"]:
        assert case_result["split_fingerprint"]
        assert case_result["input_hashes"]
        assert {run["model"] for run in case_result["runs"]} == {"dummy", "linear"}
        for run in case_result["runs"]:
            assert run["status"] == "completed"
            assert run["valid"] is True
            assert run["score"] is not None
            assert run["rows"] == 8
            assert Path(run["submission_path"]).is_file()
            assert "truth" not in run
            assert "labels" not in run
    reg = next(case for case in result["cases"] if case["name"] == "tiny-regression")
    assert next(run for run in reg["runs"] if run["model"] == "linear")["score"] < next(
        run for run in reg["runs"] if run["model"] == "dummy"
    )["score"]
    saved = json.loads((workspace / "reference_results.json").read_text())
    assert saved["source_sha256"] == result["source_sha256"]


def test_reference_rejects_mutated_task_inputs_before_fitting(tmp_path):
    workspace = _workspace(tmp_path)
    case = json.loads((workspace / "cases.json").read_text())[0]
    with (Path(case["task_dir"]) / "test.csv").open("a", encoding="utf-8") as stream:
        stream.write("bad,row\n")
    result = run_workspace(workspace, threads=2)
    assert result["run_status"] == "partial_failure"
    regression = next(item for item in result["cases"] if item["name"] == "tiny-regression")
    assert all(run["status"] == "input_hash_mismatch" for run in regression["runs"])


@pytest.mark.parametrize("failure", ["timeout", "invalid_submission", "worker_error", "deadline_exhausted"])
def test_reference_aggregate_status_does_not_hide_individual_failure(tmp_path, monkeypatch, failure):
    from universal_agent.scripts import benchmark_reference
    workspace = _workspace(tmp_path)

    def simulated_run(case, model, *args):
        if case["name"] == "tiny-binary" and model == "linear":
            return {"model": model, "status": failure, "valid": False, "score": None}
        return {"model": model, "status": "completed", "valid": True, "score": 1.0}

    monkeypatch.setattr(benchmark_reference, "_run_one_model", simulated_run)
    result = run_workspace(workspace)
    assert result["run_status"] == "partial_failure"
    assert json.loads((workspace / "reference_results.json").read_text())["run_status"] == "partial_failure"
    assert result["cases"][1]["runs"][1]["status"] == failure


def test_reference_cli_accepts_workspace_and_threads_and_writes_results(tmp_path):
    workspace = _workspace(tmp_path)
    script = Path(__file__).parents[1] / "scripts" / "benchmark_reference.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--workspace", str(workspace), "--threads", "2"],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert (workspace / "reference_results.json").is_file()
    output = json.loads((workspace / "reference_results.json").read_text())
    assert output["threads"] == 2
