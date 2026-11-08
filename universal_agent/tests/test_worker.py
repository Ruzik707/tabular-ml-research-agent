import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from automl_agent.contracts import TaskSpec
from automl_agent.modeling import baseline_code


def test_worker_evaluates_without_receiving_validation_labels_and_writes_ordered_csv(tmp_path):
    context = {
        "task_type": "regression", "metric": "rmse", "target": "value",
        "feature_columns": ["amount", "kind"], "categorical_columns": ["kind"],
        "classes": [], "seed": 3, "n_jobs": 1,
    }
    train_x = pd.DataFrame({"amount": [1, 2, 3, np.inf, 5, 6], "kind": ["a", "b"] * 3})
    valid_x = pd.DataFrame({"amount": [2, np.nan], "kind": ["unseen", None]})
    test_x = pd.DataFrame({"amount": [7, 8, 9], "kind": ["c", "a", "other"]})
    train_y = pd.DataFrame({"value": [2, 4, 6, 8, 10, 12]})
    inputs = {}
    for name, frame in {"train_features": train_x, "train_target": train_y,
                        "validation_features": valid_x, "test_features": test_x}.items():
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        inputs[name] = str(path)
    task = TaskSpec(
        task_path="/private/task.md", train_path="/private/train.csv", test_path="/private/test.csv",
        target="value", task_type="regression", metric="rmse", maximize=False,
        feature_columns=["amount", "kind"], id_columns=[], categorical_columns=["kind"],
        output_columns=["value"], data_fingerprint="fixture", seed=3,
    )
    code_path = tmp_path / "candidate.py"
    code_path.write_text(baseline_code(task, n_jobs=1, iterations=5), encoding="utf-8")
    artifact_dir = tmp_path / "artifacts"
    manifest = {"mode": "evaluate", "context": context, "inputs": inputs,
                "code_path": str(code_path), "artifact_dir": str(artifact_dir), "save_model": True}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    env = os.environ.copy()
    worktree = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(worktree) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "automl_agent.worker", "--manifest", str(manifest_path)],
        cwd=worktree, env=env, capture_output=True, text=True, timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "validation_target" not in manifest
    valid_predictions = pd.read_csv(artifact_dir / "valid_predictions.csv")
    test_predictions = pd.read_csv(artifact_dir / "test_predictions.csv")
    assert valid_predictions.columns.tolist() == ["value"]
    assert len(valid_predictions) == 2
    assert test_predictions.columns.tolist() == ["value"]
    assert len(test_predictions) == len(test_x)
    assert np.isfinite(test_predictions["value"]).all()
    assert json.loads((artifact_dir / "worker_result.json").read_text())["status"] == "ok"
    assert (artifact_dir / "model.joblib").is_file()

    load_script = (
        "import json, joblib, pandas as pd, sys; "
        "model = joblib.load(sys.argv[1]); "
        "print(json.dumps(model.predict(pd.DataFrame({'amount': [10.0], 'kind': ['new']})).tolist()))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", load_script, str(artifact_dir / "model.joblib")],
        cwd=worktree, env=env, capture_output=True, text=True, timeout=30,
    )
    assert loaded.returncode == 0, loaded.stderr
    assert len(json.loads(loaded.stdout)) == 1


def test_worker_rejects_missing_feature_columns(tmp_path):
    context = {
        "task_type": "regression", "metric": "rmse", "target": "value",
        "feature_columns": ["amount", "kind"], "categorical_columns": ["kind"],
        "classes": [], "seed": 3, "n_jobs": 1,
    }
    train_x = pd.DataFrame({"amount": [1, 2, 3]})
    train_y = pd.DataFrame({"value": [2, 4, 6]})
    valid_x = pd.DataFrame({"amount": [4], "kind": ["a"]})
    test_x = pd.DataFrame({"amount": [5], "kind": ["b"]})
    inputs = {}
    for name, frame in {"train_features": train_x, "train_target": train_y,
                        "validation_features": valid_x, "test_features": test_x}.items():
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        inputs[name] = str(path)
    code_path = tmp_path / "candidate.py"
    code_path.write_text("def build_model(context):\n    raise AssertionError('must reject schema first')\n", encoding="utf-8")
    manifest = {"mode": "evaluate", "context": context, "inputs": inputs,
                "code_path": str(code_path), "artifact_dir": str(tmp_path / "artifacts")}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    worktree = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(worktree) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "automl_agent.worker", "--manifest", str(manifest_path)],
        cwd=worktree, env=env, capture_output=True, text=True, timeout=20,
    )

    assert result.returncode != 0
    assert "missing feature columns" in result.stderr.lower()


def test_worker_writes_probability_classes_for_classification(tmp_path):
    context = {
        "task_type": "classification", "metric": "macro_f1", "target": "label",
        "feature_columns": ["score", "kind"], "categorical_columns": ["kind"],
        "classes": ["amber", "blue", "green"], "seed": 9, "n_jobs": 1,
    }
    train_x = pd.DataFrame({"score": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                            "kind": ["a", "a", "b", "b", "c", "c", "d", "d", "e"]})
    train_y = pd.DataFrame({"label": ["amber", "amber", "blue", "blue", "green", "green", "amber", "blue", "green"]})
    valid_x = pd.DataFrame({"score": [1.5, 4.5, 7.5], "kind": ["new", "c", None]})
    test_x = pd.DataFrame({"score": [9.0, 10.0], "kind": ["other", "a"]})
    inputs = {}
    for name, frame in {"train_features": train_x, "train_target": train_y,
                        "validation_features": valid_x, "test_features": test_x}.items():
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        inputs[name] = str(path)
    task = TaskSpec(
        task_path="/private/task.md", train_path="/private/train.csv", test_path="/private/test.csv",
        target="label", task_type="classification", metric="macro_f1", maximize=True,
        feature_columns=["score", "kind"], id_columns=[], categorical_columns=["kind"],
        output_columns=["label"], data_fingerprint="fixture", seed=9,
        classes=["amber", "blue", "green"],
    )
    code_path = tmp_path / "candidate.py"
    code_path.write_text(baseline_code(task, n_jobs=1, iterations=5), encoding="utf-8")
    artifact_dir = tmp_path / "classification-artifacts"
    manifest = {"mode": "evaluate", "context": context, "inputs": inputs,
                "code_path": str(code_path), "artifact_dir": str(artifact_dir), "save_model": False}
    manifest_path = tmp_path / "classification-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    worktree = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(worktree) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "automl_agent.worker", "--manifest", str(manifest_path)],
        cwd=worktree, env=env, capture_output=True, text=True, timeout=60,
    )

    assert result.returncode == 0, result.stderr
    probability = np.load(artifact_dir / "valid_probabilities.npy")
    probability_classes = json.loads((artifact_dir / "probability_classes.json").read_text())["classes"]
    assert probability.shape == (3, len(probability_classes))
    assert set(probability_classes) == {"amber", "blue", "green"}
    assert np.isfinite(probability).all()
    assert np.allclose(probability.sum(axis=1), 1.0)


def test_worker_rejects_validation_target_in_manifest(tmp_path):
    manifest = {
        "mode": "evaluate", "context": {"validation_target": ["must stay in host"]},
        "inputs": {"validation_target": str(tmp_path / "labels.csv")},
    }
    manifest_path = tmp_path / "invalid-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    worktree = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(worktree) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "automl_agent.worker", "--manifest", str(manifest_path)],
        cwd=worktree, env=env, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode != 0
    assert "validation_target" in result.stderr
