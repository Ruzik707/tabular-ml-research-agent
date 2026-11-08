"""Tests for safe reporting of the estimator that was actually fitted."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from automl_agent.contracts import TaskSpec
from automl_agent.modeling import baseline_code
from automl_agent.worker import _estimator_details


WORKTREE = Path(__file__).resolve().parents[1]


def _run_worker(tmp_path: Path, code: str, mode: str = "evaluate", *, save_model: bool = False) -> dict:
    train_x = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    train_y = pd.DataFrame({"target": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    test_x = pd.DataFrame({"feature": [6.0, 7.0]})
    frames = {"train_features": train_x, "train_target": train_y, "test_features": test_x}
    if mode == "evaluate":
        frames["validation_features"] = pd.DataFrame({"feature": [1.5, 4.5]})

    inputs = {}
    for name, frame in frames.items():
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        inputs[name] = str(path)

    code_path = tmp_path / "candidate.py"
    code_path.write_text(code, encoding="utf-8")
    artifact_dir = tmp_path / "artifacts"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "mode": mode,
        "context": {
            "task_type": "regression",
            "metric": "rmse",
            "target": "target",
            "feature_columns": ["feature"],
            "categorical_columns": [],
            "seed": 5,
            "n_jobs": 1,
        },
        "inputs": inputs,
        "code_path": str(code_path),
        "artifact_dir": str(artifact_dir),
        "save_model": save_model,
    }), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKTREE) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "automl_agent.worker", "--manifest", str(manifest_path)],
        cwd=WORKTREE,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert (artifact_dir / "test_predictions.csv").is_file()
    if save_model:
        assert (artifact_dir / "model.joblib").is_file()
    return json.loads((artifact_dir / "worker_result.json").read_text(encoding="utf-8"))


def _task_spec() -> TaskSpec:
    return TaskSpec(
        task_path="/private/task.md",
        train_path="/private/train.csv",
        test_path="/private/test.csv",
        target="target",
        task_type="regression",
        metric="rmse",
        maximize=False,
        feature_columns=["feature"],
        id_columns=[],
        categorical_columns=[],
        output_columns=["target"],
        data_fingerprint="fixture",
        seed=5,
    )


def test_adaptive_wrapper_reports_its_fitted_catboost_backend(tmp_path):
    result = _run_worker(tmp_path, baseline_code(_task_spec(), n_jobs=1, iterations=3))

    details = result["estimator_details"]
    assert details["class"].endswith("._AdaptiveModel")
    assert result["model_class"] == details["class"]
    assert any(name.startswith("catboost.") for name in details["estimator_chain"])
    assert details["estimator_chain"][-1] == "catboost.core.CatBoostRegressor"
    assert details["pipeline_steps"] == []


@pytest.mark.parametrize("mode", ["evaluate", "finalize"])
def test_pipeline_metadata_reports_real_fitted_predictor_in_both_modes(tmp_path, mode):
    code = '''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

def build_model(context):
    return Pipeline([
        ("scale", StandardScaler()),
        ("predictor", DecisionTreeRegressor(random_state=3)),
    ])
'''
    result = _run_worker(tmp_path, code, mode=mode)

    details = result["estimator_details"]
    assert details["class"] == "sklearn.pipeline.Pipeline"
    assert "sklearn.tree._classes.DecisionTreeRegressor" in details["estimator_chain"]
    assert details["pipeline_steps"] == [
        {"name": "scale", "class": "sklearn.preprocessing._data.StandardScaler"},
        {"name": "predictor", "class": "sklearn.tree._classes.DecisionTreeRegressor"},
    ]


def test_metadata_ignores_raising_properties_and_stops_on_cycles(tmp_path):
    code = '''
from sklearn.tree import DecisionTreeRegressor

class CyclicWrapper:
    def __init__(self):
        self._model = DecisionTreeRegressor(random_state=3)
        self.estimator = self

    @property
    def model(self):
        raise RuntimeError("metadata must not evaluate model property")

    @property
    def base_estimator(self):
        raise RuntimeError("metadata must not evaluate estimator property")

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

def build_model(context):
    return CyclicWrapper()
'''
    result = _run_worker(tmp_path, code, save_model=True)

    details = result["estimator_details"]
    assert details["class"].endswith(".CyclicWrapper")
    assert "sklearn.tree._classes.DecisionTreeRegressor" in details["estimator_chain"]
    assert len(details["estimator_chain"]) < 10


def test_metadata_traversal_has_depth_and_node_limits():
    class Node:
        pass

    root = Node()
    current = root
    for _ in range(100):
        child = Node()
        current.model = child
        current = child
    assert len(_estimator_details(root)["estimator_chain"]) == 7

    wide = Pipeline([(f"scale_{index}", StandardScaler()) for index in range(100)])
    details = _estimator_details(wide)
    assert len(details["estimator_chain"]) == 32
    assert len(details["pipeline_steps"]) == 32
