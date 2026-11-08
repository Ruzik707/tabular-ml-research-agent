"""Regression coverage for safe, fitted estimator parameter metadata."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl_agent import worker as worker_module
from automl_agent.worker import (
    _UNSAFE_PARAMETER,
    _bounded_parameter_value,
    _estimator_details,
)


WORKTREE = Path(__file__).resolve().parents[1]


def _fit_catboost_classifier():
    catboost = pytest.importorskip("catboost")
    model = catboost.CatBoostClassifier(
        iterations=5,
        auto_class_weights="Balanced",
        random_seed=17,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        np.arange(12, dtype=float).reshape(-1, 1),
        np.array(["rare"] * 3 + ["common"] * 9),
    )
    return model


def test_fitted_catboost_reports_bounded_real_auto_parameters():
    model = _fit_catboost_classifier()

    details = _estimator_details(model)
    actual = model.get_all_params()
    parameters = details["parameters"]

    assert details["parameters_source"].startswith("catboost.get_all_params")
    assert parameters["iterations"] == actual["iterations"] == 5
    assert parameters["learning_rate"] == actual["learning_rate"]
    assert parameters["depth"] == actual["depth"]
    assert parameters["l2_leaf_reg"] == actual["l2_leaf_reg"]
    assert parameters["border_count"] == actual["border_count"]
    assert parameters["auto_class_weights"] == actual["auto_class_weights"] == "Balanced"
    assert parameters["class_weights"] == actual["class_weights"]
    assert parameters["class_weights"] is not None
    assert parameters["thread_count"] == 1
    assert "thread_count=catboost._init_params" in details["parameters_source"]
    assert "thread_count" not in actual
    assert model.__dict__["_init_params"]["thread_count"] == 1
    assert len(parameters) <= 8
    assert "rare" not in json.dumps(parameters)
    assert "common" not in json.dumps(parameters)


def test_fitted_sklearn_parameters_come_from_constructor_state(monkeypatch):
    from sklearn.base import BaseEstimator
    from sklearn.ensemble import ExtraTreesClassifier

    model = ExtraTreesClassifier(
        n_estimators=7,
        max_depth=4,
        max_leaf_nodes=13,
        min_samples_leaf=2,
        max_features=0.75,
        class_weight="balanced",
        n_jobs=1,
        random_state=9,
    ).fit(np.arange(24).reshape(-1, 1), np.array([0] * 6 + [1] * 18))

    def forbidden_get_params(self, deep=True):
        raise AssertionError("metadata must read bounded constructor state directly")

    monkeypatch.setattr(BaseEstimator, "get_params", forbidden_get_params)
    details = _estimator_details(model)

    assert details["parameters_source"] == "sklearn.constructor_params"
    assert details["parameters"] == {
        "n_estimators": 7,
        "max_depth": 4,
        "max_leaf_nodes": 13,
        "min_samples_leaf": 2,
        "max_features": 0.75,
        "class_weight": "balanced",
        "n_jobs": 1,
    }


def test_fitted_linear_parameters_include_solver_and_thread_count(monkeypatch):
    from sklearn.base import BaseEstimator
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        C=0.7,
        max_iter=240,
        solver="saga",
        class_weight="balanced",
        n_jobs=2,
        random_state=9,
    ).fit(
        np.linspace(-1.0, 1.0, 40).reshape(-1, 1),
        np.array(["a"] * 20 + ["b"] * 20),
    )

    def forbidden_get_params(self, deep=True):
        raise AssertionError("metadata must read bounded constructor state directly")

    monkeypatch.setattr(BaseEstimator, "get_params", forbidden_get_params)
    details = _estimator_details(model)

    assert details["parameters_source"] == "sklearn.constructor_params"
    assert details["parameters"] == {
        "C": 0.7,
        "max_iter": 240,
        "solver": "saga",
        "class_weight": "balanced",
        "n_jobs": 2,
    }


def test_fitted_linear_regression_reports_safe_constructor_fields():
    from sklearn.linear_model import LinearRegression

    model = LinearRegression(
        n_jobs=2,
        fit_intercept=False,
        positive=True,
    ).fit(np.arange(12, dtype=float).reshape(-1, 1), np.arange(12, dtype=float))
    details = _estimator_details(model)

    assert details["parameters_source"] == "sklearn.constructor_params"
    assert details["parameters"] == {
        "n_jobs": 2,
        "fit_intercept": False,
        "positive": True,
    }


def test_nested_and_cyclic_parameter_containers_are_rejected():
    nested = [1.0]
    for _ in range(20):
        nested = [nested]
    cyclic = []
    cyclic.append(cyclic)

    assert _bounded_parameter_value(nested, "class_weights") is _UNSAFE_PARAMETER
    assert _bounded_parameter_value(cyclic, "class_weights") is _UNSAFE_PARAMETER


def test_catboost_omits_parameters_absent_from_native_api(monkeypatch):
    catboost = pytest.importorskip("catboost")
    estimator_class = catboost.CatBoostClassifier
    registry = worker_module._capture_native_parameter_types()
    family, fields, _getter = registry[estimator_class]
    monkeypatch.setitem(
        registry,
        estimator_class,
        (family, fields, lambda _model: {"iterations": 5}),
    )

    details = _estimator_details(estimator_class(iterations=5))

    assert details["parameters"] == {"iterations": 5}
    assert details["parameters_source"] == "catboost.get_all_params"


def test_custom_estimator_getters_and_repr_are_never_called():
    calls = []

    class HostileEstimator:
        @property
        def get_params(self):
            calls.append("get_params")
            raise AssertionError("must not call custom get_params")

        @property
        def get_all_params(self):
            calls.append("get_all_params")
            raise AssertionError("must not call custom get_all_params")

        def __repr__(self):
            calls.append("repr")
            raise AssertionError("must not call custom repr")

    details = _estimator_details(HostileEstimator())

    assert details["parameters"] == {}
    assert details["parameters_source"] == "unsupported_estimator"
    assert calls == []


def _run_catboost_worker(tmp_path: Path) -> tuple[dict, Path]:
    pytest.importorskip("catboost")
    rows = 12
    frames = {
        "train_features": pd.DataFrame({"feature": np.arange(rows, dtype=float)}),
        "train_target": pd.DataFrame({"label": ["rare"] * 3 + ["common"] * 9}),
        "validation_features": pd.DataFrame({"feature": [12.0, 13.0]}),
        "test_features": pd.DataFrame({"feature": [14.0, 15.0]}),
    }
    inputs = {}
    for name, frame in frames.items():
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        inputs[name] = str(path)

    code_path = tmp_path / "candidate.py"
    code_path.write_text(
        "from catboost import CatBoostClassifier\n\n"
        "def build_model(context):\n"
        "    return CatBoostClassifier(\n"
        "        iterations=5, auto_class_weights='Balanced', random_seed=17,\n"
        "        thread_count=1, verbose=False, allow_writing_files=False\n"
        "    )\n",
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "artifacts"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "mode": "evaluate",
        "context": {
            "task_type": "classification",
            "metric": "macro_f1",
            "target": "label",
            "feature_columns": ["feature"],
            "categorical_columns": [],
            "seed": 17,
            "n_jobs": 1,
        },
        "inputs": inputs,
        "code_path": str(code_path),
        "artifact_dir": str(artifact_dir),
        "save_model": True,
    }), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKTREE) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "automl_agent.worker", "--manifest", str(manifest_path)],
        cwd=WORKTREE,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads((artifact_dir / "worker_result.json").read_text(encoding="utf-8")), artifact_dir


def test_worker_json_and_reloaded_catboost_model_share_real_parameters(tmp_path):
    result, artifact_dir = _run_catboost_worker(tmp_path)
    import joblib

    loaded_model = joblib.load(artifact_dir / "model.joblib")
    actual = loaded_model.get_all_params()
    parameters = result["estimator_details"]["parameters"]

    assert result["status"] == "ok"
    assert result["estimator_details"]["parameters_source"].startswith("catboost.get_all_params")
    assert parameters["learning_rate"] == actual["learning_rate"]
    assert parameters["auto_class_weights"] == actual["auto_class_weights"] == "Balanced"
    assert parameters["class_weights"] == actual["class_weights"]
    assert parameters["class_weights"] is not None
    assert parameters["thread_count"] == 1
    assert "thread_count=catboost._init_params" in result["estimator_details"]["parameters_source"]
    encoded = json.dumps(parameters)
    assert "rare" not in encoded
    assert "common" not in encoded
