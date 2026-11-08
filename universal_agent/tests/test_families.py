import ast
import importlib.metadata
import importlib.util
import json
import os
import sys
import types

import cloudpickle
import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from automl_agent import families
from automl_agent.tools import validate_candidate_code


def _task(task_type):
    return types.SimpleNamespace(
        task_type=task_type,
        metric="log_loss" if task_type == "classification" else "rmse",
        feature_columns=["number", "category", "category_2"],
        categorical_columns=["category", "category_2"],
        classes=["class_a", "class_b", "class_c"] if task_type == "classification" else [],
        seed=19,
    )


def _profile():
    try:
        catboost = importlib.metadata.version("catboost")
    except importlib.metadata.PackageNotFoundError:
        catboost = "unavailable"
    return {
        "task_type": "classification",
        "metric": "log_loss",
        "train_shape": [87_000, 43],
        "feature_count": 42,
        "categorical_columns": ["category", "category_2"],
        "available_packages": {"catboost": catboost, "scikit-learn": "1.5.2"},
    }


def _family_names(candidates):
    return {candidate["family"] for candidate in candidates}


def test_generators_return_distinct_complete_ast_linted_candidates():
    candidates = families.family_candidates(_task("classification"), _profile(), n_jobs=4)

    expected = {"ExtraTrees", "Linear"}
    if _profile()["available_packages"]["catboost"] != "unavailable":
        expected.add("CatBoost")
    assert len(candidates) == len(expected)
    assert expected == _family_names(candidates)
    for candidate in candidates:
        assert set(candidate) == {"family", "hypothesis", "code", "decision"}
        assert candidate["hypothesis"]
        assert isinstance(candidate["decision"], dict)
        tree = ast.parse(candidate["code"])
        assert any(isinstance(node, ast.FunctionDef) and node.name == "build_model" for node in tree.body)
        validate_candidate_code(candidate["code"])


def test_catboost_uses_incumbent_iterations_and_native_defaults():
    candidate = next((
        item for item in families.family_candidates(_task("classification"), _profile())
        if item["family"] == "CatBoost"
    ), None)
    if candidate is None:
        pytest.skip("CatBoost is not installed")

    assert '"iterations": 150' in candidate["code"]
    assert '"depth":' not in candidate["code"]
    assert '"learning_rate":' not in candidate["code"]
    assert '"MultiClass"' in candidate["code"]
    assert candidate["decision"]["iterations"] == 150
    assert candidate["decision"]["depth"] == "CatBoost default"
    assert candidate["decision"]["learning_rate"] == "CatBoost automatic default"


def test_limit_order_and_cpu_context_are_stable():
    first = families.family_candidates(_task("regression"), _profile(), n_jobs=2, limit=2)
    second = families.family_candidates(_task("regression"), _profile(), n_jobs=2, limit=2)

    assert [item["family"] for item in first] == [item["family"] for item in second]
    assert len(first) == 2
    for candidate in first:
        assert "DEFAULT_N_JOBS = 2" in candidate["code"]


@pytest.mark.parametrize("task_type", ["regression", "classification"])
def test_each_available_family_fits_predicts_and_round_trips(task_type, tmp_path):
    task = _task(task_type)
    candidates = families.family_candidates(task, _profile(), n_jobs=2)
    X = pd.DataFrame({
        "number": [float(i) for i in range(48)],
        "category": ["known_a" if i % 2 else "known_b" for i in range(48)],
        "category_2": ["left" if i % 3 else "right" for i in range(48)],
    })
    X.loc[4, "number"] = np.nan
    X.loc[5, "number"] = np.inf
    X.loc[6, "number"] = -np.inf
    X.loc[7, "category"] = np.nan
    y = np.asarray(["class_a", "class_b", "class_c"] * 16) if task_type == "classification" else np.arange(48, dtype=float)
    validation = pd.DataFrame({
        "number": [1.5, np.inf, np.nan],
        "category": ["never_seen", np.nan, "known_a"],
        "category_2": ["unseen_here", "left", np.nan],
    })

    assert candidates
    for index, candidate in enumerate(candidates):
        module_name = f"test_candidate_{task_type}_{index}"
        module = types.ModuleType(module_name)
        exec(compile(candidate["code"], f"<{module_name}>", "exec"), module.__dict__)
        sys.modules[module_name] = module
        cloudpickle.register_pickle_by_value(module)
        try:
            model = module.build_model({
                "task_type": task_type,
                "metric": task.metric,
                "feature_columns": task.feature_columns,
                "categorical_columns": task.categorical_columns,
                "classes": task.classes,
                "seed": task.seed,
                "n_jobs": 2,
            })
            params = model.get_params(deep=True)
            if candidate["family"] == "Linear" and task_type == "regression":
                assert params["model__alpha"] == 1.0
            else:
                assert any(value == 2 for key, value in params.items() if key.endswith("n_jobs") or key.endswith("thread_count"))
            model.fit(X, y)
            prediction = model.predict(validation)
            assert len(prediction) == len(validation)
            if task_type == "classification":
                probability = model.predict_proba(validation)
                assert probability.shape == (3, 3)
                assert np.isfinite(probability).all()
                assert np.allclose(probability.sum(axis=1), 1.0, atol=1e-6)
            model_path = tmp_path / f"{task_type}-{index}.joblib"
            joblib.dump(model, model_path)
            restored = joblib.load(model_path)
            assert np.array_equal(restored.predict(validation), prediction)
        finally:
            cloudpickle.unregister_pickle_by_value(module)
            sys.modules.pop(module_name, None)


def test_unavailable_catboost_is_skipped_without_a_catboost_fallback(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    profile = _profile()
    profile["available_packages"] = {"scikit-learn": "1.5.2"}

    candidates = families.family_candidates(_task("regression"), profile)

    assert "CatBoost" not in _family_names(candidates)
    assert all("CatBoost" not in item["code"] for item in candidates)


def test_catboost_is_never_included_when_profile_marks_it_unavailable(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    profile = _profile()
    profile["available_packages"] = {"catboost": "unavailable"}

    candidates = families.family_candidates(_task("classification"), profile)

    assert "CatBoost" not in _family_names(candidates)


def test_linear_categorical_transform_stays_sparse_for_many_categories():
    candidate = next(
        item for item in families.family_candidates(_task("classification"), _profile())
        if item["family"] == "Linear"
    )
    module = types.ModuleType("test_linear_sparse_candidate")
    exec(compile(candidate["code"], "<test_linear_sparse_candidate>", "exec"), module.__dict__)
    X = pd.DataFrame({
        "number": np.arange(600, dtype=float),
        "category": [f"category_{index % 20}" for index in range(600)],
        "category_2": [f"other_{(index // 3) % 20}" for index in range(600)],
    })
    model = module.build_model({
        "task_type": "classification",
        "feature_columns": _task("classification").feature_columns,
        "categorical_columns": _task("classification").categorical_columns,
        "seed": 19,
        "n_jobs": 2,
    })

    transformed = model.named_steps["preprocess"].fit_transform(X)

    assert sparse.issparse(transformed)
    assert transformed.shape[1] > 20


@pytest.mark.parametrize("family", ["ExtraTrees", "Linear"])
@pytest.mark.parametrize("task_type", ["regression", "classification"])
def test_sklearn_families_handle_mixed_categorical_types_and_infinities(family, task_type):
    task = _task(task_type)
    candidate = next(
        item for item in families.family_candidates(task, _profile())
        if item["family"] == family
    )
    module = types.ModuleType(f"test_mixed_categories_{family}_{task_type}")
    exec(compile(candidate["code"], f"<{module.__name__}>", "exec"), module.__dict__)
    category_values = ["alpha", 7, np.inf, None, -np.inf, 3.5]
    X = pd.DataFrame({
        "number": np.arange(72, dtype=float),
        "category": [category_values[index % len(category_values)] for index in range(72)],
        "category_2": ["left" if index % 2 else None for index in range(72)],
    })
    y = (
        np.random.default_rng(19).permutation(np.repeat(["class_a", "class_b", "class_c"], 24))
        if task_type == "classification" else np.arange(72, dtype=float)
    )
    model = module.build_model({
        "task_type": task_type,
        "metric": task.metric,
        "feature_columns": task.feature_columns,
        "categorical_columns": task.categorical_columns,
        "classes": task.classes,
        "seed": task.seed,
        "n_jobs": 2,
    })

    model.fit(X, y)
    validation = pd.DataFrame({
        "number": [1.5, np.nan, 9.0],
        "category": [42, np.inf, None],
        "category_2": ["unseen", -np.inf, None],
    })
    prediction = model.predict(validation)

    assert len(prediction) == len(validation)
    encoder = model.named_steps["preprocess"].named_transformers_["categorical"].named_steps["encode"]
    observed_categories = set(encoder.categories_[0])
    assert {"7", "3.5", "inf", "-inf", "__missing__"} <= observed_categories


@pytest.mark.parametrize("task_type", ["regression", "classification"])
def test_worker_can_save_and_reload_generated_candidate(task_type, tmp_path):
    from automl_agent.worker import _run_manifest

    if importlib.util.find_spec("catboost") is None:
        pytest.skip("CatBoost is not installed")
    task = _task(task_type)
    candidate = next(
        item for item in families.family_candidates(task, _profile(), n_jobs=2)
        if item["family"] == "CatBoost"
    )
    X = pd.DataFrame({
        "number": np.arange(60, dtype=float),
        "category": ["known_a" if index % 2 else "known_b" for index in range(60)],
        "category_2": ["left" if index % 3 else "right" for index in range(60)],
    })
    X.loc[4, "number"] = np.inf
    X.loc[5, "category"] = np.nan
    y = pd.DataFrame({
        "target": np.asarray(["class_a", "class_b", "class_c"] * 20)
        if task_type == "classification" else np.arange(60, dtype=float)
    })
    paths = {name: tmp_path / f"{name}.csv" for name in ("train_features", "train_target", "validation_features", "test_features")}
    X.iloc[:48].to_csv(paths["train_features"], index=False)
    y.iloc[:48].to_csv(paths["train_target"], index=False)
    X.iloc[48:54].to_csv(paths["validation_features"], index=False)
    X.iloc[54:].to_csv(paths["test_features"], index=False)
    code_path = tmp_path / "candidate.py"
    code_path.write_text(candidate["code"], encoding="utf-8")
    artifact_dir = tmp_path / "artifacts"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "mode": "evaluate",
        "inputs": {name: str(path) for name, path in paths.items()},
        "context": {
            "target": "target",
            "task_type": task_type,
            "metric": task.metric,
            "feature_columns": task.feature_columns,
            "categorical_columns": task.categorical_columns,
            "classes": task.classes,
            "seed": task.seed,
            "n_jobs": 2,
        },
        "code_path": str(code_path),
        "artifact_dir": str(artifact_dir),
        "save_model": True,
    }), encoding="utf-8")

    result = _run_manifest(manifest_path)
    generated_module_name = f"automl_candidate_{os.getpid()}"
    sys.modules.pop(generated_module_name, None)
    restored = joblib.load(artifact_dir / "model.joblib")
    validation = pd.read_csv(paths["validation_features"])

    assert result["status"] == "ok"
    assert len(restored.predict(validation)) == len(validation)
    if task_type == "classification":
        probabilities = restored.predict_proba(validation)
        assert probabilities.shape == (len(validation), 3)
        assert np.isfinite(probabilities).all()
