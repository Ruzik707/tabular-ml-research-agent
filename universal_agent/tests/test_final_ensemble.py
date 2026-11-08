import json

import joblib
import numpy as np
import pandas as pd
import pytest

from automl_agent.final_ensemble import run_final_ensemble


def _write_fixture(tmp_path, *, task_type="classification", classes=None):
    rng = np.random.default_rng(21)
    rows = 48
    x = rng.normal(size=(rows, 2))
    if task_type == "classification":
        y = np.where(x[:, 0] + 0.3 * x[:, 1] > 0, "zebra", "apple")
    else:
        y = 2.5 * x[:, 0] - x[:, 1] + 0.2
    test = pd.DataFrame({"x1": [-1.3, 0.0, 1.4], "x2": [0.2, -0.5, 1.0]})
    train = pd.DataFrame(x, columns=["x1", "x2"])
    target = pd.DataFrame({"target": y})
    paths = {}
    for key, frame in (("train_features", train), ("train_target", target), ("test_features", test)):
        path = tmp_path / f"{key}.csv"
        frame.to_csv(path, index=False)
        paths[key] = str(path)
    context = {
        "task_type": task_type,
        "target": "target",
        "feature_columns": ["x1", "x2"],
        "categorical_columns": [],
        "classes": classes if classes is not None else (["zebra", "apple"] if task_type == "classification" else []),
        "seed": 21,
        "n_jobs": 1,
    }
    return paths, context, test


def _candidate(tmp_path, filename, code):
    path = tmp_path / filename
    path.write_text(code, encoding="utf-8")
    return str(path)


def test_final_classifier_refits_weighted_catboost_and_extratrees_and_reload(tmp_path):
    inputs, context, test = _write_fixture(tmp_path)
    cat_code = _candidate(
        tmp_path,
        "cat_candidate.py",
        "from catboost import CatBoostClassifier\n"
        "def build_model(context):\n"
        "    return CatBoostClassifier(iterations=18, depth=2, learning_rate=0.15, "
        "verbose=False, thread_count=1, random_seed=21)\n",
    )
    trees_code = _candidate(
        tmp_path,
        "trees_candidate.py",
        "from sklearn.ensemble import ExtraTreesClassifier\n"
        "def build_model(context):\n"
        "    return ExtraTreesClassifier(n_estimators=11, max_depth=4, random_state=21, n_jobs=1)\n",
    )
    artifact_dir = tmp_path / "classifier_artifacts"
    manifest = {
        "context": context,
        "inputs": inputs,
        "artifact_dir": str(artifact_dir),
        "save_model": True,
        "members": [
            {"code_path": cat_code, "fit_options": {"sample_weight": "balanced", "early_stopping_rounds": 8},
             "iteration_override": 7, "trial_id": "trial_cat"},
            {"code_path": trees_code, "fit_options": {"sample_weight": "balanced", "early_stopping_rounds": 8},
             "iteration_override": 7, "trial_id": "trial_trees"},
        ],
        "weights": [0.35, 0.65],
        "class_multipliers": [1.4, 0.8],
    }

    result = run_final_ensemble(manifest)

    assert result["status"] == "ok"
    assert result["mode"] == "finalize_ensemble"
    details = result["estimator_details"]
    assert details["parameters"]["weights"] == [0.35, 0.65]
    assert details["parameters"]["class_multipliers"] == [1.4, 0.8]
    assert [row["trial_id"] for row in details["members"]] == ["trial_cat", "trial_trees"]
    assert details["members"][0]["training"]["best_iteration"] == 7
    assert details["members"][0]["training"]["sample_weight_mode"] == "balanced"
    assert details["members"][1]["training"]["reason"] == "iteration_override_ignored_non_booster"
    assert details["training"]["train_rows"] == 48

    output = pd.read_csv(artifact_dir / "test_predictions.csv")
    assert output.columns.tolist() == ["target"]
    assert len(output) == len(test)
    probabilities = np.load(artifact_dir / "test_probabilities.npy")
    assert probabilities.shape == (len(test), 2)
    assert json.loads((artifact_dir / "probability_classes.json").read_text())["classes"] == ["zebra", "apple"]
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    restored = joblib.load(artifact_dir / "model.joblib")
    assert restored.classes_.tolist() == ["zebra", "apple"]
    assert np.array_equal(restored.predict_proba(test), probabilities)
    assert np.array_equal(restored.predict(test), output["target"].to_numpy())
    saved_result = json.loads((artifact_dir / "worker_result.json").read_text())
    assert saved_result["estimator_details"] == details


def test_final_regression_blend_serialization_reproduces_predictions(tmp_path):
    inputs, context, test = _write_fixture(tmp_path, task_type="regression")
    first = _candidate(
        tmp_path,
        "reg_a.py",
        "from sklearn.ensemble import ExtraTreesRegressor\n"
        "def build_model(context):\n"
        "    return ExtraTreesRegressor(n_estimators=9, max_depth=5, random_state=7, n_jobs=1)\n",
    )
    second = _candidate(
        tmp_path,
        "reg_b.py",
        "from sklearn.ensemble import ExtraTreesRegressor\n"
        "def build_model(context):\n"
        "    return ExtraTreesRegressor(n_estimators=13, max_depth=3, random_state=8, n_jobs=1)\n",
    )
    artifact_dir = tmp_path / "regression_artifacts"
    result = run_final_ensemble({
        "context": context,
        "inputs": inputs,
        "artifact_dir": str(artifact_dir),
        "save_model": True,
        "members": [
            {"code_path": first, "fit_options": {}, "iteration_override": 7, "trial_id": "a"},
            {"code_path": second, "fit_options": {}, "iteration_override": None, "trial_id": "b"},
        ],
        "weights": [0.25, 0.75],
        "class_multipliers": None,
    })

    predicted = pd.read_csv(artifact_dir / "test_predictions.csv", float_precision="round_trip")["target"].to_numpy()
    restored = joblib.load(artifact_dir / "model.joblib")
    assert result["estimator_details"]["training"]["train_rows"] == 48
    assert result["estimator_details"]["training"]["final_iterations"] == [None, None]
    assert len(predicted) == len(test)
    assert np.isfinite(predicted).all()
    assert np.array_equal(restored.predict(test), predicted)
    assert not (artifact_dir / "test_probabilities.npy").exists()


@pytest.mark.parametrize(
    "weights, multipliers",
    [([1.0], None), ([0.4, 0.4], None), ([float("nan"), 1.0], None), ([0.5, 0.5], [1.0])],
)
def test_invalid_ensemble_axes_and_weights_rejected_before_artifacts(tmp_path, weights, multipliers):
    inputs, context, _ = _write_fixture(tmp_path)
    code = _candidate(
        tmp_path,
        "unused_candidate.py",
        "from sklearn.ensemble import ExtraTreesClassifier\n"
        "def build_model(context):\n"
        "    return ExtraTreesClassifier(n_estimators=3, n_jobs=1)\n",
    )
    artifact_dir = tmp_path / "invalid_artifacts"
    manifest = {
        "context": context,
        "inputs": inputs,
        "artifact_dir": str(artifact_dir),
        "save_model": True,
        "members": [
            {"code_path": code, "fit_options": {}, "iteration_override": None, "trial_id": "a"},
            {"code_path": code, "fit_options": {}, "iteration_override": None, "trial_id": "b"},
        ],
        "weights": weights,
        "class_multipliers": multipliers,
    }
    with pytest.raises((TypeError, ValueError)):
        run_final_ensemble(manifest)
    assert not (artifact_dir / "worker_result.json").exists()
    assert not (artifact_dir / "model.joblib").exists()


@pytest.mark.parametrize("invalid_input", ["missing_feature", "missing_target", "validation_labels"])
def test_final_refit_rejects_invalid_schema_or_label_leakage_before_candidate_load(tmp_path, invalid_input):
    inputs, context, _ = _write_fixture(tmp_path)
    if invalid_input == "missing_feature":
        pd.DataFrame({"x1": [1.0]}).to_csv(inputs["test_features"], index=False)
    elif invalid_input == "missing_target":
        target = pd.read_csv(inputs["train_target"])
        target.loc[0, "target"] = np.nan
        target.to_csv(inputs["train_target"], index=False)
    else:
        inputs["validation_target"] = inputs["train_target"]
    code = _candidate(
        tmp_path,
        "schema_guard_candidate.py",
        "def build_model(context):\n"
        "    raise AssertionError('candidate code should not load for an invalid manifest')\n",
    )
    manifest = {
        "context": context,
        "inputs": inputs,
        "artifact_dir": str(tmp_path / "schema_artifacts"),
        "save_model": True,
        "members": [{"code_path": code, "fit_options": {}, "trial_id": "guard"}],
        "weights": [1.0],
        "class_multipliers": None,
    }
    with pytest.raises(ValueError):
        run_final_ensemble(manifest)
    assert not (tmp_path / "schema_artifacts" / "worker_result.json").exists()


@pytest.mark.parametrize("threshold", [0., .3, 1.])
def test_final_binary_threshold_preserves_probability_axis_and_survives_reload(tmp_path, threshold):
    inputs, context, _ = _write_fixture(tmp_path)
    context.update(metric="f1", positive_class="zebra")
    test = pd.DataFrame({"x1": [.3, .2, .4], "x2": [0., 0., 0.]})
    test.to_csv(inputs["test_features"], index=False)
    code = _candidate(tmp_path, "probability_candidate.py", '''import numpy as np
class Candidate:
    def fit(self, X, y):
        self.classes_ = np.array(['zebra', 'apple'])
        return self
    def predict_proba(self, X):
        p = X['x1'].clip(0, 1).to_numpy()
        return np.column_stack([p, 1-p])
    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
def build_model(context):
    return Candidate()
''')
    artifact = tmp_path / "binary_final"
    result = run_final_ensemble({
        "context": context, "inputs": inputs, "artifact_dir": str(artifact), "save_model": True,
        "members": [{"code_path": code, "fit_options": {}, "trial_id": "binary"}],
        "weights": [1.], "class_multipliers": None,
        "binary_threshold": threshold, "positive_class": "zebra",
    })
    expected = np.where(test["x1"].to_numpy() >= threshold, "zebra", "apple")
    actual = pd.read_csv(artifact / "test_predictions.csv")["target"].to_numpy()
    np.testing.assert_array_equal(actual, expected)
    restored = joblib.load(artifact / "model.joblib")
    np.testing.assert_array_equal(restored.predict(test), expected)
    np.testing.assert_array_equal(restored.predict_proba(test), np.column_stack([test["x1"], 1-test["x1"]]))
    np.testing.assert_array_equal(np.load(artifact / "test_probabilities.npy"), restored.predict_proba(test))
    assert result["estimator_details"]["parameters"]["binary_threshold"] == threshold
    assert result["estimator_details"]["parameters"]["positive_class"] == "zebra"


@pytest.mark.parametrize("threshold,positive,task_type,metric", [
    (-.1, "zebra", "classification", "f1"),
    (1.1, "zebra", "classification", "f1"),
    (float("nan"), "zebra", "classification", "f1"),
    (True, "zebra", "classification", "f1"),
    (.3, "missing", "classification", "f1"),
    (.3, "zebra", "classification", "roc_auc"),
    (.3, "zebra", "regression", "rmse"),
])
def test_invalid_binary_decision_rejected_before_candidate_load(tmp_path, threshold, positive, task_type, metric):
    inputs, context, _ = _write_fixture(tmp_path, task_type=task_type)
    context.update(metric=metric, positive_class="zebra")
    code = _candidate(tmp_path, "unreachable.py", "def build_model(context):\n    raise AssertionError('invalid decision must reject before fit')\n")
    artifact = tmp_path / "invalid_binary"
    with pytest.raises(ValueError):
        run_final_ensemble({
            "context": context, "inputs": inputs, "artifact_dir": str(artifact), "save_model": True,
            "members": [{"code_path": code, "trial_id": "guard"}], "weights": [1.],
            "binary_threshold": threshold, "positive_class": positive,
        })
    assert not (artifact / "worker_result.json").exists()
