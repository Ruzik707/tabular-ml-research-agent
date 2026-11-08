from __future__ import annotations

import inspect
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from automl_agent.training import fit_model


def _classification_data(rows=96):
    rng = np.random.default_rng(91)
    labels = np.array(["major"] * (rows * 3 // 4) + ["minor"] * (rows // 4))
    signal = np.concatenate([
        rng.normal(-0.7, 0.45, size=rows * 3 // 4),
        rng.normal(0.8, 0.45, size=rows // 4),
    ])
    order = rng.permutation(rows)
    return pd.DataFrame({"signal": signal[order], "noise": rng.normal(size=rows)}), labels[order]


def _make_native_model(backend):
    if backend == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=20,
            depth=3,
            learning_rate=0.2,
            eval_metric="Logloss",
            random_seed=13,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        ), CatBoostClassifier, "iterations"
    if backend == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=24,
            learning_rate=0.2,
            num_leaves=7,
            min_child_samples=2,
            n_jobs=1,
            random_state=13,
            verbosity=-1,
        ), LGBMClassifier, "n_estimators"
    if backend == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=24,
            max_depth=2,
            learning_rate=0.2,
            min_child_weight=0,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=13,
        ), XGBClassifier, "n_estimators"
    raise AssertionError(backend)


def _record_native_fit(monkeypatch, estimator_class):
    original_fit = estimator_class.fit
    calls = []

    def recording_fit(self, X, y, *args, **kwargs):
        sample_weight = kwargs.get("sample_weight")
        calls.append({
            "instance": self,
            "y": np.asarray(y).copy(),
            "sample_weight": None if sample_weight is None else np.asarray(sample_weight).copy(),
            "kwargs": dict(kwargs),
        })
        return original_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(estimator_class, "fit", recording_fit)
    return calls


def _expected_weights(y, mode):
    labels, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    weights = len(y) / (len(labels) * counts[inverse])
    if mode == "sqrt_balanced":
        weights = np.sqrt(weights)
        weights = weights / np.average(weights)
    return weights


@pytest.mark.parametrize(
    ("backend", "weight_mode"),
    [("catboost", "balanced"), ("lightgbm", "sqrt_balanced"), ("xgboost", "balanced")],
)
def test_native_early_stop_probes_inner_split_then_refits_full_train(monkeypatch, backend, weight_mode):
    pytest.importorskip(backend if backend != "xgboost" else "xgboost")
    model, estimator_class, iteration_parameter = _make_native_model(backend)
    calls = _record_native_fit(monkeypatch, estimator_class)
    X, y = _classification_data()

    fitted, summary = fit_model(
        model,
        X,
        y,
        context={"task_type": "classification", "seed": 13, "n_jobs": 1},
        options={
            "early_stopping_rounds": 3,
            "inner_validation_fraction": 0.2,
            "sample_weight": weight_mode,
        },
    )

    assert len(calls) == 2
    probe, refit = calls
    assert probe["instance"] is not refit["instance"]
    assert "eval_set" in probe["kwargs"]
    assert "eval_set" not in refit["kwargs"]
    assert probe["sample_weight"] is not None
    assert refit["sample_weight"] is not None
    assert len(probe["sample_weight"]) == summary["inner_train_rows"]
    assert len(refit["sample_weight"]) == len(y)
    assert np.allclose(probe["sample_weight"], _expected_weights(probe["y"], weight_mode))
    assert np.allclose(refit["sample_weight"], _expected_weights(y, weight_mode))
    assert summary["early_stopping_used"] is True
    assert summary["inner_train_rows"] + summary["inner_valid_rows"] == len(y)
    assert summary["train_rows"] == len(y)
    assert summary["best_iteration"] >= 1
    assert summary["best_iteration"] <= summary["requested_iterations"]
    assert summary["sample_weight_mode"] == weight_mode
    assert summary["refit_seconds"] >= 0
    assert summary["fit_seconds"] >= summary["refit_seconds"]
    assert refit["instance"].get_params(deep=False)[iteration_parameter] == summary["best_iteration"]
    assert json.loads(json.dumps(summary))["train_rows"] == len(y)
    native = fitted.estimator if hasattr(fitted, "estimator") else fitted
    if backend == "catboost":
        assert native.tree_count_ == summary["best_iteration"]
    elif backend == "lightgbm":
        assert native.n_estimators_ == summary["best_iteration"]
    else:
        assert native.get_booster().num_boosted_rounds() == summary["best_iteration"]
    if backend == "catboost":
        assert model.tree_count_ is None
    else:
        assert not hasattr(model, "booster_")


def test_iteration_override_refits_full_train_without_inner_probe(monkeypatch):
    pytest.importorskip("xgboost")
    model, estimator_class, _ = _make_native_model("xgboost")
    calls = _record_native_fit(monkeypatch, estimator_class)
    X, y = _classification_data()

    fitted, summary = fit_model(
        model,
        X,
        y,
        context={"task_type": "classification", "seed": 4},
        options={"early_stopping_rounds": 8, "sample_weight": None},
        iteration_override=6,
    )

    assert len(calls) == 1
    assert "eval_set" not in calls[0]["kwargs"]
    assert len(calls[0]["y"]) == len(y)
    assert summary["early_stopping_used"] is False
    assert summary["best_iteration"] == 6
    assert summary["requested_iterations"] == 6
    assert summary["reason"] == "iteration_override_supplied"
    native = fitted.estimator
    assert native.get_params(deep=False)["n_estimators"] == 6


@pytest.mark.parametrize("backend", ["catboost", "lightgbm", "xgboost"])
def test_native_preflight_override_fits_ten_trees_and_survives_serialization(
    tmp_path, monkeypatch, backend,
):
    pytest.importorskip(backend)
    model, estimator_class, iteration_parameter = _make_native_model(backend)
    calls = _record_native_fit(monkeypatch, estimator_class)
    X, y = _classification_data(rows=48)

    fitted, summary = fit_model(
        model,
        X,
        y,
        context={"task_type": "classification", "seed": 11},
        options={"early_stopping_rounds": 4},
        iteration_override=10,
    )

    assert len(calls) == 1
    assert "eval_set" not in calls[0]["kwargs"]
    assert len(calls[0]["y"]) == len(y)
    assert summary["requested_iterations"] == summary["best_iteration"] == 10
    assert summary["early_stopping_used"] is False
    native = fitted.estimator if hasattr(fitted, "estimator") else fitted
    assert native.get_params(deep=False)[iteration_parameter] == 10
    if backend == "catboost":
        assert native.tree_count_ == 10
    elif backend == "lightgbm":
        assert native.n_estimators_ == 10
    else:
        assert native.get_booster().num_boosted_rounds() == 10

    path = tmp_path / f"{backend}-preflight.joblib"
    joblib.dump(fitted, path)
    loaded = joblib.load(path)
    assert len(loaded.predict(X.iloc[:4])) == 4
    assert loaded.predict_proba(X.iloc[:4]).shape == (4, 2)


def test_iteration_override_is_ignored_for_ordinary_sklearn_estimator():
    X, y = _classification_data(rows=48)
    model = ExtraTreesClassifier(n_estimators=5, max_depth=3, random_state=4)

    fitted, summary = fit_model(
        model,
        X,
        y,
        context={"task_type": "classification", "seed": 4},
        iteration_override=10,
    )

    assert fitted.get_params(deep=False)["n_estimators"] == 5
    assert summary["requested_iterations"] is None
    assert summary["best_iteration"] is None
    assert summary["early_stopping_used"] is False
    assert summary["reason"] == "iteration_override_ignored_non_booster"

    with pytest.raises(ValueError, match="early_stopping_rounds requires native"):
        fit_model(
            model,
            X,
            y,
            context={"task_type": "classification", "seed": 4},
            options={"early_stopping_rounds": 2},
            iteration_override=10,
        )


class _FitRowRecorder(BaseEstimator, TransformerMixin):
    fit_rows = []

    def fit(self, X, y=None):
        self.n_samples_seen_ = len(X)
        type(self).fit_rows.append(len(X))
        return self

    def transform(self, X):
        return X


def test_pipeline_preprocessing_fits_inner_train_then_full_train_and_allows_unseen_categories(monkeypatch):
    pytest.importorskip("lightgbm")
    from lightgbm import LGBMClassifier

    _FitRowRecorder.fit_rows.clear()
    model = Pipeline([
        ("track", _FitRowRecorder()),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
        ("model", LGBMClassifier(
            n_estimators=18, min_child_samples=2, n_jobs=1, random_state=13, verbosity=-1,
        )),
    ])
    calls = _record_native_fit(monkeypatch, LGBMClassifier)
    X, y = _classification_data()
    categorical = pd.DataFrame({"category": np.where(X["signal"] > 0, "warm", "cool")})

    fitted, summary = fit_model(
        model,
        categorical,
        y,
        context={"task_type": "classification", "seed": 13},
        options={"early_stopping_rounds": 2, "sample_weight": "balanced"},
    )

    assert summary["early_stopping_used"] is True
    assert _FitRowRecorder.fit_rows == [summary["inner_train_rows"], len(y)]
    assert len(calls) == 2
    assert len(calls[0]["sample_weight"]) == summary["inner_train_rows"]
    assert len(calls[1]["sample_weight"]) == len(y)
    predictions = fitted.predict(pd.DataFrame({"category": ["unseen"]}))
    assert len(predictions) == 1


def test_chronological_inner_split_uses_only_local_outer_train_indices(monkeypatch):
    pytest.importorskip("catboost")
    from catboost import CatBoostRegressor

    X = pd.DataFrame({"time": np.arange(48, dtype=float)})
    y = np.sin(np.arange(48, dtype=float) / 4.0) + np.arange(48, dtype=float) / 20.0
    calls = _record_native_fit(monkeypatch, CatBoostRegressor)
    model = CatBoostRegressor(
        iterations=12,
        depth=3,
        learning_rate=0.2,
        loss_function="RMSE",
        random_seed=7,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    )

    _fitted, summary = fit_model(
        model,
        X,
        y,
        context={"task_type": "regression", "seed": 7},
        options={"early_stopping_rounds": 2},
        inner_split={"train_indices": np.arange(30), "valid_indices": np.arange(30, 48)},
    )

    assert len(calls) == 2
    assert np.array_equal(calls[0]["y"], y[:30])
    assert np.array_equal(calls[0]["kwargs"]["eval_set"][1], y[30:])
    assert summary["inner_train_rows"] == 30
    assert summary["inner_valid_rows"] == 18


def test_xgboost_regression_does_not_use_class_label_decoder():
    pytest.importorskip("xgboost")
    from xgboost import XGBRegressor

    X = np.arange(24, dtype=float).reshape(-1, 1)
    y = np.sin(X[:, 0] / 5.0)
    fitted, summary = fit_model(
        XGBRegressor(n_estimators=6, max_depth=2, n_jobs=1, tree_method="hist"),
        X,
        y,
        context={"task_type": "regression", "seed": 3},
    )

    assert isinstance(fitted, XGBRegressor)
    assert summary["best_iteration"] == fitted.get_booster().num_boosted_rounds() == 6


def test_xgboost_string_labels_decode_and_survive_joblib_reload(tmp_path):
    pytest.importorskip("xgboost")
    model, _estimator_class, _iteration_parameter = _make_native_model("xgboost")
    X, y = _classification_data()
    categories = pd.DataFrame({"category": np.where(X["signal"] > 0, "warm", "cool")})
    pipeline = Pipeline([
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ("model", model),
    ])

    fitted, summary = fit_model(
        pipeline,
        categories,
        y,
        context={"task_type": "classification", "seed": 13},
        options={"sample_weight": "balanced"},
    )
    path = tmp_path / "fitted.joblib"
    joblib.dump(fitted, path)
    loaded = joblib.load(path)

    unseen = pd.DataFrame({"category": ["unseen"] * 8})
    predictions = loaded.predict(unseen)
    probabilities = loaded.predict_proba(unseen)
    decoder = loaded.steps[-1][1]
    assert summary["sample_weight_mode"] == "balanced"
    assert set(loaded.classes_) == set(np.unique(y))
    assert set(predictions).issubset(set(np.unique(y)))
    assert probabilities.shape == (8, len(loaded.classes_))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert decoder._model is decoder.estimator
    assert type(decoder._model).__name__ == "XGBClassifier"


def test_rare_class_skips_inner_early_stopping_with_reason():
    pytest.importorskip("catboost")
    from catboost import CatBoostClassifier

    X = pd.DataFrame({"x": np.arange(16, dtype=float)})
    y = np.array(["common"] * 15 + ["rare"])
    fitted, summary = fit_model(
        CatBoostClassifier(iterations=5, thread_count=1, verbose=False, allow_writing_files=False),
        X,
        y,
        context={"task_type": "classification", "seed": 2},
        options={"early_stopping_rounds": 2},
    )

    assert fitted.tree_count_ > 0
    assert summary["early_stopping_used"] is False
    assert "rare_class" in summary["reason"]
    assert summary["inner_train_rows"] is None


def test_sample_weight_rejects_constructor_class_weight_doubling():
    pytest.importorskip("catboost")
    from catboost import CatBoostClassifier

    X, y = _classification_data()
    model = CatBoostClassifier(
        iterations=5,
        auto_class_weights="Balanced",
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    )

    with pytest.raises(ValueError, match="double weighting"):
        fit_model(
            model,
            X,
            y,
            context={"task_type": "classification"},
            options={"sample_weight": "balanced"},
        )


@pytest.mark.parametrize("backend", ["lightgbm", "xgboost"])
def test_native_constructor_weighting_rejects_sample_weight_doubling(backend):
    X, y = _classification_data()
    if backend == "lightgbm":
        pytest.importorskip("lightgbm")
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(n_estimators=4, class_weight="balanced", n_jobs=1, verbosity=-1)
    else:
        pytest.importorskip("xgboost")
        from xgboost import XGBClassifier

        model = XGBClassifier(n_estimators=4, scale_pos_weight=3, n_jobs=1, tree_method="hist")

    with pytest.raises(ValueError, match="double weighting"):
        fit_model(
            model,
            X,
            y,
            context={"task_type": "classification"},
            options={"sample_weight": "balanced"},
        )


def test_balanced_sample_weight_is_rejected_for_regression():
    from sklearn.linear_model import LinearRegression

    with pytest.raises(ValueError, match="only supported for classification"):
        fit_model(
            LinearRegression(),
            np.arange(10, dtype=float).reshape(-1, 1),
            np.arange(10, dtype=float),
            context={"task_type": "regression"},
            options={"sample_weight": "balanced"},
        )


def test_unknown_options_are_rejected_and_outer_validation_is_not_in_signature():
    with pytest.raises(ValueError, match="unknown fit option"):
        fit_model(
            ExtraTreesClassifier(n_estimators=3),
            np.arange(12).reshape(-1, 1),
            np.array([0, 1] * 6),
            context={"task_type": "classification"},
            options={"early_stopping_rounds": 0, "evaluation_labels": [0, 1]},
        )

    assert "outer_validation_y" not in inspect.signature(fit_model).parameters
