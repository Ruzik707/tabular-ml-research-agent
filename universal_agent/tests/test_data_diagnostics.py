import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from automl_agent.data_diagnostics import dataset_diagnostics, feature_importance


def _task(tmp_path, train, test, *, target="target", ids=("row_id",), features=None):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    feature_columns = features or [column for column in train.columns if column not in {*ids, target}]
    return SimpleNamespace(
        train_path=str(train_path),
        test_path=str(test_path),
        target=target,
        task_type="regression",
        feature_columns=feature_columns,
        id_columns=list(ids),
        categorical_columns=[column for column in feature_columns if train[column].dtype == object],
        seed=17,
    )


def test_dataset_diagnostics_reports_future_only_temporal_shift_and_domain_cv(tmp_path):
    train = pd.DataFrame({
        "row_id": range(120),
        "target": np.arange(120) * 100,
        "period_year": np.repeat([1, 2, 3, 4, 5], 24),
        "segment": np.tile(["a", "b", "c"], 40),
        "amount": np.arange(120, dtype=float),
    })
    test = pd.DataFrame({
        "row_id": range(1000, 1120),
        "period_year": [6] * 120,
        "segment": ["new"] * 120,
        "amount": [float(value) for value in range(120, 240)],
    })
    task = _task(tmp_path, train, test)

    result = dataset_diagnostics(task, n_jobs=2, max_rows=100)

    assert "target" not in result["features"]
    assert "row_id" not in result["features"]
    assert result["features"]["segment"]["category_shift"]["test_only_count"] == 1
    assert result["features"]["segment"]["category_shift"]["test_top"][0]["value"] == "new"
    assert result["features"]["amount"]["train_missing_rate"] == 0
    temporal = result["temporal_candidates"]
    assert len(temporal) == 1
    assert temporal[0]["feature"] == "period_year"
    assert temporal[0]["train_group_count"] == 5
    assert temporal[0]["train_max"] == 5
    assert temporal[0]["test_min"] == 6
    assert temporal[0]["future_only"] is True
    assert temporal[0]["direction"] == "future"

    adversarial = result["adversarial_cv"]
    assert adversarial["status"] == "ok"
    assert len(adversarial["fold_roc_auc"]) == 3
    assert adversarial["rows_per_domain"] == {"train": 100, "test": 100}
    assert "target" not in adversarial["feature_names"]
    assert "row_id" not in adversarial["feature_names"]
    without_temporal = adversarial["without_temporal"]
    assert without_temporal["status"] == "ok"
    assert "period_year" not in without_temporal["feature_names"]
    json.dumps(result, allow_nan=False)


def test_dataset_diagnostics_skips_three_fold_auc_when_test_sample_is_too_small(tmp_path):
    train = pd.DataFrame({"row_id": range(9), "target": range(9), "feature": range(9)})
    test = pd.DataFrame({"row_id": [10, 11], "feature": [9, 10]})
    task = _task(tmp_path, train, test)

    result = dataset_diagnostics(task, n_jobs=1, max_rows=10)

    assert result["adversarial_cv"]["status"] == "skipped"
    assert "100" in result["adversarial_cv"]["reason"]


def test_temporal_detection_accepts_date_semantics_when_test_is_after_training(tmp_path):
    train = pd.DataFrame({
        "row_id": [1, 2, 3],
        "target": [0.0, 0.0, 0.0],
        "event_date": ["2020-01-01", "2020-02-01", "2020-03-01"],
    })
    test = pd.DataFrame({"row_id": [4], "event_date": ["2020-04-01"]})
    task = _task(tmp_path, train, test)

    result = dataset_diagnostics(task, n_jobs=1, max_rows=10)

    candidate = result["temporal_candidates"][0]
    assert candidate["feature"] == "event_date"
    assert candidate["kind"] == "datetime"
    assert candidate["train_group_count"] == 3
    assert candidate["future_only"] is True
    assert result["adversarial_cv"]["status"] == "skipped"


def test_feature_importance_aggregates_ohe_and_missing_indicators_to_original_features():
    frame = pd.DataFrame({
        "amount": [0.0, 1.0, np.nan, 0.5, 5.0, 6.0, np.nan, 7.0, 0.2, 8.0],
        "segment": ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c"],
    })
    target = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), ["amount"]),
        ("category", OneHotEncoder(handle_unknown="ignore"), ["segment"]),
    ])
    model = Pipeline([
        ("preprocess", preprocessing),
        ("model", ExtraTreesClassifier(n_estimators=8, random_state=3)),
    ]).fit(frame, target)

    result = feature_importance(model, {"feature_columns": ["amount", "segment"]})

    assert result["status"] == "ok"
    assert result["mapping_status"] == "original_features"
    assert {item["feature"] for item in result["items"]} == {"amount", "segment"}
    assert sum(item["importance"] for item in result["items"]) == pytest.approx(1.0)
    assert result["method"] == "native_feature_importances_"
    assert "causal" in result["interpretation"]


def test_feature_importance_marks_unknown_estimators_unsupported_without_getters():
    class Unknown:
        @property
        def feature_importances_(self):
            raise AssertionError("must not call an arbitrary custom property")

    result = feature_importance(Unknown(), {"feature_columns": ["x"]})

    assert result["status"] == "unsupported"
    assert result["items"] == []
