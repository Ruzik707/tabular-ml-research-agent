import json
from types import SimpleNamespace

import pandas as pd
import pytest

from automl_agent.diagnostics import validation_diagnostics


def _task(task_type="classification", classes=None, categorical_columns=None):
    return SimpleNamespace(
        task_type=task_type,
        target="label",
        classes=classes or [],
        categorical_columns=categorical_columns or [],
        feature_columns=list(categorical_columns or []),
    )


def _write_inputs(tmp_path, actual, predicted, actual_column="label", predicted_column="label"):
    split_dir = tmp_path / "split"
    trial_dir = tmp_path / "trial"
    split_dir.mkdir()
    trial_dir.mkdir()
    actual_path = split_dir / "validation_target.csv"
    prediction_path = trial_dir / "valid_predictions.csv"
    pd.DataFrame({actual_column: actual}).to_csv(actual_path, index=False)
    pd.DataFrame({predicted_column: predicted}).to_csv(prediction_path, index=False)
    return {"validation_target": actual_path}, trial_dir


def test_regression_diagnostics_are_empty_without_reading_artifacts(tmp_path):
    assert validation_diagnostics(_task("regression"), {}, tmp_path) == {}


def test_classification_reports_rare_and_zero_support_labels_without_rows(tmp_path):
    split, trial_dir = _write_inputs(
        tmp_path,
        ["common", "common", "common", "rare", "gone"],
        ["common", "common", "rare", "common", "phantom"],
    )

    result = validation_diagnostics(
        _task(classes=["common", "rare", "gone", "phantom", "unobserved"]),
        split,
        trial_dir,
    )

    assert result["total_classes"] == 5
    assert result["partial"] is False
    matrix = result["confusion_matrix"]
    assert matrix["total_classes"] == 5
    assert matrix["partial"] is False
    assert len(matrix["labels"]) == 5
    assert len(matrix["matrix"]) == 5
    assert all(len(row) == 5 for row in matrix["matrix"])
    assert "common" in matrix["labels"]
    common_index = matrix["labels"].index("common")
    rare_index = matrix["labels"].index("rare")
    assert matrix["matrix"][common_index][common_index] == 2
    assert matrix["matrix"][common_index][rare_index] == 1
    rows = {row["label"]: row for row in result["per_class"]}
    assert rows["common"] == {
        "label": "common", "precision": pytest.approx(2 / 3),
        "recall": pytest.approx(2 / 3), "f1": pytest.approx(2 / 3), "support": 3,
    }
    assert rows["rare"]["support"] == 1
    assert rows["rare"]["f1"] == 0
    assert rows["phantom"]["support"] == 0
    assert rows["phantom"]["f1"] == 0
    assert rows["unobserved"]["support"] == 0
    assert rows["unobserved"]["f1"] == 0
    assert [row["f1"] for row in result["per_class"]] == sorted(
        row["f1"] for row in result["per_class"]
    )
    assert set(result) == {"per_class", "total_classes", "partial", "confusion_matrix"}
    assert all(set(row) == {"label", "precision", "recall", "f1", "support"} for row in result["per_class"])
    json.dumps(result, allow_nan=False)


def test_integer_labels_are_returned_as_json_safe_integers(tmp_path):
    split, trial_dir = _write_inputs(tmp_path, [1, 1, 2], [1, 2, 2])

    result = validation_diagnostics(_task(), split, trial_dir)

    assert {row["label"] for row in result["per_class"]} == {1, 2}
    assert all(type(row["label"]) is int for row in result["per_class"])
    json.dumps(result, allow_nan=False)


def test_numeric_float_class_labels_are_normalized_to_json_safe_labels(tmp_path):
    split, trial_dir = _write_inputs(tmp_path, [0.0, 1.5], [0.0, 0.0])

    result = validation_diagnostics(_task(), split, trial_dir)

    assert {row["label"] for row in result["per_class"]} == {0, "1.5"}
    json.dumps(result, allow_nan=False)


def test_classification_diagnostics_are_bounded_and_sort_worst_f1_first(tmp_path):
    labels = list(range(40))
    split, trial_dir = _write_inputs(tmp_path, labels, labels[1:] + labels[:1])

    result = validation_diagnostics(_task(), split, trial_dir)

    assert result["total_classes"] == 40
    assert result["partial"] is True
    assert len(result["per_class"]) == 30
    assert len(result["confusion_matrix"]["labels"]) == 30
    assert result["confusion_matrix"]["partial"] is True
    assert [row["f1"] for row in result["per_class"]] == sorted(
        row["f1"] for row in result["per_class"]
    )


@pytest.mark.parametrize(
    ("actual", "predicted", "actual_column", "predicted_column", "message"),
    [
        (["a", "b"], ["a"], "label", "label", "Размер"),
        (["a", "b"], ["a", "b"], "other", "label", "один столбец"),
        (["a", "b"], ["a", "b"], "label", "other", "один столбец"),
    ],
)
def test_invalid_lengths_or_prediction_schemas_raise(tmp_path, actual, predicted,
                                                     actual_column, predicted_column, message):
    split, trial_dir = _write_inputs(
        tmp_path, actual, predicted, actual_column, predicted_column
    )

    with pytest.raises(ValueError, match=message):
        validation_diagnostics(_task(), split, trial_dir)


def test_regression_reports_only_supported_worst_categorical_segments(tmp_path):
    split_dir = tmp_path / "split"
    trial_dir = tmp_path / "trial"
    split_dir.mkdir()
    trial_dir.mkdir()
    categories = ["bad"] * 25 + ["good"] * 22 + ["rare"] * 19
    actual = [0.0] * len(categories)
    predicted = [10.0] * 25 + [0.0] * 22 + [100.0] * 19
    pd.DataFrame({"label": actual}).to_csv(split_dir / "validation_target.csv", index=False)
    pd.DataFrame({"label": predicted}).to_csv(trial_dir / "valid_predictions.csv", index=False)
    features_path = split_dir / "validation_features.csv"
    pd.DataFrame({"segment": categories}).to_csv(features_path, index=False)

    result = validation_diagnostics(
        _task("regression", categorical_columns=["segment"]),
        {"validation_target": split_dir / "validation_target.csv", "validation_features": features_path},
        trial_dir,
    )

    assert result["total_segments"] == 2
    assert result["partial"] is False
    assert [segment["category"] for segment in result["segments"]] == ["bad", "good"]
    assert result["segments"][0]["count"] == 25
    assert result["segments"][0]["rmse"] == pytest.approx(10)
    assert result["segments"][0]["mae"] == pytest.approx(10)
