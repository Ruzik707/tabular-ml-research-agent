import json

import numpy as np
import pandas as pd
import pytest

from automl_agent.contracts import TaskSpec
from automl_agent.evaluation import publish_submission, score_trial, validate_submission


def make_task(tmp_path, task_type="regression"):
    train = pd.DataFrame({"key": [1, 2, 3, 4], "x": [0, 1, 2, 3],
                          "label": [0.0, 1.0, 2.0, 3.0] if task_type == "regression" else ["a", "b", "b", "a"]})
    test = pd.DataFrame({"key": [10, 11, 12], "x": [4, 5, 6]})
    sample = pd.DataFrame({"key": test.key,
                           "label": [0, 0, 0] if task_type == "regression" else ["a", "a", "a"]})
    train_path, test_path, sample_path = tmp_path / "train.csv", tmp_path / "test.csv", tmp_path / "sample.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    sample.to_csv(sample_path, index=False)
    return TaskSpec(
        task_path=str(tmp_path / "task.md"), train_path=str(train_path), test_path=str(test_path),
        target="label", task_type=task_type, metric="rmse" if task_type == "regression" else "macro_f1",
        maximize=task_type == "classification", feature_columns=["x"], id_columns=["key"],
        categorical_columns=[], output_columns=["key", "label"], data_fingerprint="fixture",
        sample_path=str(sample_path), classes=["a", "b"] if task_type == "classification" else [],
    )


def test_host_scores_rmse_and_publishes_atomically_in_sample_column_and_row_order(tmp_path):
    task = make_task(tmp_path)
    split = tmp_path / "split"
    split.mkdir()
    pd.DataFrame({"label": [1.0, 2.0]}).to_csv(split / "validation_target.csv", index=False)
    trial = tmp_path / "trial"
    trial.mkdir()
    pd.DataFrame({"label": [0.0, 4.0]}).to_csv(trial / "valid_predictions.csv", index=False)
    metrics = score_trial(task, {"validation_target": str(split / "validation_target.csv")}, trial)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.5))

    predictions = tmp_path / "test_predictions.csv"
    pd.DataFrame({"label": [5.0, 6.0, 7.0]}).to_csv(predictions, index=False)
    destination = tmp_path / "submission.csv"
    report = publish_submission(task, predictions, destination)
    result = pd.read_csv(destination)
    assert result.columns.tolist() == ["key", "label"]
    assert result.key.tolist() == [10, 11, 12]
    assert result.label.tolist() == [5.0, 6.0, 7.0]
    assert report["rows"] == 3
    assert validate_submission(task, destination)["valid"] is True


def test_invalid_predictions_do_not_replace_working_submission(tmp_path):
    task = make_task(tmp_path)
    destination = tmp_path / "submission.csv"
    destination.write_text("known-good\n", encoding="utf-8")
    predictions = tmp_path / "bad.csv"
    pd.DataFrame({"label": [1.0, np.inf]}).to_csv(predictions, index=False)

    with pytest.raises(ValueError):
        publish_submission(task, predictions, destination)
    assert destination.read_text(encoding="utf-8") == "known-good\n"


def test_classification_score_uses_macro_f1_and_rejects_unknown_submission_classes(tmp_path):
    task = make_task(tmp_path, "classification")
    split = tmp_path / "split"
    split.mkdir()
    trial = tmp_path / "trial"
    trial.mkdir()
    pd.DataFrame({"label": ["a", "b", "b", "b"]}).to_csv(split / "validation_target.csv", index=False)
    pd.DataFrame({"label": ["a", "a", "b", "b"]}).to_csv(trial / "valid_predictions.csv", index=False)
    metrics = score_trial(task, {"validation_target": str(split / "validation_target.csv")}, trial)
    assert metrics["macro_f1"] == pytest.approx(11 / 15)

    bad = tmp_path / "bad_classes.csv"
    pd.DataFrame({"label": ["a", "intruder", "b"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        publish_submission(task, bad, tmp_path / "new.csv")


def test_probability_metrics_use_host_artifacts_and_validation_labels(tmp_path):
    task = make_task(tmp_path, "classification")
    task.metric = "log_loss"
    split = tmp_path / "split"
    split.mkdir()
    trial = tmp_path / "trial"
    trial.mkdir()
    pd.DataFrame({"label": ["a", "b", "b"]}).to_csv(split / "validation_target.csv", index=False)
    pd.DataFrame({"label": ["a", "b", "a"]}).to_csv(trial / "valid_predictions.csv", index=False)
    np.save(trial / "valid_probabilities.npy", np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]))
    (trial / "probability_classes.json").write_text('{"classes": ["a", "b"]}', encoding="utf-8")

    metrics = score_trial(task, {"validation_target": str(split / "validation_target.csv")}, trial)
    assert metrics["log_loss"] == pytest.approx(-np.log([0.9, 0.8, 0.4]).mean())


def test_binary_f1_uses_positive_class_not_macro_average(tmp_path):
    task = make_task(tmp_path, "classification")
    task.metric = "f1"
    split = tmp_path / "split"
    trial = tmp_path / "trial"
    split.mkdir()
    trial.mkdir()
    pd.DataFrame({"label": ["b", "a", "a", "a", "a", "a"]}).to_csv(
        split / "validation_target.csv", index=False
    )
    pd.DataFrame({"label": ["a"] * 6}).to_csv(trial / "valid_predictions.csv", index=False)

    metrics = score_trial(task, {"validation_target": str(split / "validation_target.csv")}, trial)

    assert metrics["f1"] == pytest.approx(0.0)
    assert metrics["macro_f1"] == pytest.approx(5 / 11)


def test_f1_metric_rejects_multiclass_instead_of_silently_using_macro(tmp_path):
    task = make_task(tmp_path, "classification")
    task.metric = "f1"
    task.classes = ["a", "b", "c"]
    split = tmp_path / "split"
    trial = tmp_path / "trial"
    split.mkdir()
    trial.mkdir()
    pd.DataFrame({"label": ["a", "b", "c"]}).to_csv(split / "validation_target.csv", index=False)
    pd.DataFrame({"label": ["a", "b", "c"]}).to_csv(trial / "valid_predictions.csv", index=False)

    with pytest.raises(ValueError, match="бинарной задачи"):
        score_trial(task, {"validation_target": str(split / "validation_target.csv")}, trial)


def test_one_row_rmse_omits_undefined_r2_and_serializes_strictly(tmp_path):
    task = make_task(tmp_path)
    split = tmp_path / "split"
    trial = tmp_path / "trial"
    split.mkdir()
    trial.mkdir()
    pd.DataFrame({"label": [3.0]}).to_csv(split / "validation_target.csv", index=False)
    pd.DataFrame({"label": [2.0]}).to_csv(trial / "valid_predictions.csv", index=False)

    metrics = score_trial(task, {"validation_target": str(split / "validation_target.csv")}, trial)

    assert metrics["rmse"] == pytest.approx(1.0)
    assert "r2" not in metrics
    assert all(np.isfinite(value) for value in metrics.values())
    json.dumps(metrics, allow_nan=False)


def test_one_row_primary_r2_raises_clear_error(tmp_path):
    task = make_task(tmp_path)
    task.metric = "r2"
    split = tmp_path / "split"
    trial = tmp_path / "trial"
    split.mkdir()
    trial.mkdir()
    pd.DataFrame({"label": [3.0]}).to_csv(split / "validation_target.csv", index=False)
    pd.DataFrame({"label": [2.0]}).to_csv(trial / "valid_predictions.csv", index=False)

    with pytest.raises(ValueError, match="r2.*one validation row"):
        score_trial(task, {"validation_target": str(split / "validation_target.csv")}, trial)


def test_publish_and_validate_preserve_zero_padded_ids(tmp_path):
    task = make_task(tmp_path)
    pd.DataFrame({"key": ["001", "010", "020"], "x": [4, 5, 6]}).to_csv(
        task.test_path, index=False
    )
    pd.DataFrame({"key": ["001", "010", "020"], "label": [0, 0, 0]}).to_csv(
        task.sample_path, index=False
    )
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"label": [5.0, 6.0, 7.0]}).to_csv(predictions, index=False)
    destination = tmp_path / "submission.csv"

    publish_submission(task, predictions, destination)

    assert pd.read_csv(destination, dtype={"key": "string"})["key"].tolist() == ["001", "010", "020"]
    assert validate_submission(task, destination)["valid"] is True
