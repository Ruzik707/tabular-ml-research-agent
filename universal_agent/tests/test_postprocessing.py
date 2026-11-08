import json

import numpy as np
import pytest

from automl_agent.postprocessing import apply_postprocessing, select_postprocessing


def test_regression_postprocessing_never_enumerates_continuous_targets_as_classes(monkeypatch):
    def reject_class_discovery(*args):
        raise AssertionError('Continuous targets must not enter class discovery')
    monkeypatch.setattr('automl_agent.postprocessing._ordered_classes', reject_class_discovery)
    y=np.linspace(0,1000,50000)
    result=select_postprocessing(y=y, predictions=[y+1,y-1], probabilities=None, classes=[],
        fold_ids=np.arange(len(y))%5, metric='rmse', maximize=False)
    assert result['accepted'] and result['calibrated_oof_score'] < 1e-10


def _binary_fixture(folds=4):
    y = np.asarray(["major", "minor"] * (folds * 10))
    fold_ids = np.repeat(np.arange(folds), 20)
    probability = np.asarray(
        [[0.9, 0.1] if label == "major" else [0.55, 0.45] for label in y],
        dtype=float,
    )
    hard_prediction = np.repeat("major", len(y))
    return y, fold_ids, probability, hard_prediction


def test_single_model_macro_f1_calibration_is_cross_fitted_and_improves_unseen_folds():
    y, fold_ids, probability, hard_prediction = _binary_fixture()
    result = select_postprocessing(
        y=y,
        predictions=[hard_prediction],
        probabilities=[probability],
        classes=["major", "minor"],
        fold_ids=fold_ids,
        metric="macro_f1",
        maximize=True,
    )

    assert result["assessment"] == "cross_fitted"
    assert result["accepted"] is True
    assert result["weights"] == [1.0]
    assert result["class_multipliers"] is not None
    assert result["assessed_score"] > result["baseline_score"]
    assert len(result["fold_scores"]) == 4
    labels, adjusted_probability = apply_postprocessing(
        [hard_prediction],
        [probability],
        ["major", "minor"],
        result["weights"],
        result["class_multipliers"],
        task_type="classification",
    )
    assert np.array_equal(labels, y)
    assert np.allclose(adjusted_probability.sum(axis=1), 1.0)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("fold_count", [1, 4])
@pytest.mark.parametrize(
    ("label_pair", "classes"),
    [([0, 1], [0, 1]), (["major", "minor"], ["major", "minor"])],
)
def test_baseline_preserves_custom_classifier_predictions_instead_of_probability_argmax(
    fold_count, label_pair, classes
):
    y = np.tile(label_pair, 20)
    fold_ids = np.repeat(np.arange(fold_count), len(y) // fold_count)
    probabilities = np.tile([[0.9, 0.1], [0.55, 0.45]], (20, 1))
    result = select_postprocessing(
        y=y,
        predictions=[y.copy()],  # Custom estimator threshold already predicts both classes correctly.
        probabilities=[probabilities],
        classes=classes,
        fold_ids=fold_ids,
        metric="macro_f1",
        maximize=True,
    )

    assert result["baseline_score"] == pytest.approx(1.0)
    if fold_count == 1:
        assert result["accepted"] is False
    else:
        assert result["accepted"] is False
        assert result["assessed_score"] == pytest.approx(1.0)


def test_iid_blend_learns_two_model_weights_and_accepts_only_cross_fold_gain():
    y, fold_ids, probability, hard_prediction = _binary_fixture()
    second_probability = np.asarray(
        [[0.95, 0.05] if label == "major" else [0.05, 0.95] for label in y],
        dtype=float,
    )
    second_prediction = np.where(second_probability[:, 0] > 0.5, "major", "minor")
    result = select_postprocessing(
        y=y,
        predictions=[hard_prediction, second_prediction],
        probabilities=[probability, second_probability],
        classes=["major", "minor"],
        fold_ids=fold_ids,
        metric="macro_f1",
        maximize=True,
    )

    assert result["accepted"] is True
    assert len(result["weights"]) == 2
    assert sum(result["weights"]) == pytest.approx(1.0)
    assert all(weight >= 0 for weight in result["weights"])
    assert result["assessed_score"] > result["baseline_score"]
    assert result["assessment"] == "cross_fitted"


def test_temporal_assessment_trains_only_on_earlier_oof_folds():
    y = np.arange(24, dtype=float)
    fold_ids = np.repeat(["year1", "year2", "year3"], 8)
    predictions = [y + 2.0, y - 2.0]
    result = select_postprocessing(
        y=y,
        predictions=predictions,
        probabilities=None,
        classes=[],
        fold_ids=fold_ids,
        metric="rmse",
        maximize=False,
        temporal=True,
    )

    assert result["assessment"] == "forward"
    assert [row["fold"] for row in result["fold_scores"]] == ["year2", "year3"]
    assert result["fold_scores"][0]["train_folds"] == ["year1"]
    assert result["fold_scores"][1]["train_folds"] == ["year1", "year2"]
    assert result["baseline_score"] == pytest.approx(2.0)
    assert result["assessed_score"] == pytest.approx(0.0)
    assert result["accepted"] is True


@pytest.mark.parametrize(
    ("metric", "maximize", "expected"),
    [("rmse", False, 0.0), ("mae", False, 0.0), ("r2", True, 1.0)],
)
def test_regression_metrics_optimize_correctly(metric, maximize, expected):
    y = np.asarray([0, 1, 4, 9, 16, 25, 36, 49], dtype=float)
    fold_ids = np.repeat([0, 1, 2, 3], 2)
    result = select_postprocessing(
        y=y,
        predictions=[y + 3.0, y - 3.0],
        probabilities=None,
        classes=[],
        fold_ids=fold_ids,
        metric=metric,
        maximize=maximize,
    )

    assert result["accepted"] is True
    assert result["weights"] == pytest.approx([0.5, 0.5])
    assert result["assessed_score"] == pytest.approx(expected)
    assert result["class_multipliers"] is None


def test_non_macro_class_metric_uses_metric_score_without_class_multipliers():
    y, fold_ids, probability, hard_prediction = _binary_fixture()
    improved_probability = np.asarray(
        [[0.9, 0.1] if label == "major" else [0.1, 0.9] for label in y],
        dtype=float,
    )
    improved_prediction = np.where(improved_probability[:, 0] > 0.5, "major", "minor")
    result = select_postprocessing(
        y=y,
        predictions=[hard_prediction, improved_prediction],
        probabilities=[probability, improved_probability],
        classes=["major", "minor"],
        fold_ids=fold_ids,
        metric="accuracy",
        maximize=True,
    )

    assert result["accepted"] is True
    assert result["assessed_score"] == pytest.approx(1.0)
    assert result["class_multipliers"] is None


def test_log_loss_uses_aligned_probabilities_and_minimizes_on_heldout_folds():
    y, fold_ids, probability, hard_prediction = _binary_fixture()
    better_probability = np.asarray(
        [[0.99, 0.01] if label == "major" else [0.01, 0.99] for label in y],
        dtype=float,
    )
    better_prediction = np.where(better_probability[:, 0] > 0.5, "major", "minor")
    result = select_postprocessing(
        y=y,
        predictions=[hard_prediction, better_prediction],
        probabilities=[probability, better_probability],
        classes=["major", "minor"],
        fold_ids=fold_ids,
        metric="log_loss",
        maximize=False,
    )

    assert result["accepted"] is True
    assert result["assessed_score"] < result["baseline_score"]
    assert result["class_multipliers"] is None


def test_identical_hard_predictions_are_not_accepted_without_strict_gain():
    y, fold_ids, _, hard_prediction = _binary_fixture()
    result = select_postprocessing(
        y=y,
        predictions=[hard_prediction, hard_prediction.copy()],
        probabilities=None,
        classes=["major", "minor"],
        fold_ids=fold_ids,
        metric="accuracy",
        maximize=True,
    )

    assert result["accepted"] is False
    assert result["assessed_score"] == result["baseline_score"]


def test_one_fold_skips_selection_gracefully_and_returns_json_safe_values():
    y, _, probability, hard_prediction = _binary_fixture(folds=1)
    result = select_postprocessing(
        y=y,
        predictions=[hard_prediction],
        probabilities=[probability],
        classes=["major", "minor"],
        fold_ids=np.zeros(len(y), dtype=int),
        metric="macro_f1",
        maximize=True,
    )

    assert result["accepted"] is False
    assert result["warnings"]
    assert result["fold_scores"] == []
    assert result["weights"] == [1.0]
    json.dumps(result, allow_nan=False)


def test_unknown_metrics_skip_with_warning_and_no_nan_values():
    y = np.asarray([0.0, 1.0, 2.0, 3.0])
    result = select_postprocessing(
        y=y,
        predictions=[y],
        probabilities=None,
        classes=[],
        fold_ids=np.asarray([0, 0, 1, 1]),
        metric="made_up_score",
        maximize=True,
    )

    assert result["accepted"] is False
    assert result["warnings"]
    json.dumps(result, allow_nan=False)


def test_apply_postprocessing_normalizes_probabilities_and_supports_numeric_classes():
    classes = [0, 1, 2]
    predictions = [np.asarray([0, 1]), np.asarray([2, 2])]
    probabilities = [
        np.asarray([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]),
        np.asarray([[0.1, 0.2, 0.7], [0.1, 0.1, 0.8]]),
    ]
    labels, adjusted = apply_postprocessing(
        predictions,
        probabilities,
        classes,
        [0.25, 0.75],
        [1.0, 0.5, 2.0],
        task_type="classification",
    )

    assert labels.tolist() == [2, 2]
    assert adjusted.shape == (2, 3)
    assert np.allclose(adjusted.sum(axis=1), 1.0)


def test_apply_postprocessing_blends_regression_predictions():
    values, probability = apply_postprocessing(
        [np.asarray([1.0, 3.0]), np.asarray([5.0, 7.0])],
        None,
        [],
        [0.25, 0.75],
        None,
        task_type="regression",
    )

    assert values.tolist() == pytest.approx([4.0, 6.0])
    assert probability is None
