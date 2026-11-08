"""Metric decisions must be assessed out of fold and survive final refit."""
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from automl_agent.ensemble import choose_postprocessing
from automl_agent.contracts import ExperimentResult, TaskSpec
from automl_agent.postprocessing import apply_postprocessing, select_postprocessing


def _select(y, p, folds, *, classes=None, positive="alert", temporal=False, metric="f1"):
    classes = ["alert", "normal"] if classes is None else classes
    p = np.asarray(p, dtype=float)
    # The positive class deliberately precedes the negative class in the axis.
    probability = np.column_stack([p, 1 - p])
    hard = np.asarray(classes)[probability.argmax(axis=1)]
    return select_postprocessing(
        y=np.asarray(y), predictions=[hard], probabilities=[probability],
        classes=classes, fold_ids=np.asarray(folds), metric=metric,
        maximize=metric != "log_loss", positive_class=positive, temporal=temporal,
    )


def _brute_threshold(y, p, positive="alert"):
    """Independent exhaustive reference; equal scores prefer larger cutoffs."""
    actual = np.asarray(y) == positive
    scores = [(f1_score(actual, np.asarray(p) >= t, zero_division=0), t)
              for t in sorted(set(p), reverse=True)]
    return max(scores, key=lambda pair: pair[0])


def test_binary_f1_threshold_improves_cross_fitted_folds_with_positive_axis_first():
    y = np.tile(["normal", "alert", "normal", "alert"], 4)
    p = np.tile([0.1, 0.3, 0.1, 0.4], 4)
    folds = np.repeat(np.arange(4), 4)
    result = _select(y, p, folds)

    assert result["accepted"] is True  # The old argmax-only F1 selector cannot improve.
    assert result["class_multipliers"] is None
    assert result["binary_threshold"] == pytest.approx(0.3)
    assert result["positive_class"] == "alert"
    assert result["baseline_score"] == 0
    assert result["assessed_score"] == 1
    probability = np.column_stack([p, 1 - p])
    labels, unchanged = apply_postprocessing(
        [np.repeat("normal", len(y))], [probability], ["alert", "normal"],
        result["weights"], None, "classification",
        binary_threshold=result["binary_threshold"], positive_class=result["positive_class"],
    )
    np.testing.assert_array_equal(labels, y)
    np.testing.assert_array_equal(unchanged, probability)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("seed", range(8))
def test_exact_threshold_matches_exhaustive_search_with_ties_and_boundaries(seed):
    rng = np.random.default_rng(seed)
    p = np.r_[0., 1., rng.integers(0, 11, size=48) / 10.]
    y = np.where(rng.integers(0, 2, size=len(p)), "alert", "normal")
    result = _select(y, p, np.zeros(len(p), dtype=int))
    reference_score, _ = _brute_threshold(y, p)
    assert result["calibrated_oof_score"] == pytest.approx(reference_score)
    assert result["accepted"] is False  # Fitting one fold never proves generalization.
    threshold = result["binary_threshold"]
    if threshold is not None:
        assert f1_score(y == "alert", p >= threshold, zero_division=0) == pytest.approx(reference_score)


@pytest.mark.parametrize("temporal", [False, True])
def test_each_threshold_is_trained_without_assessment_fold(temporal):
    p = np.array([.1, .2, .3, .4, .1, .2, .3, .4, .1, .2, .3, .4])
    y = np.array(["alert", "alert", "normal", "normal",
                  "normal", "normal", "alert", "alert",
                  "normal", "alert", "normal", "alert"])
    folds = np.repeat(np.arange(3), 4)
    result = _select(y, p, folds, temporal=temporal)
    truth, heldout = [], []
    for row in result["fold_scores"]:
        train = np.isin(folds, row["train_folds"])
        test = folds == row["fold"]
        expected_score, expected_threshold = _brute_threshold(y[train], p[train])
        # Every baseline prediction is negative, so each positive F1 gain installs a threshold.
        assert expected_score > 0
        assert row["binary_threshold"] == expected_threshold
        prediction = np.where(p[test] >= expected_threshold, "alert", "normal")
        assert row["assessed_score"] == pytest.approx(f1_score(
            y[test], prediction, pos_label="alert", zero_division=0))
        truth.extend(y[test]); heldout.extend(prediction)
        if temporal:
            assert max(row["train_folds"]) < row["fold"]
    assert result["assessed_score"] == pytest.approx(f1_score(
        truth, heldout, pos_label="alert", zero_division=0))


def test_full_oof_threshold_gain_is_rejected_if_it_fails_on_excluded_folds():
    # Incompatible fold optima: the full OOF fit improves while independent
    # threshold application actually makes the pooled F1 worse.
    p = [.7, .7, .4, .4, .6, .1, .9, .7, .1, .7, .9, .9]
    y = np.where([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0], "alert", "normal")
    result = _select(y, p, np.repeat([0, 1], 6))
    assert result["calibrated_oof_score"] > result["baseline_score"]
    assert result["assessed_score"] < result["baseline_score"]
    assert result["accepted"] is False


@pytest.mark.parametrize("metric", ["roc_auc", "log_loss", "accuracy", "weighted_f1", "balanced_accuracy"])
def test_non_binary_f1_metrics_do_not_install_binary_threshold(metric):
    result = _select(np.tile(["normal", "alert"], 4), np.tile([.1, .4], 4),
                     np.repeat([0, 1], 4), metric=metric)
    assert result.get("binary_threshold") is None


def test_numeric_one_is_default_positive_even_if_first_in_class_axis():
    result = _select(np.tile([0, 1], 4), np.tile([.1, .3], 4),
                     np.repeat([0, 1], 4), classes=[1, 0], positive=None)
    assert result["accepted"] is True
    assert result["positive_class"] == 1
    assert result["binary_threshold"] == pytest.approx(.3)


def test_reversing_class_axis_preserves_semantic_threshold_and_score():
    y = np.tile(["normal", "alert"], 4)
    p = np.tile([.1, .3], 4)
    first = _select(y, p, np.repeat([0, 1], 4))
    reversed_axis = select_postprocessing(
        y=y, predictions=[np.repeat("normal", len(y))],
        probabilities=[np.column_stack([1-p, p])], classes=["normal", "alert"],
        fold_ids=np.repeat([0, 1], 4), metric="f1", maximize=True, positive_class="alert",
    )
    assert first["binary_threshold"] == reversed_axis["binary_threshold"]
    assert first["assessed_score"] == reversed_axis["assessed_score"] == 1
    assert first["positive_class"] == reversed_axis["positive_class"] == "alert"


@pytest.mark.parametrize("temporal", [False, True])
def test_fold_without_positive_meta_labels_does_not_borrow_a_future_threshold(temporal):
    y = np.array(["normal", "normal", "alert", "normal"])
    p = np.array([.1, .1, .3, .1])
    result = _select(y, p, [0, 0, 1, 1], temporal=temporal)
    row = next(row for row in result["fold_scores"] if row["fold"] == 1)
    assert row["train_folds"] == [0]
    assert row["binary_threshold"] is None
    assert row["assessed_score"] == 0
    assert result["calibrated_oof_score"] == 1
    assert result["accepted"] is False


def test_no_probabilities_or_no_positive_labels_leave_threshold_disabled():
    result = select_postprocessing(
        y=np.tile([0, 1], 4), predictions=[np.zeros(8, dtype=int)], probabilities=None,
        classes=[0, 1], fold_ids=np.repeat([0, 1], 4), metric="f1", maximize=True,
    )
    assert result["binary_threshold"] is None
    assert result["accepted"] is False
    absent = _select(np.repeat("normal", 8), np.repeat(.1, 8), np.repeat([0, 1], 4))
    assert absent["binary_threshold"] is None
    assert absent["accepted"] is False


def test_binary_decision_preserves_mixed_scalar_label_identity():
    probability = np.array([[.9, .1], [.1, .9]])
    labels, unchanged = apply_postprocessing(
        [np.array([0, "yes"], dtype=object)], [probability], [0, "yes"], [1.], None,
        "classification", binary_threshold=.5, positive_class=0,
    )
    assert labels.tolist() == [0, "yes"]
    assert isinstance(labels[0], int)
    np.testing.assert_array_equal(unchanged, probability)


def test_legacy_prediction_ensemble_without_new_attributes_keeps_argmax():
    from automl_agent.prediction import PredictionEnsemble
    model = SimpleNamespace(classes_=np.array(["alert", "normal"]),
                            predict_proba=lambda X: np.tile([.3, .7], (len(X), 1)))
    ensemble = PredictionEnsemble([model], "classification", classes=["alert", "normal"])
    del ensemble.binary_threshold
    del ensemble.positive_class
    assert ensemble.predict(np.zeros((3, 1))).tolist() == ["normal"] * 3


def test_host_applies_selected_threshold_to_test_labels_but_keeps_probabilities(tmp_path):
    y = np.tile(["normal", "alert"], 6)
    p = np.tile([.1, .3], 6)
    pd.DataFrame({"target": y}).to_csv(tmp_path / "valid_y.csv", index=False)
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(test_path, index=False)
    task = TaskSpec("task.md", "train.csv", str(test_path), "target", "classification", "f1", True,
                    ["x"], [], [], ["target"], "hash", classes=["alert", "normal"], positive_class="alert")
    artifact = tmp_path / "candidate"
    artifact.mkdir()
    pd.DataFrame({"target": np.repeat("normal", len(y))}).to_csv(artifact / "valid_predictions.csv", index=False)
    pd.DataFrame({"target": ["normal"] * 3}).to_csv(artifact / "test_predictions.csv", index=False)
    np.save(artifact / "oof_fold_ids.npy", np.repeat(np.arange(3), 4))
    np.save(artifact / "valid_probabilities.npy", np.column_stack([p, 1 - p]))
    test_probability = np.array([[.3, .7], [.2, .8], [.4, .6]])
    np.save(artifact / "test_probabilities.npy", test_probability)
    (artifact / "probability_classes.json").write_text(json.dumps({"classes": ["alert", "normal"]}))
    result = ExperimentResult("trial_0001", "ok", str(artifact), "code", "threshold", metric=0.,
                              model_details={"training": {"split_fingerprint": "same", "estimated_refit_seconds": 1.}})
    plan = choose_postprocessing(task, {"mode": "evaluate_cv", "split_fingerprint": "same",
                                "validation_target": str(tmp_path / "valid_y.csv"), "strategy": "stratified"},
                                [result], tmp_path, remaining_seconds=100, overhead_seconds=1)
    assert plan["accepted"] is True
    assert pd.read_csv(plan["prediction_path"])["target"].tolist() == ["alert", "normal", "alert"]
    np.testing.assert_array_equal(np.load(plan["test_probabilities_path"]), test_probability)

    # Exercise the actual host manifest builder and supervised worker, rather
    # than reconstructing a final manifest manually in the test.
    from automl_agent.agent import ResearchRun
    source = artifact / "code.py"
    source.write_text('''import numpy as np
class Candidate:
    def fit(self, X, y):
        self.classes_ = np.array(['alert', 'normal'])
        return self
    def predict_proba(self, X):
        p = X['x'].to_numpy()
        return np.column_stack([p, 1-p])
    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
def build_model(context):
    return Candidate()
''')
    result.code_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    full_features = tmp_path / "full_x.csv"
    pd.DataFrame({"x": p}).to_csv(full_features, index=False)
    test_features = pd.DataFrame({"x": test_probability[:, 0]})
    test_features.to_csv(test_path, index=False)
    research = ResearchRun.__new__(ResearchRun)
    research.directory = tmp_path
    research.task = task
    research.config = SimpleNamespace(n_jobs=1, memory_limit_mb=2000)
    research.budget = SimpleNamespace(remaining=lambda: 45.)
    from automl_agent.budgeting import plan_finalization
    result.seconds = 1.  # A successful measured trial supplies a complete admission plan.
    research.reserve_plan = plan_finalization(total_seconds=60., validation_fraction=.2, best=result)
    research.postprocessing_plan = plan
    research.results = [result]
    research.split = {"full_train_features": str(full_features),
                      "full_train_target": str(tmp_path / "valid_y.csv"), "test_features": str(test_path)}
    research.output = tmp_path / "submission.csv"
    research.checkpoint = lambda: None
    research.journal = SimpleNamespace(append=lambda *a, **kw: None, redact=lambda value: value)
    final = research.finalize(result)
    assert final["status"] == "ok", final
    manifest = json.loads((Path(final["artifact_dir"]) / "manifest.json").read_text())
    assert manifest["mode"] == "finalize_ensemble"
    assert manifest["binary_threshold"] == pytest.approx(.3)
    assert manifest["positive_class"] == "alert"
    restored = joblib.load(final["model_path"])
    published = pd.read_csv(research.output)["target"].to_numpy()
    np.testing.assert_array_equal(published, ["alert", "normal", "alert"])
    np.testing.assert_array_equal(restored.predict(test_features), published)
    np.testing.assert_array_equal(restored.predict_proba(test_features), test_probability)
