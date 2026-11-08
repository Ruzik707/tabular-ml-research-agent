import json

import numpy as np
import pandas as pd
import pytest

from automl_agent.data import discover_task
from automl_agent.evaluation import publish_submission, score_trial, validate_submission


def _task_dir(tmp_path, *, classes, metric, sample_columns, description=""):
    root = tmp_path
    root.mkdir(parents=True, exist_ok=True)
    n = len(classes) * 6
    labels = [classes[i % len(classes)] for i in range(n)]
    pd.DataFrame({"row_id": range(n), "x": np.arange(n), "label": labels}).to_csv(root / "train.csv", index=False)
    pd.DataFrame({"row_id": [101, 102], "x": [n + 1, n + 2]}).to_csv(root / "test.csv", index=False)
    pd.DataFrame({"feature": ["row_id", "x", "label"],
                  "role": ["identifier", "feature", "target"],
                  "dtype": ["integer", "number", "category"]}).to_csv(root / "data_dictionary.csv", index=False)
    sample = {name: [101, 102] if name == "row_id" else [0.0, 0.0] for name in sample_columns}
    pd.DataFrame(sample).to_csv(root / "sample_submission.csv", index=False)
    (root / "task.md").write_text(
        "Classification task.\nMain metric: " + metric + ".\n" + description,
        encoding="utf-8",
    )
    return root / "task.md"


def _write_probability_artifacts(directory, *, valid, test, classes):
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "valid_probabilities.npy", np.asarray(valid, dtype=float), allow_pickle=False)
    np.save(directory / "test_probabilities.npy", np.asarray(test, dtype=float), allow_pickle=False)
    (directory / "probability_classes.json").write_text(
        json.dumps({"classes": list(classes)}), encoding="utf-8"
    )


def test_binary_probability_contract_uses_explicit_positive_class_for_score_and_output(tmp_path):
    task_path = _task_dir(
        tmp_path, classes=[0, 1], metric="ROC-AUC", sample_columns=["row_id", "label"],
        description="Positive class: 0",
    )
    task = discover_task(task_path)
    assert task.output_kind == "binary_probability"
    assert task.positive_class == 0
    assert task.output_columns == ["row_id", "label"]

    split = tmp_path / "split"
    trial = tmp_path / "trial"
    split.mkdir(); trial.mkdir()
    pd.DataFrame({"label": [0, 1, 0, 1]}).to_csv(split / "validation_target.csv", index=False)
    pd.DataFrame({"label": [0, 1, 0, 1]}).to_csv(trial / "valid_predictions.csv", index=False)
    # Deliberately reverse class axis; the positive class is 0, not the last column.
    _write_probability_artifacts(
        trial, valid=[[.9, .1], [.1, .9], [.8, .2], [.2, .8]],
        test=[[.75, .25], [.05, .95]], classes=[0, 1],
    )
    assert score_trial(task, {"validation_target": str(split / "validation_target.csv")}, trial)["roc_auc"] == pytest.approx(1.0)

    pd.DataFrame({"label": [1, 1]}).to_csv(trial / "test_predictions.csv", index=False)
    destination = tmp_path / "submission.csv"
    publish_submission(task, trial / "test_predictions.csv", destination)
    frame = pd.read_csv(destination)
    assert frame.columns.tolist() == ["row_id", "label"]
    assert frame["row_id"].tolist() == [101, 102]
    assert frame["label"].tolist() == pytest.approx([.75, .05])
    assert validate_submission(task, destination)["valid"]


def test_multiclass_probability_columns_map_to_classes_and_keep_sample_order(tmp_path):
    task = discover_task(_task_dir(
        tmp_path, classes=["red", "green", "blue"], metric="LogLoss",
        sample_columns=["row_id", "prob_blue", "red", "prob_green"],
    ))
    assert task.output_kind == "class_probabilities"
    assert task.output_columns == ["row_id", "prob_blue", "red", "prob_green"]
    assert task.probability_column_classes == {"prob_blue": "blue", "red": "red", "prob_green": "green"}

    artifact = tmp_path / "artifact"
    _write_probability_artifacts(
        artifact, valid=np.tile([.2, .3, .5], (3, 1)),
        test=[[.2, .3, .5], [.6, .1, .3]], classes=["red", "green", "blue"],
    )
    pd.DataFrame({"label": ["red", "green"]}).to_csv(artifact / "test_predictions.csv", index=False)
    destination = tmp_path / "submission.csv"
    publish_submission(task, artifact / "test_predictions.csv", destination)
    frame = pd.read_csv(destination)
    assert frame.columns.tolist() == ["row_id", "prob_blue", "red", "prob_green"]
    np.testing.assert_allclose(frame[["prob_blue", "red", "prob_green"]], [[.5, .2, .3], [.3, .6, .1]])
    assert np.allclose(frame[["prob_blue", "red", "prob_green"]].sum(axis=1), 1.0)
    assert validate_submission(task, destination)["valid"]


def test_ambiguous_multiclass_probability_schema_fails_clearly(tmp_path):
    path = _task_dir(
        tmp_path, classes=["cat", "dog", "bird"], metric="LogLoss",
        sample_columns=["row_id", "prob_cat", "prob_dog", "prob_other"],
    )
    with pytest.raises(ValueError, match="probability|вероятност|class|класс"):
        discover_task(path)


def test_probability_publication_rejects_incomplete_class_axis(tmp_path):
    task=discover_task(_task_dir(tmp_path,classes=['red','green','blue'],metric='LogLoss',
                               sample_columns=['row_id','red','green','blue']))
    artifact=tmp_path/'artifact'
    _write_probability_artifacts(artifact,valid=[[.5,.5]],test=[[.5,.5],[.2,.8]],classes=['red','blue'])
    pd.DataFrame({'label':['red','blue']}).to_csv(artifact/'test_predictions.csv',index=False)
    with pytest.raises(ValueError,match='class|класс|axis'):
        publish_submission(task,artifact/'test_predictions.csv',tmp_path/'submission.csv')


def test_probability_schema_rejects_extra_label_column_at_discovery(tmp_path):
    path=_task_dir(tmp_path,classes=['red','green','blue'],metric='LogLoss',
                   sample_columns=['row_id','label','red','green','blue'])
    with pytest.raises(ValueError,match='target|label|схем'):
        discover_task(path)


def test_prediction_ensemble_preserves_mixed_class_types():
    from automl_agent.prediction import PredictionEnsemble
    class Model:
        classes_=[0,'yes']
        def predict_proba(self,X): return np.tile([.8,.2],(len(X),1))
    model=PredictionEnsemble([Model()],'classification',[0,'yes'])
    assert list(model.classes_) == [0,'yes']
    np.testing.assert_allclose(model.predict_proba(pd.DataFrame({'x':[1,2]})),[[.8,.2],[.8,.2]])


def test_multiclass_probability_metric_without_explicit_output_mapping_fails(tmp_path):
    path = _task_dir(
        tmp_path, classes=["cat", "dog", "bird"], metric="LogLoss",
        sample_columns=["row_id", "label"],
    )
    with pytest.raises(ValueError, match="multiclass.*probability|multiclass.*LogLoss"):
        discover_task(path)


def test_binary_positive_class_defaults_to_numeric_one_then_sorted_last(tmp_path):
    task_one = discover_task(_task_dir(
        tmp_path / "one", classes=[0, 1], metric="ROC-AUC", sample_columns=["row_id", "label"]
    ))
    assert task_one.positive_class == 1
    task_sorted = discover_task(_task_dir(
        tmp_path / "sorted", classes=["alpha", "zeta"], metric="LogLoss",
        sample_columns=["row_id", "label"],
    ))
    assert task_sorted.positive_class == "zeta"


def test_meta_logloss_respects_unsorted_class_axis():
    from automl_agent.postprocessing import _score
    probabilities=np.asarray([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]])
    y=np.asarray(['z','a','m'])
    assert _score(y,y,probabilities,['z','a','m'],'log_loss') == pytest.approx(-np.log(.8))


def test_binary_f1_meta_assessment_respects_explicit_positive_label():
    from sklearn.metrics import f1_score
    from automl_agent.postprocessing import select_postprocessing
    y=np.asarray([0,0,0,1]*4)
    prediction=np.asarray([0,1,0,1]*4)
    result=select_postprocessing(y=y,predictions=[prediction],probabilities=None,classes=[0,1],
                                fold_ids=np.arange(16)%2,metric='f1',maximize=True,positive_class=0)
    assert result['baseline_score'] == pytest.approx(f1_score(y,prediction,pos_label=0))


def _regression_task(tmp_path, metric, target_values):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": range(len(target_values)), "x": range(len(target_values)),
                  "value": target_values}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"row_id": [100, 101], "x": [100, 101]}).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"feature": ["row_id", "x", "value"],
                  "role": ["identifier", "feature", "target"],
                  "dtype": ["integer", "number", "number"]}).to_csv(tmp_path / "data_dictionary.csv", index=False)
    pd.DataFrame({"row_id": [100, 101], "value": [0.0, 0.0]}).to_csv(tmp_path / "sample_submission.csv", index=False)
    task_path = tmp_path / "task.md"
    task_path.write_text(f"Regression task. Main metric: {metric}.", encoding="utf-8")
    return task_path


@pytest.mark.parametrize("metric,expected", [
    ("balanced accuracy", "balanced_accuracy"),
    ("MSE", "mse"),
    ("MAPE", "mape"),
    ("RMSLE", "rmsle"),
])
def test_additional_metrics_are_discovered_as_primary(tmp_path, metric, expected):
    if expected == "balanced_accuracy":
        path = _task_dir(tmp_path, classes=["a", "b"], metric=metric,
                         sample_columns=["row_id", "label"])
    else:
        path = _regression_task(tmp_path, metric, [0.0, 1.0, 2.0, 3.0])
    assert discover_task(path).metric == expected


def test_balanced_accuracy_scores_the_requested_metric(tmp_path):
    task = discover_task(_task_dir(
        tmp_path, classes=["a", "b"], metric="balanced accuracy",
        sample_columns=["row_id", "label"],
    ))
    split, trial = tmp_path / "split", tmp_path / "trial"
    split.mkdir(); trial.mkdir()
    pd.DataFrame({"label": ["a", "a", "a", "b"]}).to_csv(split / "validation_target.csv", index=False)
    pd.DataFrame({"label": ["a", "a", "b", "b"]}).to_csv(trial / "valid_predictions.csv", index=False)
    scores = score_trial(task, {"validation_target": str(split / "validation_target.csv")}, trial)
    assert scores["balanced_accuracy"] == pytest.approx(5 / 6)


@pytest.mark.parametrize("metric,truth,predicted,expected", [
    ("MSE", [1.0, 3.0], [2.0, 1.0], 2.5),
    ("MAPE", [0.0, 2.0], [1.0, 1.0], None),
    ("RMSLE", [1.0, 3.0], [-2.0, 5.0], None),
])
def test_host_scores_mse_mape_and_rmsle_with_defined_semantics(tmp_path, metric, truth, predicted, expected):
    from sklearn.metrics import mean_absolute_percentage_error

    task = discover_task(_regression_task(tmp_path, metric, truth + [4.0, 5.0]))
    split, trial = tmp_path / "split", tmp_path / "trial"
    split.mkdir(); trial.mkdir()
    pd.DataFrame({"value": truth}).to_csv(split / "validation_target.csv", index=False)
    pd.DataFrame({"value": predicted}).to_csv(trial / "valid_predictions.csv", index=False)
    scores = score_trial(task, {"validation_target": str(split / "validation_target.csv")}, trial)
    if expected is not None:
        assert scores[task.metric] == pytest.approx(expected)
    elif metric == "MAPE":
        assert scores["mape"] == pytest.approx(mean_absolute_percentage_error(truth, predicted))
    else:
        clipped = np.maximum(predicted, 0.0)
        expected_rmsle = np.sqrt(np.mean((np.log1p(truth) - np.log1p(clipped)) ** 2))
        assert scores["rmsle"] == pytest.approx(expected_rmsle)
        pd.DataFrame({"value": [-1.0, 2.0]}).to_csv(trial / "test_predictions.csv", index=False)
        submission = tmp_path / "submission.csv"
        publish_submission(task, trial / "test_predictions.csv", submission)
        assert pd.read_csv(submission)["value"].tolist() == [0.0, 2.0]


def test_rmsle_rejects_negative_training_targets(tmp_path):
    path = _regression_task(tmp_path, "RMSLE", [-1.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="RMSLE|rmsle"):
        discover_task(path)


def test_holdout_worker_writes_test_probability_matrix_and_class_axis(tmp_path):
    from automl_agent.worker import _run_manifest

    task = discover_task(_task_dir(
        tmp_path, classes=[0, 1], metric="ROC-AUC", sample_columns=["row_id", "label"]
    ))
    train = pd.read_csv(task.train_path)
    test = pd.read_csv(task.test_path)
    valid_features = train[task.feature_columns].iloc[:4]
    inputs = {}
    frames = {
        "train_features": train[task.feature_columns],
        "train_target": train[[task.target]],
        "validation_features": valid_features,
        "test_features": test[task.feature_columns],
    }
    for name, frame in frames.items():
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        inputs[name] = str(path)
    code = tmp_path / "candidate.py"
    code.write_text(
        "from sklearn.ensemble import ExtraTreesClassifier\n"
        "def build_model(context): return ExtraTreesClassifier(n_estimators=8, random_state=4, n_jobs=1)\n",
        encoding="utf-8",
    )
    artifacts = tmp_path / "artifacts"
    manifest = {
        "mode": "evaluate", "context": task.model_context(1), "inputs": inputs,
        "code_path": str(code), "artifact_dir": str(artifacts), "save_model": False,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = _run_manifest(path)
    probabilities = np.load(artifacts / "test_probabilities.npy", allow_pickle=False)
    assert probabilities.shape == (len(test), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert json.loads((artifacts / "probability_classes.json").read_text())["classes"] == [0, 1]
    assert result["test_probabilities"] == "test_probabilities.npy"
    assert result['estimator_details']['training']['train_rows'] == len(train)


def test_postprocessed_ensemble_persists_test_probability_artifacts(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from automl_agent.contracts import ExperimentResult, TaskSpec
    from automl_agent.ensemble import choose_postprocessing

    test_path = tmp_path / "test.csv"
    pd.DataFrame({"key": [10, 11], "x": [1, 2]}).to_csv(test_path, index=False)
    valid = np.array(["a", "b", "a", "b"])
    pd.DataFrame({"label": valid}).to_csv(tmp_path / "validation_target.csv", index=False)
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    pd.DataFrame({"label": valid}).to_csv(trial_dir / "valid_predictions.csv", index=False)
    pd.DataFrame({"label": ["a", "b"]}).to_csv(trial_dir / "test_predictions.csv", index=False)
    np.save(trial_dir / "oof_fold_ids.npy", np.array([0, 0, 1, 1]))
    _write_probability_artifacts(
        trial_dir, valid=[[.9, .1], [.1, .9], [.8, .2], [.2, .8]],
        test=[[.7, .3], [.2, .8]], classes=["a", "b"],
    )
    task = TaskSpec(
        "task.md", "train.csv", str(test_path), "label", "classification", "macro_f1", True,
        ["x"], ["key"], [], ["key", "label"], "hash", classes=["a", "b"],
    )
    result = ExperimentResult(
        "trial_0001", "ok", str(trial_dir), "hash", "candidate", metric=.8,
        recipe={"family": "ExtraTrees"}, model_details={"training": {"split_fingerprint": "split", "estimated_refit_seconds": 1.}},
    )
    monkeypatch.setattr("automl_agent.ensemble.select_postprocessing", lambda **kwargs: {
        "accepted": True, "weights": [1.0], "class_multipliers": [1.0, 1.0],
        "baseline_score": .5, "assessed_score": .6, "calibrated_oof_score": .6,
    })
    plan = choose_postprocessing(
        task, {"mode": "evaluate_cv", "split_fingerprint": "split",
               "validation_target": str(tmp_path / "validation_target.csv"), "strategy": "stratified"},
        [result], tmp_path, remaining_seconds=100, overhead_seconds=1,
    )
    assert plan["accepted"]
    probabilities = np.load(tmp_path / "test_probabilities.npy", allow_pickle=False)
    np.testing.assert_allclose(probabilities, [[.7, .3], [.2, .8]])
    assert json.loads((tmp_path / "probability_classes.json").read_text())["classes"] == ["a", "b"]
