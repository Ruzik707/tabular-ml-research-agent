import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from automl_agent.contracts import ExperimentResult, TaskSpec
from automl_agent.prompts import build_messages


def _task(feature_count=200, description_size=5000):
    features = [f"feature_{index:03d}" for index in range(feature_count)]
    return TaskSpec(
        task_path="task.md", train_path="train.csv", test_path="test.csv", target="label",
        task_type="classification", metric="macro_f1", maximize=True,
        feature_columns=features, id_columns=[], categorical_columns=features[:20],
        output_columns=["label"], data_fingerprint="fingerprint", description="Task context " * description_size,
        classes=["common", "rare"], seed=42,
    )


def _diagnostics(trial_id, count=200):
    return {
        "validation": {
            "strategy": "forward_group", "coverage": 0.91, "mean": 0.73, "std": 0.04,
            "folds": [{"fold": i, "rows": 1000, "metric": 0.7 + i / 100, "training_seconds": 9000} for i in range(5)],
            "warning": f"validation marker {trial_id}",
        },
        "per_class": [
            {"label": f"class_{trial_id}_{i:03d}_" + "label" * 8, "precision": 0.2, "recall": 0.3,
             "f1": i / max(count, 1), "support": 10} for i in range(count)
        ],
        "total_classes": count,
        "feature_importance": {
            "method": "native", "interpretation": "Association only",
            "items": [{"feature": f"feature_{i:03d}", "importance": 1 / (i + 1)} for i in range(count)],
        },
    }


def _results(count=30, latest_status="invalid_code"):
    results = []
    for index in range(count):
        trial_id = f"trial_{index + 1:04d}"
        is_best = index == 0
        latest = index == count - 1
        recipe = None
        details = {"class": "sklearn.ensemble.ExtraTreesClassifier", "parameters": {"n_estimators": 180}}
        if is_best:
            recipe = {"family": "CatBoost", "parameters": {"iterations": 1234, "learning_rate": 0.48828},
                      "preprocessing": {"add_missing_indicators": False}, "fit_options": {"early_stopping_rounds": 0}}
            details = {"class": "catboost.CatBoostClassifier",
                       "parameters": {"iterations": 1234, "learning_rate": 0.48828, "auto_class_weights": "None"},
                       "parameters_source": "catboost.get_all_params",
                       "training": {"folds": [{"training_seconds": 9999, "best_iteration": 1234} for _ in range(5)]}}
        status = latest_status if latest else "ok"
        results.append(ExperimentResult(
            trial_id=trial_id, status=status, artifact_dir="/missing/artifact", code_hash=f"hash-{index}",
            hypothesis=f"hypothesis {index}: " + "detail " * 80, metric=(0.91 if is_best else 0.5 + index / 1000),
            seconds=3.5, error=("Traceback (most recent call last):\n  File \"worker.py\"\nRuntimeError: LATEST_ERROR_MARKER " + "context " * 500) if latest and latest_status != "ok" else None,
            recipe=recipe, model_details=details,
            diagnostics=_diagnostics(trial_id) if is_best or latest else _diagnostics(trial_id, 60),
        ))
    return results


def _profile(feature_count=200):
    feature_names = [f"feature_{index:03d}" for index in range(feature_count)]
    return {
        "task_type": "classification", "metric": "macro_f1", "train_shape": [50000, feature_count],
        "test_shape": [10000, feature_count], "feature_count": feature_count,
        "categorical_columns": feature_names[:20],
        "target": {"name": "label", "dtype": "object", "unique": 2,
                   "class_counts": {"common": 48000, "rare": 2000}},
        "available_packages": {"catboost": "1.2.8", "lightgbm": "4.6.0", "xgboost": "2.1.4", "scikit-learn": "1.5.2"},
        "validation_plan": {"strategy": "forward_group", "fold_count": 5, "coverage": 0.91,
                            "time_column": "feature_000", "warnings": ["forward-only marker"]},
        "features": {name: {"dtype": "float64", "missing_fraction": i / 250,
                            "unique": 500 + i, "non_finite": i % 2, "numeric": True,
                            "unused_detail": "x" * 2000} for i, name in enumerate(feature_names)},
        "dataset_diagnostics": {
            "temporal_candidates": [{"feature": "feature_000", "future_only": True, "score": 0.99}],
            "adversarial_cv": {"status": "ok", "metric": "roc_auc", "roc_auc_mean": 0.97,
                               "roc_auc_std": 0.01, "fold_roc_auc": [0.96, 0.97, 0.98],
                               "feature_names": feature_names, "feature_count": feature_count,
                               "huge_unused": "y" * 20000},
            "features": {name: {"shift": i / 100, "missing_delta": i / 1000, "unused": "z" * 1000}
                         for i, name in enumerate(feature_names)},
            "total_features": feature_count, "features_partial": False,
        },
        "research_strategy": {"facts": {"feature_count": feature_count, "unused": "q" * 20000},
                              "candidate_shortlist": [{"model_family": name, "reason": "rationale " * 100}
                                                      for name in ("CatBoost", "ExtraTrees", "Linear")]},
        "family_comparison": {
            "selected_trial": "trial_0001", "metric": "macro_f1",
            "trials": [{"family": "CatBoost", "trial_id": "trial_0001", "status": "ok", "metric": 0.91,
                        "seconds": 10, "model_details": {"parameters": {"learning_rate": 0.48828},
                                                          "training": {"folds": [{"training_seconds": 9999} for _ in range(5)]}}}],
        },
    }


@pytest.mark.parametrize(("context_chars", "compact"), [(48000, False), (24000, True)])
def test_recipe_prompt_is_hard_bounded_and_preserves_best_and_latest_evidence(context_chars, compact):
    task, profile, results = _task(), _profile(), _results()
    best = results[0]
    task_before, profile_before, results_before = copy.deepcopy(task.to_dict()), copy.deepcopy(profile), copy.deepcopy(results)
    limit = max(5000, context_chars // 2) if compact else max(5000, context_chars)

    messages = build_messages(task, profile, results, best,
                              {"n_jobs": 4, "parallel_trials": 2, "experiment_mode": "recipe"},
                              feedback="last tool feedback", context_chars=context_chars, compact=compact)
    content = json.loads(messages[1]["content"])

    assert len(messages[1]["content"]) <= limit
    assert content["task"]["task_type"] == "classification"
    assert content["task"]["metric"] == "macro_f1"
    assert "CLASSIFICATION" in content["task_constraints"]
    assert content["data_profile"]["target"]["class_counts"] == {"common": 48000, "rare": 2000}
    assert content["data_profile"]["validation_plan"]["coverage"] == 0.91
    assert content["data_profile"]["validation_plan"]["strategy"] == "forward_group"
    assert content["data_profile"]["dataset_diagnostics"]["adversarial_cv"]["roc_auc_mean"] == 0.97
    assert content["best_experiment"]["trial_id"] == "trial_0001", "best must remain available outside the last eight"
    assert content["best_experiment"]["recipe"]["parameters"]["learning_rate"] == 0.48828
    assert content["best_experiment"]["model_details"]["parameters"]["learning_rate"] == 0.48828
    assert any(row["label"].startswith("class_trial_0001") for row in content["best_experiment"]["diagnostics"]["per_class"])
    assert content["best_experiment"]["diagnostics"]["feature_importance"]["items"]
    assert "LATEST_ERROR_MARKER" in content["latest_experiment"]["error"]
    assert "training_seconds" not in json.dumps(content["family_comparison"])
    assert all("diagnostics" not in row for row in content["experiments"] if row["trial_id"] not in {"trial_0001", "trial_0030"})

    assert task.to_dict() == task_before
    assert profile == profile_before
    assert results == results_before


def test_recipe_prompt_drops_stale_errors_after_latest_success():
    task, profile, results = _task(10, 100), _profile(10), _results(10, latest_status="ok")
    best = results[0]

    messages = build_messages(task, profile, results, best,
                              {"n_jobs": 2, "experiment_mode": "recipe"}, context_chars=24000, compact=True)
    content = json.loads(messages[1]["content"])

    assert content["latest_experiment"]["status"] == "ok"
    assert "error" not in content["latest_experiment"]
    assert "LATEST_ERROR_MARKER" not in messages[1]["content"]


def test_recipe_catalog_prefers_best_measured_base_and_prompt_runs_to_reserve():
    task, profile, results = _task(10, 100), _profile(10), _results(3)
    results[1].recipe = {"family": "CatBoost", "parameters": {"iterations": 900}, "preprocessing": {}}
    results[1].model_details = {"parameters": {"iterations": 900, "learning_rate": 0.05}}

    messages = build_messages(task, profile, results, results[0],
                              {"n_jobs": 2, "experiment_mode": "recipe"}, compact=True)
    content = json.loads(messages[1]["content"])
    measured = next(row for row in content["model_catalog"]["measured_bases"] if row["family"] == "CatBoost")

    assert measured["trial_id"] == "trial_0001"
    assert measured["parameters"]["learning_rate"] == 0.48828
    assert "Continue proposing measured experiments until the reserved deadline" in messages[0]["content"]
    assert "Stop at the reserved deadline or after measured convergence" not in messages[0]["content"]


def test_measured_bases_keep_requested_tree_cap_preprocessing_and_fit_weights():
    task, profile, results = _task(10, 100), _profile(10), _results(3)
    results[1].recipe={'family':'LightGBM','parameters':{'n_estimators':5000},
        'preprocessing':{'add_missing_indicators':True},
        'fit_options':{'early_stopping_rounds':100,'sample_weight':'sqrt_balanced'}}
    results[1].model_details={'parameters':{'n_estimators':148,'class_weight':None}}
    messages=build_messages(task,profile,results,results[0],{'n_jobs':4,'experiment_mode':'recipe'},compact=True)
    catalog=json.loads(messages[1]['content'])['model_catalog']
    base=next(row for row in catalog['measured_bases'] if row['family']=='LightGBM')
    assert base['parameters']['n_estimators']==148
    assert base['recipe']['parameters']['n_estimators']==5000
    assert base['recipe']['fit_options']['sample_weight']=='sqrt_balanced'
    assert base['recipe']['preprocessing']['add_missing_indicators'] is True


def test_recipe_catalog_excludes_families_without_a_successful_base():
    task, profile, results = _task(10, 100), _profile(10), _results(2)
    results[0].recipe={'family':'XGBoost','parameters':{'n_estimators':3000},'preprocessing':{}}
    messages=build_messages(task,profile,results,results[0],{'n_jobs':4,'experiment_mode':'recipe'},compact=True)
    catalog=json.loads(messages[1]['content'])['model_catalog']
    available=set(catalog.get('families',{}))|set(catalog.get('boosting_families',{}))
    assert available=={'XGBoost'}
    assert 'CatBoost auto_class_weights=' not in messages[0]['content']
    assert 'fit_options.sample_weight' in messages[0]['content']
    assert 'family is fixed by base_trial_id' in messages[0]['content']


def test_recipe_diagnostics_keep_confusion_axis_partial_flag_and_nested_auc():
    task, profile, results = _task(10, 100), _profile(10), _results(2)
    labels = [f"class_{index}" for index in range(10)]
    results[0].diagnostics.update({
        "partial": True,
        "total_classes": 10,
        "per_class": [{"label": label, "precision": 0.5, "recall": 0.4, "f1": 0.44, "support": 100}
                      for label in labels],
        "confusion_matrix": {"labels": labels, "matrix": [[int(i == j) for j in range(10)] for i in range(10)],
                             "total_classes": 10, "partial": True, "selection": "lowest_f1_first"},
    })
    profile["dataset_diagnostics"]["adversarial_cv"]["without_temporal"] = {
        "status": "ok", "metric": "roc_auc", "roc_auc_mean": 1.0, "roc_auc_std": 0.0, "folds": 3,
        "private_payload": "discard me",
    }

    messages = build_messages(task, profile, results, results[0],
                              {"n_jobs": 2, "experiment_mode": "recipe"}, compact=True)
    content = json.loads(messages[1]["content"])
    diagnostics = content["best_experiment"]["diagnostics"]
    matrix = diagnostics["confusion_matrix"]

    assert matrix["labels"] == labels
    assert len(matrix["matrix"]) == 10 and all(len(row) == 10 for row in matrix["matrix"])
    assert [row["label"] for row in diagnostics["per_class"]] == labels
    assert matrix["partial"] is True and diagnostics["partial"] is True
    assert content["data_profile"]["dataset_diagnostics"]["adversarial_cv"]["without_temporal"]["roc_auc_mean"] == 1.0
    assert "private_payload" not in json.dumps(content)


def test_recipe_profile_compacts_real_numeric_and_categorical_shift_fields():
    task, profile, results = _task(10, 100), _profile(10), _results(2)
    profile["dataset_diagnostics"]["features"] = {
        "amount": {"kind": "numeric", "train_missing_rate": 0.1, "test_missing_rate": 0.3,
                   "train_mean": 10.0, "test_mean": 20.0, "train_min": 0.0, "test_max": 99.0,
                   "mean_difference": 10.0, "raw_rows": ["drop"] * 200},
        "region": {"kind": "categorical", "train_missing_rate": 0.0, "test_missing_rate": 0.1,
                   "category_shift": {"train_only_count": 4, "test_only_count": 7,
                                      "test_only_top": [{"value": "new-region", "count": 12}] * 20}},
    }

    messages = build_messages(task, profile, results, results[0],
                              {"n_jobs": 2, "experiment_mode": "recipe"}, compact=True)
    content = json.loads(messages[1]["content"])
    shifts = content["data_profile"]["dataset_diagnostics"]["feature_shift"]

    assert shifts["amount"]["train_missing_rate"] == 0.1
    assert shifts["amount"]["test_mean"] == 20.0
    assert shifts["region"]["category_shift"]["test_only_count"] == 7
    assert len(json.dumps(shifts)) < 1000


def test_recipe_payload_hard_bound_with_thousand_trials_keeps_valid_catalog_and_best():
    task, profile = _task(200, 5000), _profile(200)
    profile["family_comparison"]["trials"] = [
        {"family": "CatBoost", "trial_id": f"base_{index:04d}", "status": "ok", "metric": 0.5,
         "seconds": 20, "model_details": {"parameters": {"trace": "x" * 100}}}
        for index in range(1000)
    ]
    results = []
    for index in range(1000):
        best = index == 0
        latest = index == 999
        results.append(ExperimentResult(
            trial_id=f"trial_{index + 1:04d}", status="invalid_code" if latest else "ok",
            artifact_dir="/missing/artifact", code_hash=f"hash-{index}",
            hypothesis=f"hypothesis {index}: " + "evidence " * 100,
            metric=0.91 if best else 0.5, error=("Latest failure marker " + "traceback " * 1000) if latest else None,
            recipe=({"family": "CatBoost", "parameters": {"iterations": 1234, "learning_rate": 0.48828},
                     "preprocessing": {"add_missing_indicators": False}} if best else None),
            model_details=({"class": "catboost.CatBoostClassifier", "parameters": {"iterations": 1234, "learning_rate": 0.48828}}
                           if best else {}),
            diagnostics=(_diagnostics("trial_0001", 9) if best else {}),
        ))
    best = results[0]
    limit = 12_000

    messages = build_messages(task, profile, results, best,
                              {"n_jobs": 4, "parallel_trials": 2, "experiment_mode": "recipe"},
                              context_chars=24_000, compact=True)
    content = json.loads(messages[1]["content"])

    assert len(messages[1]["content"]) <= limit
    assert content["best_experiment"]["trial_id"] == "trial_0001"
    assert content["best_experiment"]["recipe"]["parameters"]["learning_rate"] == 0.48828
    assert content["best_experiment"]["model_details"]["parameters"]["learning_rate"] == 0.48828
    assert content["data_profile"]["target"]["class_counts"] == {"common": 48000, "rare": 2000}
    assert content["data_profile"]["validation_plan"]["strategy"] == "forward_group"
    assert "Latest failure marker" in content["latest_experiment"]["error"]
    assert len(content["experiments"]) <= 13
    assert content["model_catalog"]["task_type"] == "classification"
    assert "families" in content["model_catalog"]
    assert "family_comparison" not in content, "overflow fallback should shed secondary comparisons, not core facts"


def test_code_mode_keeps_its_existing_code_context_contract(tmp_path):
    task, profile, results = _task(2, 10), _profile(2), _results(2, latest_status="ok")
    code_path = Path(results[0].artifact_dir)
    # Keep the fixture self-contained without creating or reading a real artifact.
    results[0].artifact_dir = str(tmp_path)
    (tmp_path / "code.py").write_text("def build_model(context):\n    return None\n", encoding="utf-8")

    messages = build_messages(task, profile, results, results[0],
                              {"n_jobs": 2, "experiment_mode": "code"}, context_chars=16000)
    content = json.loads(messages[1]["content"])

    assert "build_model(context)" in content["best_code"]
    assert "model_catalog" not in content
