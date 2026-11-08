import json

import pytest

from automl_agent.contracts import ExperimentResult, TaskSpec
from automl_agent.prompts import build_messages


def _essential_payload_case():
    """A synthetic public-multiclass-shaped prompt with four measured bases."""
    labels = [f"class_{index:02d}_" + (chr(ord("a") + index) * 60) for index in range(10)]
    features = [f"public_feature_{index:03d}_" + ("x" * 27) for index in range(50)]
    task = TaskSpec(
        task_path="task.md", train_path="train.csv", test_path="test.csv", target="target",
        task_type="classification", metric="macro_f1", maximize=True,
        feature_columns=features, id_columns=[], categorical_columns=features[:4],
        output_columns=["target"], data_fingerprint="synthetic", classes=labels,
        positive_class=labels[-1], description="Synthetic public multiclass task.",
    )
    profile = {
        "task_type": "classification", "metric": "macro_f1", "feature_count": len(features),
        "train_shape": [100_000, len(features)], "test_shape": [20_000, len(features)],
        "categorical_columns": features[:4],
        "target": {"name": "target", "dtype": "object", "unique": len(labels),
                   "class_counts": {label: 10_000 - index for index, label in enumerate(labels)}},
        "available_packages": {"catboost": "1.2", "scikit-learn": "1.5", "lightgbm": "4.0", "xgboost": "2.0"},
        "validation_plan": {"strategy": "stratified", "fold_count": 5, "coverage": 1.0},
    }
    recipes = [
        ("CatBoost", "trial_cat", {"iterations": 1200, "learning_rate": 0.035}),
        ("ExtraTrees", "trial_extra", {"n_estimators": 500, "min_samples_leaf": 2}),
        ("LightGBM", "trial_lgbm", {"n_estimators": 900, "num_leaves": 31}),
        ("XGBoost", "trial_xgb", {"n_estimators": 700, "max_depth": 7}),
    ]
    results = []
    for family, trial_id, params in recipes:
        recipe = {"family": family, "parameters": params, "preprocessing": {}}
        results.append(ExperimentResult(
            trial_id=trial_id, status="ok", artifact_dir="/missing", code_hash=trial_id,
            hypothesis=f"Measured {family} synthetic base", metric=0.82,
            model_details={"class": f"{family}Classifier", "parameters": params}, recipe=recipe,
            diagnostics={"total_classes": len(labels),
                         "per_class": [{"label": label, "precision": 0.8, "recall": 0.7,
                                        "f1": 0.74, "support": 100}
                                       for label in labels]},
        ))
    # Keep the exact best recipe and its model parameters visible in best_experiment.
    results[0].recipe["preprocessing"]["feature_ops"] = [
        {"feature": feature, "operation": "identity"} for feature in features[:5]
    ]
    return task, profile, results, results[0], recipes


def test_compact_recipe_prompt_spills_essential_payload_to_hard_cap():
    task, profile, results, best, recipes = _essential_payload_case()

    messages = build_messages(
        task, profile, results, best,
        {"n_jobs": 4, "parallel_trials": 2, "experiment_mode": "recipe"},
        context_chars=48_000, compact=True,
    )

    content = json.loads(messages[1]["content"])
    assert len(messages[1]["content"]) > 9_000
    assert len(messages[1]["content"]) <= 12_000
    assert content["task"]["metric"] == "macro_f1"
    assert content["task"]["positive_class"] == task.positive_class
    assert content["task"]["classes"] == task.classes
    measured = {row["family"]: row for row in content["model_catalog"]["measured_bases"]}
    for family, trial_id, params in recipes:
        assert measured[family]["trial_id"] == trial_id
    assert content["best_experiment"]["recipe"] == best.recipe
    assert content["best_experiment"]["model_details"]["parameters"] == best.model_details["parameters"]
    assert [row["label"] for row in content["best_experiment"]["diagnostics"]["per_class"]] == task.classes


def test_compact_recipe_prompt_still_honors_smaller_explicit_context_cap():
    task, profile, results, best, _ = _essential_payload_case()

    with pytest.raises(ValueError, match="Recipe prompt essentials require .* above the 8000-character budget"):
        build_messages(
            task, profile, results, best,
            {"n_jobs": 4, "parallel_trials": 2, "experiment_mode": "recipe"},
            context_chars=16_000, compact=True,
        )
