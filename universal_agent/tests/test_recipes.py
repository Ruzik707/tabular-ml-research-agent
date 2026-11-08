import ast
import json
import sys
import types

import cloudpickle
import joblib
import numpy as np
import pandas as pd
import pytest

from automl_agent import recipes
from automl_agent.tools import validate_candidate_code


@pytest.mark.parametrize(
    ("family", "task_type", "expected"),
    [
        ("CatBoost", "classification", {"iterations": 150, "auto_class_weights": None}),
        (
            "ExtraTrees",
            "classification",
            {
                "n_estimators": 180,
                "max_leaf_nodes": 128,
                "min_samples_leaf": 3,
                "max_features": 0.8,
                "max_depth": None,
                "class_weight": None,
            },
        ),
        (
            "Linear",
            "classification",
            {"C": 0.5, "max_iter": 300, "solver": "saga", "class_weight": None},
        ),
        ("Linear", "regression", {"alpha": 1.0}),
    ],
)
def test_default_recipes_are_normalized_json(family, task_type, expected):
    recipe = recipes.default_recipe(family, task_type)

    assert recipe == {
        "family": family,
        "parameters": expected,
        "preprocessing": {"add_missing_indicators": False},
    }
    assert json.loads(json.dumps(recipe)) == recipe


@pytest.mark.parametrize(
    ("family", "task_type", "recipe", "expected"),
    [
        (
            "ExtraTrees",
            "classification",
            {"family": "ExtraTrees", "parameters": {"n_estimators": 250}},
            {
                "family": "ExtraTrees",
                "parameters": {
                    "n_estimators": 250,
                    "max_leaf_nodes": 128,
                    "min_samples_leaf": 3,
                    "max_features": 0.8,
                    "max_depth": None,
                    "class_weight": None,
                },
                "preprocessing": {"add_missing_indicators": False},
            },
        ),
        (
            "CatBoost",
            "classification",
            {"family": "CatBoost", "parameters": {"iterations": 225, "depth": 7}},
            {
                "family": "CatBoost",
                "parameters": {"iterations": 225, "auto_class_weights": None, "depth": 7},
                "preprocessing": {"add_missing_indicators": False},
            },
        ),
    ],
)
def test_normalize_recipe_fills_defaults_and_preserves_explicit_dynamic_values(
    family, task_type, recipe, expected
):
    assert recipes.normalize_recipe(recipe, task_type) == expected


def test_catboost_string_none_normalizes_to_unweighted_default_and_same_fingerprint():
    default = recipes.default_recipe("CatBoost", "classification")
    realized = {
        "family": "CatBoost",
        "parameters": {"auto_class_weights": "None"},
    }

    normalized_realized = recipes.normalize_recipe(realized, "classification")
    assert normalized_realized == default
    assert recipes.recipe_fingerprint(realized, "classification") == recipes.recipe_fingerprint(
        default, "classification"
    )


def test_sample_weight_error_guides_to_catboost_and_sklearn_class_weight_fields():
    recipe = {"family": "CatBoost", "parameters": {"sample_weight": [1, 2]}}

    with pytest.raises(ValueError) as error:
        recipes.normalize_recipe(recipe, "classification")

    assert "auto_class_weights" in str(error.value)
    assert "class_weight" in str(error.value)


@pytest.mark.parametrize(
    ("task_type", "family", "parameters", "hint"),
    [
        ("classification", "CatBoost", {"class_weights": "Balanced"}, "auto_class_weights"),
        ("classification", "ExtraTrees", {"sample_weight": [1, 2]}, "class_weight"),
        ("classification", "CatBoost", {"early_stopping_rounds": 20}, "eval_set"),
        ("classification", "ExtraTrees", {"thread_count": 8}, "host-controlled"),
        ("classification", "CatBoost", {"iterations": 0}, "iterations"),
        ("classification", "ExtraTrees", {"max_depth": -1}, "max_depth"),
        ("classification", "Linear", {"C": float("nan")}, "C"),
        ("regression", "ExtraTrees", {"class_weight": "balanced"}, "classification"),
        ("regression", "CatBoost", {"auto_class_weights": "Balanced"}, "classification"),
        ("classification", "ExtraTrees", {"n_estimators": True}, "integer"),
        ("classification", "Linear", {"solver": "lbfgs"}, "fixed"),
    ],
)
def test_normalize_recipe_rejects_unsafe_or_invalid_parameters(
    task_type, family, parameters, hint
):
    with pytest.raises(ValueError, match=hint):
        recipes.normalize_recipe({"family": family, "parameters": parameters}, task_type)


@pytest.mark.parametrize(
    "recipe",
    [
        {"family": "ExtraTrees", "parameters": {"surprise": 1}},
        {"family": "ExtraTrees", "preprocessing": {"use_magic": True}},
        {"family": "CatBoost", "preprocessing": {"add_missing_indicators": True}},
        {"family": "Linear", "parameters": None},
        {"family": "Linear", "parameters": {"n_jobs": 4}},
    ],
)
def test_normalize_recipe_rejects_unknown_keys_and_incompatible_preprocessing(recipe):
    with pytest.raises(ValueError):
        recipes.normalize_recipe(recipe, "classification")


@pytest.mark.parametrize("family", ["CatBoost", "ExtraTrees", "Linear"])
@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_default_and_explicit_default_recipes_share_fingerprint(family, task_type):
    default = recipes.default_recipe(family, task_type)

    assert recipes.recipe_fingerprint(default, task_type) == recipes.recipe_fingerprint(
        recipes.normalize_recipe(default, task_type), task_type
    )


def test_default_linear_classifier_preserves_the_measured_regularization_baseline():
    recipe = recipes.default_recipe("Linear", "classification")
    module = _run_code(recipes.recipe_code(recipe, "classification"), "recipe_linear_baseline")
    try:
        model = module.build_model(
            {"task_type": "classification", "feature_columns": ["number"], "n_jobs": 2, "seed": 17}
        )
    finally:
        cloudpickle.unregister_pickle_by_value(module)
        sys.modules.pop("recipe_linear_baseline", None)

    params = model.get_params(deep=True)
    assert params["model__C"] == 0.5
    assert params["model__max_iter"] == 300
    assert params["model__solver"] == "saga"
    assert params["model__class_weight"] is None


def test_fingerprint_changes_for_model_weights_and_missing_indicators():
    base = recipes.default_recipe("ExtraTrees", "classification")
    weighted = {**base, "parameters": {**base["parameters"], "class_weight": "balanced"}}
    indicators = {**base, "preprocessing": {"add_missing_indicators": True}}

    fingerprints = {
        recipes.recipe_fingerprint(item, "classification")
        for item in (base, weighted, indicators)
    }
    assert len(fingerprints) == 3


def test_dynamic_catboost_defaults_are_not_materialized_or_overridden():
    recipe = recipes.default_recipe("CatBoost", "classification")

    assert not {"learning_rate", "depth", "l2_leaf_reg", "border_count"} & recipe["parameters"].keys()
    assert recipes.recipe_code(recipe, "classification").count('"learning_rate"') == 0


@pytest.mark.parametrize("family", ["ExtraTrees", "Linear"])
def test_recipe_code_adds_optional_numeric_missing_indicators(family):
    recipe = recipes.default_recipe(family, "classification")
    recipe["preprocessing"]["add_missing_indicators"] = True
    source = recipes.recipe_code(recipe, "classification", n_jobs=2)

    assert "from automl_agent" not in source
    assert 'model.set_params(preprocess__numeric__impute__add_indicator=True)' in source
    assert f"DEFAULT_N_JOBS = 2" in source
    ast.parse(source)
    validate_candidate_code(source)


def test_missing_indicators_fail_clearly_when_context_has_no_numeric_branch():
    recipe = recipes.default_recipe("ExtraTrees", "classification")
    recipe["preprocessing"]["add_missing_indicators"] = True
    module = _run_code(recipes.recipe_code(recipe, "classification"), "recipe_indicators_no_numeric")
    try:
        with pytest.raises(ValueError, match="requires at least one numeric feature"):
            module.build_model(
                {
                    "task_type": "classification",
                    "feature_columns": ["kind"],
                    "categorical_columns": ["kind"],
                    "n_jobs": 1,
                }
            )
    finally:
        cloudpickle.unregister_pickle_by_value(module)
        sys.modules.pop("recipe_indicators_no_numeric", None)


@pytest.mark.parametrize("family", ["CatBoost", "ExtraTrees", "Linear"])
@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_recipe_code_for_each_task_is_standalone_and_passes_candidate_lint(family, task_type):
    source = recipes.recipe_code(recipes.default_recipe(family, task_type), task_type)

    assert "from automl_agent" not in source
    validate_candidate_code(source)


def _run_code(code, name):
    module = types.ModuleType(name)
    exec(compile(code, f"<{name}>", "exec"), module.__dict__)
    sys.modules[name] = module
    cloudpickle.register_pickle_by_value(module)
    return module


@pytest.mark.parametrize("family", ["ExtraTrees", "Linear"])
@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_generated_sklearn_recipe_fits_mixed_data_and_round_trips(family, task_type, tmp_path):
    recipe = recipes.default_recipe(family, task_type)
    if task_type == "classification":
        recipe["parameters"]["class_weight"] = "balanced"
    recipe["preprocessing"]["add_missing_indicators"] = True
    code = recipes.recipe_code(recipe, task_type, n_jobs=2)
    module_name = f"recipe_{family}_{task_type}"
    module = _run_code(code, module_name)
    try:
        X = pd.DataFrame(
            {
                "number": np.arange(48, dtype=float),
                "category": ["alpha" if i % 2 else "beta" for i in range(48)],
            }
        )
        X.loc[3, "number"] = np.nan
        X.loc[4, "number"] = np.inf
        X.loc[5, "number"] = -np.inf
        X.loc[6, "category"] = None
        y = (
            np.asarray(["label_a", "label_b"] * 24)
            if task_type == "classification"
            else np.arange(48, dtype=float)
        )
        context = {
            "task_type": task_type,
            "feature_columns": ["number", "category"],
            "categorical_columns": ["category"],
            "classes": ["label_a", "label_b"] if task_type == "classification" else [],
            "seed": 29,
            "n_jobs": 2,
        }
        model = module.build_model(context)
        if task_type == "classification":
            assert model.get_params(deep=True)["model__class_weight"] == "balanced"
        assert model.get_params(deep=True)[
            "preprocess__numeric__impute__add_indicator"
        ] is True
        model.fit(X, y)
        validation = pd.DataFrame(
            {"number": [np.nan, np.inf, 2.5], "category": [None, "new", "alpha"]}
        )
        prediction = model.predict(validation)
        assert len(prediction) == 3
        if task_type == "classification":
            probabilities = model.predict_proba(validation)
            assert probabilities.shape == (3, 2)
            assert np.isfinite(probabilities).all()
        artifact = tmp_path / f"{family}-{task_type}.joblib"
        joblib.dump(model, artifact)
        restored = joblib.load(artifact)
        assert np.array_equal(restored.predict(validation), prediction)
    finally:
        cloudpickle.unregister_pickle_by_value(module)
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_generated_catboost_recipe_fits_native_categories_and_round_trips(task_type, tmp_path):
    pytest.importorskip("catboost")
    recipe = recipes.default_recipe("CatBoost", task_type)
    recipe["parameters"].update(
        {"depth": 4, "learning_rate": 0.1, "l2_leaf_reg": 2.0, "border_count": 32}
    )
    if task_type == "classification":
        recipe["parameters"]["auto_class_weights"] = "Balanced"
    module_name = f"recipe_CatBoost_{task_type}"
    module = _run_code(recipes.recipe_code(recipe, task_type, n_jobs=2), module_name)
    try:
        X = pd.DataFrame(
            {
                "number": np.arange(36, dtype=float),
                "category": ["amber" if i % 2 else "blue" for i in range(36)],
            }
        )
        X.loc[3, "number"] = np.nan
        X.loc[4, "number"] = np.inf
        X.loc[5, "number"] = -np.inf
        X.loc[6, "category"] = None
        y = (
            np.asarray(["label_a", "label_b"] * 18)
            if task_type == "classification"
            else np.arange(36, dtype=float)
        )
        model = module.build_model(
            {
                "task_type": task_type,
                "feature_columns": ["number", "category"],
                "categorical_columns": ["category"],
                "classes": ["label_a", "label_b"] if task_type == "classification" else [],
                "seed": 19,
                "n_jobs": 2,
            }
        )
        assert model.get_params(deep=True)["model__iterations"] == 150
        assert model.get_params(deep=True)["model__depth"] == 4
        assert model.get_params(deep=True)["model__learning_rate"] == 0.1
        assert model.get_params(deep=True)["model__l2_leaf_reg"] == 2.0
        assert model.get_params(deep=True)["model__border_count"] == 32
        if task_type == "classification":
            assert model.get_params(deep=True)["model__auto_class_weights"] == "Balanced"
        model.fit(X, y)
        validation = pd.DataFrame({"number": [np.nan, np.inf], "category": [None, "new"]})
        prediction = model.predict(validation)
        assert len(prediction) == 2
        artifact = tmp_path / f"CatBoost-{task_type}.joblib"
        joblib.dump(model, artifact)
        restored = joblib.load(artifact)
        assert np.array_equal(restored.predict(validation), prediction)
    finally:
        cloudpickle.unregister_pickle_by_value(module)
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_recipe_source_runs_through_worker_and_saves_reloadable_model(task_type, tmp_path):
    from automl_agent.worker import _run_manifest

    recipe = recipes.default_recipe("Linear", task_type)
    recipe["preprocessing"]["add_missing_indicators"] = True
    code_path = tmp_path / "candidate.py"
    code_path.write_text(recipes.recipe_code(recipe, task_type, n_jobs=1), encoding="utf-8")
    X = pd.DataFrame(
        {
            "amount": np.arange(24, dtype=float),
            "kind": ["a" if i % 2 else "b" for i in range(24)],
        }
    )
    X.loc[2, "amount"] = np.nan
    X.loc[3, "amount"] = np.inf
    X.loc[4, "amount"] = -np.inf
    X.loc[5, "kind"] = None
    y = pd.DataFrame(
        {
            "target": ["amber", "blue"] * 12
            if task_type == "classification"
            else np.arange(24, dtype=float)
        }
    )
    validation = pd.DataFrame({"amount": [np.nan, np.inf], "kind": [None, "unseen"]})
    test = pd.DataFrame({"amount": [2.5, 4.5], "kind": ["a", "new"]})
    inputs = {}
    for name, frame in {
        "train_features": X,
        "train_target": y,
        "validation_features": validation,
        "test_features": test,
    }.items():
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        inputs[name] = str(path)
    artifact_dir = tmp_path / "artifacts"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "mode": "evaluate",
                "inputs": inputs,
                "context": {
                    "target": "target",
                    "task_type": task_type,
                    "metric": "macro_f1" if task_type == "classification" else "rmse",
                    "feature_columns": ["amount", "kind"],
                    "categorical_columns": ["kind"],
                    "classes": ["amber", "blue"] if task_type == "classification" else [],
                    "seed": 11,
                    "n_jobs": 1,
                },
                "code_path": str(code_path),
                "artifact_dir": str(artifact_dir),
                "save_model": True,
            }
        ),
        encoding="utf-8",
    )

    result = _run_manifest(manifest_path)

    assert result["status"] == "ok"
    assert len(pd.read_csv(artifact_dir / "valid_predictions.csv")) == 2
    restored = joblib.load(artifact_dir / "model.joblib")
    assert len(restored.predict(test)) == 2
    if task_type == "classification":
        assert np.load(artifact_dir / "valid_probabilities.npy").shape == (2, 2)


def test_catalog_is_json_serializable_and_describes_task_specific_options():
    catalog = recipes.recipe_catalog("classification")

    assert set(catalog["families"]) == {"CatBoost", "ExtraTrees", "Linear"}
    assert catalog["families"]["CatBoost"]["parameters"]["auto_class_weights"]["options"] == [
        None,
        "Balanced",
        "SqrtBalanced",
    ]
    assert json.loads(json.dumps(catalog)) == catalog
    regression = recipes.recipe_catalog("regression")
    assert "class_weight" not in regression["families"]["ExtraTrees"]["parameters"]
