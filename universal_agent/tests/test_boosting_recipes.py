import importlib.metadata
import json
import subprocess
import sys
import types
from pathlib import Path

import cloudpickle
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from automl_agent import recipes
from automl_agent.tools import validate_candidate_code


def _packages(**versions):
    return versions


def _initial_by_family(task_type, packages):
    return {
        recipe["family"]: recipe
        for recipe in recipes.initial_research_recipes(task_type, packages)
    }


def test_importing_recipe_module_does_not_load_optional_booster_packages():
    script = (
        "import sys; from automl_agent import recipes; "
        "assert 'lightgbm' not in sys.modules; assert 'xgboost' not in sys.modules"
    )
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, capture_output=True, text=True, timeout=15
    )

    assert result.returncode == 0, result.stderr


def test_initial_recipes_include_strong_available_cpu_candidates_and_linear_backup():
    available = _initial_by_family(
        "classification",
        _packages(catboost="1.2.8", lightgbm="4.6.0", xgboost="2.1.4"),
    )

    assert set(available) == {"CatBoost", "LightGBM", "ExtraTrees", "XGBoost", "Linear"}
    assert [recipe["family"] for recipe in recipes.initial_research_recipes(
        "classification", _packages(catboost="1.2.8", lightgbm="4.6.0", xgboost="2.1.4")
    )] == ["LightGBM", "Linear", "ExtraTrees", "XGBoost", "CatBoost"]
    catboost = available["CatBoost"]
    assert catboost["parameters"]["iterations"] == 3000
    assert catboost["parameters"]["learning_rate"] == 0.05
    assert catboost["parameters"]["depth"] == 6
    assert catboost["fit_options"]["early_stopping_rounds"] == 100
    lightgbm = available["LightGBM"]
    assert lightgbm["parameters"]["n_estimators"] == 5000
    assert lightgbm["parameters"]["learning_rate"] == 0.05
    assert lightgbm["parameters"]["num_leaves"] == 31
    assert lightgbm["fit_options"]["early_stopping_rounds"] == 100
    extratrees = available["ExtraTrees"]
    assert extratrees["parameters"]["n_estimators"] == 180
    assert extratrees["parameters"]["max_leaf_nodes"] == 128
    xgboost = available["XGBoost"]
    assert xgboost["parameters"]["n_estimators"] == 3000
    assert xgboost["parameters"]["learning_rate"] == 0.05
    assert xgboost["parameters"]["max_depth"] == 6
    assert xgboost["fit_options"]["early_stopping_rounds"] == 100
    assert all("fit_options" in recipe for recipe in available.values())
    assert available["Linear"]["parameters"]["C"] == 0.5

    no_boosters = _initial_by_family(
        "regression",
        _packages(catboost="unavailable", lightgbm="unavailable", xgboost="unavailable"),
    )
    assert set(no_boosters) == {"ExtraTrees", "Linear"}


@pytest.mark.parametrize("family", ["LightGBM", "XGBoost"])
@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_normalization_defaults_and_fingerprints_include_host_fit_options(family, task_type):
    recipe = recipes.default_recipe(family, task_type)
    explicit_defaults = {
        **recipe,
        "fit_options": {
            "early_stopping_rounds": 0,
            "inner_validation_fraction": 0.15,
            "sample_weight": None,
        },
    }
    changed = {
        **recipe,
        "fit_options": {"early_stopping_rounds": 100},
    }

    assert recipe["fit_options"] == {
        "early_stopping_rounds": 0,
        "inner_validation_fraction": 0.15,
        "sample_weight": None,
    }
    assert recipes.recipe_fingerprint(recipe, task_type) == recipes.recipe_fingerprint(
        explicit_defaults, task_type
    )
    assert recipes.recipe_fingerprint(recipe, task_type) != recipes.recipe_fingerprint(
        changed, task_type
    )
    assert json.loads(json.dumps(recipe)) == recipe


@pytest.mark.parametrize("sample_weight", [None, "balanced", "sqrt_balanced"])
def test_fit_options_validate_supported_weight_modes_and_keep_defaults(sample_weight):
    recipe = recipes.normalize_recipe(
        {
            "family": "LightGBM",
            "fit_options": {"sample_weight": sample_weight},
        },
        "classification",
    )

    assert recipe["fit_options"] == {
        "early_stopping_rounds": 0,
        "inner_validation_fraction": 0.15,
        "sample_weight": sample_weight,
    }


@pytest.mark.parametrize(
    ("family", "task_type", "parameters", "fit_options", "message"),
    [
        (
            "CatBoost",
            "classification",
            {"auto_class_weights": "Balanced"},
            {"sample_weight": "balanced"},
            "double weighting",
        ),
        (
            "ExtraTrees",
            "classification",
            {"class_weight": "balanced"},
            {"sample_weight": "sqrt_balanced"},
            "double weighting",
        ),
        (
            "XGBoost",
            "regression",
            {},
            {"sample_weight": "balanced"},
            "classification",
        ),
        (
            "ExtraTrees",
            "classification",
            {},
            {"early_stopping_rounds": 10},
            "early stopping is supported only",
        ),
        (
            "LightGBM",
            "classification",
            {},
            {"early_stopping_rounds": 301},
            "early_stopping_rounds",
        ),
        (
            "XGBoost",
            "classification",
            {},
            {"inner_validation_fraction": 0.31},
            "inner_validation_fraction",
        ),
    ],
)
def test_normalize_recipe_rejects_conflicting_or_out_of_range_fit_options(
    family, task_type, parameters, fit_options, message
):
    with pytest.raises(ValueError, match=message):
        recipes.normalize_recipe(
            {"family": family, "parameters": parameters, "fit_options": fit_options},
            task_type,
        )


@pytest.mark.parametrize(
    ("family", "task_type", "parameters"),
    [
        ("LightGBM", "classification", {"n_estimators": 5001}),
        ("LightGBM", "classification", {"num_leaves": 1}),
        ("LightGBM", "classification", {"n_jobs": 2}),
        ("XGBoost", "regression", {"max_depth": 1}),
        ("XGBoost", "regression", {"tree_method": "gpu_hist"}),
        ("XGBoost", "classification", {"scale_pos_weight": 2}),
        ("CatBoost", "classification", {"iterations": 5001}),
    ],
)
def test_native_boosting_parameters_are_bounded_and_host_controls_are_rejected(
    family, task_type, parameters
):
    with pytest.raises(ValueError):
        recipes.normalize_recipe(
            {"family": family, "parameters": parameters}, task_type
        )


@pytest.mark.parametrize("family", ["LightGBM", "XGBoost"])
@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_generated_boosting_source_is_standalone_and_candidate_linted(family, task_type):
    recipe = recipes.default_recipe(family, task_type)
    recipe["parameters"]["n_estimators"] = 24
    source = recipes.recipe_code(recipe, task_type, n_jobs=2)

    assert "from automl_agent" not in source
    assert '"tree_method": "hist"' in source if family == "XGBoost" else '"device_type": "cpu"' in source
    assert "fit_options" not in source
    validate_candidate_code(source)


def _load_generated_module(source, name):
    module = types.ModuleType(name)
    exec(compile(source, f"<{name}>", "exec"), module.__dict__)
    sys.modules[name] = module
    cloudpickle.register_pickle_by_value(module)
    return module


@pytest.mark.parametrize("family", ["LightGBM", "XGBoost"])
@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_generated_boosting_recipes_fit_mixed_data_and_serialize(
    family, task_type, tmp_path
):
    try:
        importlib.metadata.version(family.lower())
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(f"{family} is not installed")
    recipe = recipes.default_recipe(family, task_type)
    recipe["parameters"]["n_estimators"] = 24
    recipe["parameters"]["learning_rate"] = 0.1
    recipe["preprocessing"]["add_missing_indicators"] = True
    source = recipes.recipe_code(recipe, task_type, n_jobs=1)
    module_name = f"boosting_{family}_{task_type}"
    module = _load_generated_module(source, module_name)
    try:
        rows = 96
        X = pd.DataFrame(
            {
                "number": np.arange(rows, dtype=float),
                "category": ["amber" if i % 2 else "blue" for i in range(rows)],
            }
        )
        X.loc[3, "number"] = np.nan
        X.loc[4, "number"] = np.inf
        X.loc[5, "number"] = -np.inf
        X.loc[6, "category"] = None
        if task_type == "classification":
            raw_y = np.asarray(["label_a", "label_b"] * (rows // 2))
            y = LabelEncoder().fit_transform(raw_y) if family == "XGBoost" else raw_y
        else:
            y = np.arange(rows, dtype=float)
        context = {
            "task_type": task_type,
            "feature_columns": ["number", "category"],
            "categorical_columns": ["category"],
            "classes": ["label_a", "label_b"] if task_type == "classification" else [],
            "seed": 19,
            "n_jobs": 1,
        }
        model = module.build_model(context)
        params = model.get_params(deep=True)
        assert params["model__n_jobs"] == 1
        assert params["model__random_state"] == 19
        if family == "LightGBM":
            assert params["model__device_type"] == "cpu"
        else:
            assert params["model__tree_method"] == "hist"
            assert params["model__device"] == "cpu"
        model.fit(X, y)
        validation = pd.DataFrame(
            {"number": [np.nan, np.inf, 3.5], "category": [None, "new", "amber"]}
        )
        predictions = model.predict(validation)
        assert len(predictions) == len(validation)
        if task_type == "classification":
            probability = model.predict_proba(validation)
            assert probability.shape == (3, 2)
            assert np.isfinite(probability).all()
            assert np.allclose(probability.sum(axis=1), 1.0, atol=1e-6)
        artifact = tmp_path / f"{family}-{task_type}.joblib"
        joblib.dump(model, artifact)
        restored = joblib.load(artifact)
        assert np.array_equal(restored.predict(validation), predictions)
    finally:
        cloudpickle.unregister_pickle_by_value(module)
        sys.modules.pop(module_name, None)


def test_recipe_catalog_explains_host_fit_options_and_excludes_optional_unavailable_models():
    catalog = recipes.recipe_catalog("classification")

    assert "fit_options" in catalog
    assert catalog["fit_options"]["source"].startswith("automl_agent.training.fit_model")
    assert set(catalog["fit_options"]["options"]) == {
        "early_stopping_rounds",
        "inner_validation_fraction",
        "sample_weight",
    }
    available = _initial_by_family(
        "classification",
        _packages(catboost="unavailable", lightgbm=None, xgboost="not installed"),
    )
    assert "LightGBM" not in available
    assert "XGBoost" not in available
    assert "CatBoost" not in available
