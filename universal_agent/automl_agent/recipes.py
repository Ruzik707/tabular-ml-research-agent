"""Validated, declarative model recipes with standalone candidate generators."""

from __future__ import annotations

import hashlib
import json
import math
import textwrap
from pathlib import Path
from collections.abc import Mapping

from . import families
from .features import FEATURE_OPERATIONS


_FAMILIES = ("CatBoost", "LightGBM", "ExtraTrees", "XGBoost", "Linear")
_BOOSTING_FAMILIES = {"CatBoost", "LightGBM", "XGBoost"}
_TASK_TYPES = ("classification", "regression")
_HOST_CONTROLLED = {
    "n_jobs", "thread_count", "random_state", "random_seed", "seed", "nthread",
    "num_threads", "device", "device_type", "tree_method", "task_type",
}
_REJECTED_GUIDANCE = {
    "class_weights": (
        "class_weights is not a valid recipe parameter; use CatBoost auto_class_weights "
        "or sklearn class_weight."
    ),
    "sample_weight": (
        "sample_weight is per-row fit data, not a recipe parameter; use CatBoost "
        "auto_class_weights='Balanced' or 'SqrtBalanced', or sklearn class_weight="
        "'balanced' (ExtraTrees also accepts 'balanced_subsample')."
    ),
    "early_stopping_rounds": (
        "early_stopping_rounds belongs in fit_options and requires the host training "
        "helper to provide a validated inner eval_set."
    ),
}


def _task_type(task_type):
    if task_type not in _TASK_TYPES:
        raise ValueError("task_type must be 'classification' or 'regression'.")
    return task_type


def _family(family):
    if family not in _FAMILIES:
        raise ValueError(f"family must be one of {', '.join(_FAMILIES)}.")
    return family


def _default_parameters(family, task_type):
    if family == "CatBoost":
        parameters = {"iterations": 150, "auto_class_weights": None}
        if task_type == "classification":
            return parameters
        parameters.pop("auto_class_weights")
        return parameters
    if family == "LightGBM":
        return {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "reg_lambda": 0.0,
        }
    if family == "XGBoost":
        return {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1.0,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
        }
    if family == "ExtraTrees":
        parameters = {
            "n_estimators": 180,
            "max_leaf_nodes": 128,
            "min_samples_leaf": 3,
            "max_features": 0.8,
            "max_depth": None,
        }
        if task_type == "classification":
            parameters["class_weight"] = None
        return parameters
    if task_type == "classification":
        return {"C": 0.5, "max_iter": 300, "solver": "saga", "class_weight": None}
    return {"alpha": 1.0}


def default_recipe(family, task_type):
    """Return a complete normalized recipe for one supported model family."""
    family = _family(family)
    task_type = _task_type(task_type)
    return normalize_recipe({"family": family}, task_type)


def _integer(name, value, minimum, maximum):
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}].")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}].")
    return value


def _number(name, value, minimum, maximum, *, exclusive_minimum=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        relation = "greater than" if exclusive_minimum else "at least"
        raise ValueError(f"{name} must be a finite number {relation} {minimum} and at most {maximum}.")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number.")
    minimum_valid = normalized > minimum if exclusive_minimum else normalized >= minimum
    if not minimum_valid or normalized > maximum:
        left = "(" if exclusive_minimum else "["
        raise ValueError(f"{name} must be in {left}{minimum}, {maximum}].")
    return normalized


def _strict_mapping(value, field_name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object.")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} keys must be strings.")
    return value


def _normalize_fit_options(family, task_type, options, parameters):
    options = _strict_mapping(options, "fit_options")
    defaults = {
        "early_stopping_rounds": 0,
        "inner_validation_fraction": 0.15,
        "sample_weight": None,
    }
    unknown = set(options) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown fit_options field {sorted(unknown)[0]!r}.")

    early_stopping = options.get("early_stopping_rounds", defaults["early_stopping_rounds"])
    early_stopping = _integer("early_stopping_rounds", early_stopping, 0, 300)
    if early_stopping and family not in _BOOSTING_FAMILIES:
        raise ValueError(
            "early stopping is supported only for CatBoost, LightGBM, and XGBoost recipes."
        )

    fraction = _number(
        "inner_validation_fraction",
        options.get("inner_validation_fraction", defaults["inner_validation_fraction"]),
        0.1,
        0.3,
    )
    if not early_stopping and fraction != defaults["inner_validation_fraction"]:
        raise ValueError(
            "inner_validation_fraction is only used when early_stopping_rounds is greater than zero."
        )

    sample_weight = options.get("sample_weight", defaults["sample_weight"])
    if sample_weight not in (None, "balanced", "sqrt_balanced"):
        raise ValueError("sample_weight must be None, 'balanced', or 'sqrt_balanced'.")
    if task_type != "classification" and sample_weight is not None:
        raise ValueError("sample_weight modes 'balanced' and 'sqrt_balanced' are classification-only.")
    configured_weight = None
    if family == "CatBoost":
        configured_weight = parameters.get("auto_class_weights")
    elif family in {"ExtraTrees", "Linear"}:
        configured_weight = parameters.get("class_weight")
    if sample_weight is not None and configured_weight is not None:
        raise ValueError(
            "double weighting is not allowed: do not combine fit_options.sample_weight with constructor class weights; "
            "choose one weighting method."
        )

    return {
        "early_stopping_rounds": early_stopping,
        "inner_validation_fraction": fraction,
        "sample_weight": sample_weight,
    }


def _normalize_parameter(family, task_type, name, value):
    if name in _HOST_CONTROLLED:
        raise ValueError(f"{name} is host-controlled and cannot be changed in a recipe.")
    if name in _REJECTED_GUIDANCE:
        raise ValueError(_REJECTED_GUIDANCE[name])

    if family == "CatBoost":
        if name == "iterations":
            return _integer(name, value, 10, 5000)
        if name == "depth":
            return _integer(name, value, 3, 10)
        if name == "learning_rate":
            return _number(name, value, 0, 1, exclusive_minimum=True)
        if name == "l2_leaf_reg":
            return _number(name, value, 0, 100)
        if name == "border_count":
            return _integer(name, value, 16, 255)
        if name == "auto_class_weights":
            if task_type != "classification":
                raise ValueError("auto_class_weights is available only for classification.")
            if value == "None":
                return None
            if value not in (None, "Balanced", "SqrtBalanced"):
                raise ValueError("auto_class_weights must be None, 'Balanced', or 'SqrtBalanced'.")
            return value
    elif family == "LightGBM":
        if name == "n_estimators":
            return _integer(name, value, 10, 5000)
        if name == "learning_rate":
            return _number(name, value, 0, 1, exclusive_minimum=True)
        if name == "num_leaves":
            return _integer(name, value, 2, 255)
        if name == "max_depth":
            if type(value) is int and value == -1:
                return -1
            return _integer(name, value, 2, 32)
        if name == "min_child_samples":
            return _integer(name, value, 5, 200)
        if name == "reg_lambda":
            return _number(name, value, 0, 100)
    elif family == "XGBoost":
        if name == "n_estimators":
            return _integer(name, value, 10, 5000)
        if name == "learning_rate":
            return _number(name, value, 0, 1, exclusive_minimum=True)
        if name == "max_depth":
            return _integer(name, value, 2, 12)
        if name == "min_child_weight":
            return _number(name, value, 0, 100)
        if name in {"subsample", "colsample_bytree"}:
            return _number(name, value, 0, 1, exclusive_minimum=True)
        if name == "reg_lambda":
            return _number(name, value, 0, 100)
    elif family == "ExtraTrees":
        if name == "n_estimators":
            return _integer(name, value, 10, 1000)
        if name == "max_leaf_nodes":
            return _integer(name, value, 2, 2048)
        if name == "min_samples_leaf":
            return _integer(name, value, 1, 100)
        if name == "max_depth":
            if value is None:
                return None
            return _integer(name, value, 2, 64)
        if name == "max_features":
            if value is None or value in ("sqrt", "log2"):
                return value
            return _number(name, value, 0, 1, exclusive_minimum=True)
        if name == "class_weight":
            if task_type != "classification":
                raise ValueError("class_weight is available only for classification.")
            if value not in (None, "balanced", "balanced_subsample"):
                raise ValueError("class_weight must be None, 'balanced', or 'balanced_subsample'.")
            return value
    else:
        if task_type == "classification":
            if name == "C":
                return _number(name, value, 0, 100, exclusive_minimum=True)
            if name == "max_iter":
                return _integer(name, value, 100, 3000)
            if name == "solver":
                if value != "saga":
                    raise ValueError("Linear classifier solver is fixed to 'saga'.")
                return "saga"
            if name == "class_weight":
                if value not in (None, "balanced"):
                    raise ValueError("Linear class_weight must be None or 'balanced'.")
                return value
        elif name == "alpha":
            return _number(name, value, 0, 10000, exclusive_minimum=True)

    raise ValueError(f"Unknown or unsupported {family} recipe parameter {name!r} for {task_type}.")


def normalize_recipe(recipe, task_type):
    """Validate recipe shape, task-specific options, types, and bounds."""
    task_type = _task_type(task_type)
    recipe = _strict_mapping(recipe, "recipe")
    unknown_top_level = set(recipe) - {"family", "parameters", "preprocessing", "fit_options"}
    if unknown_top_level:
        key = sorted(unknown_top_level)[0]
        if key in _REJECTED_GUIDANCE:
            raise ValueError(_REJECTED_GUIDANCE[key])
        if key in _HOST_CONTROLLED:
            raise ValueError(f"{key} is host-controlled and cannot be changed in a recipe.")
        raise ValueError(
            f"Unknown recipe field {key!r}; allowed fields are family, parameters, preprocessing, fit_options."
        )
    if "family" not in recipe:
        raise ValueError("recipe.family is required.")
    family = _family(recipe["family"])

    parameters = dict(_default_parameters(family, task_type))
    supplied_parameters = _strict_mapping(recipe.get("parameters", {}), "parameters")
    for name, value in supplied_parameters.items():
        parameters[name] = _normalize_parameter(family, task_type, name, value)

    preprocessing_defaults = {"add_missing_indicators": False}
    preprocessing = _strict_mapping(recipe.get("preprocessing", {}), "preprocessing")
    unknown_preprocessing = set(preprocessing) - {*preprocessing_defaults, 'feature_ops'}
    if unknown_preprocessing:
        key = sorted(unknown_preprocessing)[0]
        raise ValueError(
            f"Unknown preprocessing option {key!r}; use add_missing_indicators or feature_ops."
        )
    if "add_missing_indicators" in preprocessing:
        value = preprocessing["add_missing_indicators"]
        if type(value) is not bool:
            raise ValueError("add_missing_indicators must be a boolean.")
        if family == "CatBoost" and value:
            raise ValueError("CatBoost uses native missing-value handling; add_missing_indicators must be false.")
        preprocessing_defaults["add_missing_indicators"] = value
    if 'feature_ops' in preprocessing:
        from .features import validate_feature_ops
        ops = validate_feature_ops(preprocessing['feature_ops'])
        if ops:
            preprocessing_defaults['feature_ops'] = ops

    normalized = {
        "family": family,
        "parameters": parameters,
        "preprocessing": preprocessing_defaults,
    }
    fit_options = _normalize_fit_options(
        family, task_type, recipe.get("fit_options", {}), parameters
    )
    # Preserve the exact shape of legacy default recipes. Boosting recipes and
    # recipes that explicitly carry host options expose the full normalized set.
    if family in {"LightGBM", "XGBoost"} or "fit_options" in recipe:
        normalized["fit_options"] = fit_options
    return normalized


def _boosting_source(family, n_jobs):
    source = families._source(n_jobs, "ExtraTrees")
    import_marker = "from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor"
    if source.count(import_marker) != 1:
        raise RuntimeError("The verified ExtraTrees template import changed unexpectedly.")
    if family == "LightGBM":
        source = source.replace(import_marker, "from lightgbm import LGBMClassifier, LGBMRegressor", 1)
        estimator_block = textwrap.dedent('''\
            common = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 20,
                "reg_lambda": 0.0,
                "n_jobs": _threads(context),
                "random_state": _seed(context),
                "device_type": "cpu",
                "verbosity": -1,
            }
            if context.get("task_type") == "classification":
                estimator = LGBMClassifier(**common)
            else:
                estimator = LGBMRegressor(**common)
        ''')
    elif family == "XGBoost":
        source = source.replace(import_marker, "from xgboost import XGBClassifier, XGBRegressor", 1)
        estimator_block = textwrap.dedent('''\
            common = {
                "n_estimators": 300,
                "learning_rate": 0.1,
                "max_depth": 6,
                "min_child_weight": 1.0,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 1.0,
                "n_jobs": _threads(context),
                "random_state": _seed(context),
                "tree_method": "hist",
                "device": "cpu",
                "verbosity": 0,
            }
            if context.get("task_type") == "classification":
                estimator = XGBClassifier(**common)
            else:
                estimator = XGBRegressor(**common)
        ''')
    else:
        raise ValueError(f"No standalone source template for {family!r}.")

    start_marker = "    common = {"
    end_marker = "    return Pipeline([("
    start = source.find(start_marker)
    end = source.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError("The verified ExtraTrees estimator block changed unexpectedly.")
    return source[:start] + textwrap.indent(estimator_block, "    ") + "\n" + source[end:]


def recipe_code(recipe, task_type, n_jobs=4):
    """Build complete standalone Python source from a validated recipe."""
    normalized = normalize_recipe(recipe, task_type)
    if type(n_jobs) is not int or not 1 <= n_jobs <= 8:
        raise ValueError("n_jobs must be an integer in [1, 8]; worker parallelism is host-controlled.")
    family = normalized["family"]
    source = (
        _boosting_source(family, n_jobs)
        if family in {"LightGBM", "XGBoost"}
        else families._source(n_jobs, family)
    )
    expected = "def build_model(context):"
    if source.count(expected) != 1:
        raise RuntimeError("The verified family template does not contain one build_model function.")
    source = source.replace(expected, "def _build_template_model(context):", 1)
    ops = normalized['preprocessing'].get('feature_ops', [])
    if ops:
        # Embed the verified transformer rather than importing the host package
        # from untrusted candidate code; generated files remain self-contained.
        source += '\n\n' + Path(__file__).with_name('features.py').read_text(encoding='utf-8')
    params = {f"model__{key}": value for key, value in normalized["parameters"].items()}
    if normalized["family"] == "CatBoost":
        # CatBoost treats the constructor default as unset and rejects JSON null
        # for auto_class_weights; a None recipe value means preserve that default.
        params = {key: value for key, value in params.items() if value is not None}
    wrapper = "\n\ndef build_model(context):\n    model = _build_template_model(context)\n"
    if ops:
        wrapper = '\n\ndef build_model(context):\n'
        wrapper += f'    ops = {ops!r}\n'
        wrapper += '    context = dict(context)\n'
        wrapper += '    original_categories = list(context.get("categorical_columns", []))\n'
        wrapper += '    columns, categories = feature_schema(context["feature_columns"], original_categories, ops)\n'
        wrapper += '    context.update(feature_columns=columns, categorical_columns=categories)\n'
        wrapper += '    model = _build_template_model(context)\n'
    wrapper += f"    model.set_params(**{params!r})\n"
    if normalized["preprocessing"]["add_missing_indicators"]:
        wrapper += "    preprocess = model.named_steps['preprocess']\n"
        wrapper += "    if not any(name == 'numeric' for name, _, _ in preprocess.transformers):\n"
        wrapper += "        raise ValueError('add_missing_indicators requires at least one numeric feature column.')\n"
        wrapper += "    model.set_params(preprocess__numeric__impute__add_indicator=True)\n"
    if ops:
        wrapper += '    model.steps.insert(0, ("features", FeatureTransformer(ops, original_categories)))\n'
    wrapper += "    return model\n"
    return source.rstrip() + wrapper


def recipe_fingerprint(recipe, task_type):
    """Return SHA-256 of canonical normalized JSON for recipe deduplication."""
    normalized = normalize_recipe(recipe, task_type)
    normalized.setdefault(
        "fit_options",
        {
            "early_stopping_rounds": 0,
            "inner_validation_fraction": 0.15,
            "sample_weight": None,
        },
    )
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _profile_package_available(packages, package_name):
    if not isinstance(packages, Mapping):
        raise ValueError("available_packages must be a JSON object of package names to versions.")
    aliases = {
        "catboost": {"catboost"},
        "lightgbm": {"lightgbm", "light-gbm"},
        "xgboost": {"xgboost", "xg-boost"},
    }[package_name]
    value = None
    found = False
    for name, version in packages.items():
        if isinstance(name, str) and name.strip().lower() in aliases:
            value = version
            found = True
            break
    if not found or value is None:
        return False
    return str(value).strip().lower() not in {
        "", "unavailable", "none", "not installed", "missing", "false",
    }


def initial_research_recipes(task_type, available_packages):
    """Return bounded strong CPU candidates supported by the profiled packages."""
    task_type = _task_type(task_type)
    packages = _strict_mapping(available_packages, "available_packages")
    candidates = []

    def add(family, updates=None, early_stopping_rounds=0):
        parameters = _default_parameters(family, task_type)
        parameters.update(updates or {})
        candidates.append(
            normalize_recipe(
                {
                    "family": family,
                    "parameters": parameters,
                    "preprocessing": {"add_missing_indicators": False},
                    "fit_options": {
                        "early_stopping_rounds": early_stopping_rounds,
                        "inner_validation_fraction": 0.15,
                        "sample_weight": None,
                    },
                },
                task_type,
            )
        )

    if _profile_package_available(packages, "lightgbm"):
        add("LightGBM", {"n_estimators": 5000, "learning_rate": 0.05, "num_leaves": 31}, 100)
    # A cheap, regularized linear baseline should consume an early trial even
    # when one or more tree packages are installed.
    add("Linear")
    add("ExtraTrees")
    if _profile_package_available(packages, "xgboost"):
        add("XGBoost", {"n_estimators": 3000, "learning_rate": 0.05, "max_depth": 6}, 100)
    if _profile_package_available(packages, "catboost"):
        add("CatBoost", {"iterations": 3000, "learning_rate": 0.05, "depth": 6}, 100)
    return candidates


def recipe_catalog(task_type):
    """Describe supported recipe fields and safe options for a model task."""
    task_type = _task_type(task_type)
    families_catalog = {}
    if task_type == "classification":
        families_catalog["CatBoost"] = {
            "parameters": {
                "iterations": {"type": "integer", "minimum": 10, "maximum": 5000, "default": 150},
                "depth": {"type": "integer", "minimum": 3, "maximum": 10, "default": "native"},
                "learning_rate": {"type": "number", "exclusive_minimum": 0, "maximum": 1, "default": "native automatic"},
                "l2_leaf_reg": {"type": "number", "minimum": 0, "maximum": 100, "default": "native"},
                "border_count": {"type": "integer", "minimum": 16, "maximum": 255, "default": "native"},
                "auto_class_weights": {"type": "enum", "options": [None, "Balanced", "SqrtBalanced"], "default": None},
            },
            "preprocessing": {"add_missing_indicators": {"type": "boolean", "default": False, "fixed": True}},
        }
        families_catalog["ExtraTrees"] = {
            "parameters": {
                "n_estimators": {"type": "integer", "minimum": 10, "maximum": 1000, "default": 180},
                "max_leaf_nodes": {"type": "integer", "minimum": 2, "maximum": 2048, "default": 128},
                "min_samples_leaf": {"type": "integer", "minimum": 1, "maximum": 100, "default": 3},
                "max_features": {"type": "number|string|null", "options": ["sqrt", "log2", None], "exclusive_minimum": 0, "maximum": 1, "default": 0.8},
                "max_depth": {"type": "integer|null", "minimum": 2, "maximum": 64, "default": None},
                "class_weight": {"type": "enum", "options": [None, "balanced", "balanced_subsample"], "default": None},
            },
            "preprocessing": {"add_missing_indicators": {"type": "boolean", "default": False}},
        }
        families_catalog["Linear"] = {
            "parameters": {
                "C": {"type": "number", "exclusive_minimum": 0, "maximum": 100, "default": 0.5},
                "max_iter": {"type": "integer", "minimum": 100, "maximum": 3000, "default": 300},
                "solver": {"type": "enum", "options": ["saga"], "default": "saga", "fixed": True},
                "class_weight": {"type": "enum", "options": [None, "balanced"], "default": None},
            },
            "preprocessing": {"add_missing_indicators": {"type": "boolean", "default": False}},
        }
    else:
        families_catalog["CatBoost"] = {
            "parameters": {
                "iterations": {"type": "integer", "minimum": 10, "maximum": 5000, "default": 150},
                "depth": {"type": "integer", "minimum": 3, "maximum": 10, "default": "native"},
                "learning_rate": {"type": "number", "exclusive_minimum": 0, "maximum": 1, "default": "native automatic"},
                "l2_leaf_reg": {"type": "number", "minimum": 0, "maximum": 100, "default": "native"},
                "border_count": {"type": "integer", "minimum": 16, "maximum": 255, "default": "native"},
            },
            "preprocessing": {"add_missing_indicators": {"type": "boolean", "default": False, "fixed": True}},
        }
        families_catalog["ExtraTrees"] = {
            "parameters": {
                "n_estimators": {"type": "integer", "minimum": 10, "maximum": 1000, "default": 180},
                "max_leaf_nodes": {"type": "integer", "minimum": 2, "maximum": 2048, "default": 128},
                "min_samples_leaf": {"type": "integer", "minimum": 1, "maximum": 100, "default": 3},
                "max_features": {"type": "number|string|null", "options": ["sqrt", "log2", None], "exclusive_minimum": 0, "maximum": 1, "default": 0.8},
                "max_depth": {"type": "integer|null", "minimum": 2, "maximum": 64, "default": None},
            },
            "preprocessing": {"add_missing_indicators": {"type": "boolean", "default": False}},
        }
        families_catalog["Linear"] = {
            "parameters": {
                "alpha": {"type": "number", "exclusive_minimum": 0, "maximum": 10000, "default": 1.0}
            },
            "preprocessing": {"add_missing_indicators": {"type": "boolean", "default": False}},
        }

    boosting_catalog = {
        "LightGBM": {
            "parameters": {
                "n_estimators": {"type": "integer", "minimum": 10, "maximum": 5000, "default": 100},
                "learning_rate": {"type": "number", "exclusive_minimum": 0, "maximum": 1, "default": 0.1},
                "num_leaves": {"type": "integer", "minimum": 2, "maximum": 255, "default": 31},
                "max_depth": {"type": "integer", "minimum": 2, "maximum": 32, "also_allow": [-1], "default": -1},
                "min_child_samples": {"type": "integer", "minimum": 5, "maximum": 200, "default": 20},
                "reg_lambda": {"type": "number", "minimum": 0, "maximum": 100, "default": 0.0},
            },
            "preprocessing": {"add_missing_indicators": {"type": "boolean", "default": False}},
        },
        "XGBoost": {
            "parameters": {
                "n_estimators": {"type": "integer", "minimum": 10, "maximum": 5000, "default": 300},
                "learning_rate": {"type": "number", "exclusive_minimum": 0, "maximum": 1, "default": 0.1},
                "max_depth": {"type": "integer", "minimum": 2, "maximum": 12, "default": 6},
                "min_child_weight": {"type": "number", "minimum": 0, "maximum": 100, "default": 1.0},
                "subsample": {"type": "number", "exclusive_minimum": 0, "maximum": 1, "default": 1.0},
                "colsample_bytree": {"type": "number", "exclusive_minimum": 0, "maximum": 1, "default": 1.0},
                "reg_lambda": {"type": "number", "minimum": 0, "maximum": 100, "default": 1.0},
            },
            "preprocessing": {"add_missing_indicators": {"type": "boolean", "default": False}},
        },
    }

    return {
        "task_type": task_type,
        "feature_ops": {
            'location': 'preprocessing.feature_ops',
            'semantics': 'Ordered array, at most 16 operations/128 new columns. Learned statistics fit on each training fold only. cols/pairs/by must name existing feature columns; target and IDs are not available.',
            'operations': FEATURE_OPERATIONS,
        },
        "families": families_catalog,
        "boosting_families": boosting_catalog,
        "fit_options": {
            "source": "automl_agent.training.fit_model",
            "execution": "Host-controlled; fit options are not emitted into standalone candidate code.",
            "options": {
                "early_stopping_rounds": {
                    "type": "integer", "minimum": 0, "maximum": 300,
                    "default": 0, "supported_families": ["CatBoost", "LightGBM", "XGBoost"],
                },
                "inner_validation_fraction": {
                    "type": "number", "minimum": 0.1, "maximum": 0.3, "default": 0.15,
                },
                "sample_weight": {
                    "type": "enum", "options": [None, "balanced", "sqrt_balanced"],
                    "default": None, "classification_only": True,
                    "mutually_exclusive_with_constructor_class_weight": True,
                },
            },
        },
        "host_controlled": sorted(_HOST_CONTROLLED),
        "unsupported": {
            "class_weights": "Use CatBoost auto_class_weights or sklearn class_weight.",
            "sample_weight": "Use fit_options.sample_weight; do not combine it with constructor class weights.",
            "early_stopping_rounds": "Use fit_options.early_stopping_rounds; the host creates the inner eval set.",
        },
    }
