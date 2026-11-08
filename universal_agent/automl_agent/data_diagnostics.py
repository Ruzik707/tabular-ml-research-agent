"""Bounded, host-side diagnostics for train/test feature distributions."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


_MAX_ROWS_PER_DOMAIN = 10_000
_MIN_DOMAIN_ROWS_FOR_CV = 100
_MAX_THREADS = 4
_MAX_FEATURES_IN_PROFILE = 100
_MAX_FEATURES_IN_CV = 128
_MAX_CATEGORY_LEVELS = 64
_MAX_TOP_CATEGORIES = 10
_TEMPORAL_WORDS = re.compile(
    r"(?:^|[^a-z])(?:year|time|date|period|quarter|month|week|day|timestamp|season|cycle)(?:$|[^a-z])",
    re.IGNORECASE,
)

_KNOWN_TREE_ESTIMATORS = {
    ("sklearn.ensemble._forest", "ExtraTreesClassifier"),
    ("sklearn.ensemble._forest", "ExtraTreesRegressor"),
    ("sklearn.ensemble._forest", "RandomForestClassifier"),
    ("sklearn.ensemble._forest", "RandomForestRegressor"),
    ("sklearn.ensemble._gb", "GradientBoostingClassifier"),
    ("sklearn.ensemble._gb", "GradientBoostingRegressor"),
    ("sklearn.tree._classes", "DecisionTreeClassifier"),
    ("sklearn.tree._classes", "DecisionTreeRegressor"),
    ("catboost.core", "CatBoostClassifier"),
    ("catboost.core", "CatBoostRegressor"),
    ("lightgbm.sklearn", "LGBMClassifier"),
    ("lightgbm.sklearn", "LGBMRegressor"),
    ("xgboost.sklearn", "XGBClassifier"),
    ("xgboost.sklearn", "XGBRegressor"),
}
_KNOWN_LINEAR_ESTIMATORS = {
    ("sklearn.linear_model._logistic", "LogisticRegression"),
    ("sklearn.linear_model._base", "LinearRegression"),
    ("sklearn.linear_model._ridge", "Ridge"),
}


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:120]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)[:120]


def _feature_columns(task: Any) -> list[str]:
    target = _value(task, "target")
    ids = set(_value(task, "id_columns", []) or [])
    columns = list(_value(task, "feature_columns", []) or [])
    return [column for column in columns if column != target and column not in ids]


def _read_feature_frames(task: Any) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    columns = _feature_columns(task)
    if not columns:
        train_path = _value(task, "train_path")
        test_path = _value(task, "test_path")
        train_headers = pd.read_csv(train_path, nrows=0).columns.tolist()
        test_headers = pd.read_csv(test_path, nrows=0).columns.tolist()
        excluded = set(_value(task, "id_columns", []) or []) | {_value(task, "target")}
        columns = [name for name in train_headers if name in test_headers and name not in excluded]
    if not columns:
        return pd.DataFrame(), pd.DataFrame(), []
    train = pd.read_csv(_value(task, "train_path"), usecols=columns, low_memory=False)
    test = pd.read_csv(_value(task, "test_path"), usecols=columns, low_memory=False)
    # Preserve TaskSpec feature order, even if the CSV files present columns differently.
    return train.loc[:, columns], test.loc[:, columns], columns


def _numeric_pair(train: pd.Series, test: pd.Series, force_category: bool) -> bool:
    if force_category or pd.api.types.is_bool_dtype(train.dtype) or pd.api.types.is_bool_dtype(test.dtype):
        return False
    if pd.api.types.is_numeric_dtype(train.dtype) and pd.api.types.is_numeric_dtype(test.dtype):
        return True
    train_num = pd.to_numeric(train, errors="coerce")
    test_num = pd.to_numeric(test, errors="coerce")
    return bool(
        (train.notna().sum() == train_num.notna().sum())
        and (test.notna().sum() == test_num.notna().sum())
        and (train.notna().any() or test.notna().any())
    )


def _clean_numeric(values: pd.Series) -> pd.Series:
    converted = pd.to_numeric(values, errors="coerce").astype(float)
    return converted.where(np.isfinite(converted), np.nan)


def _numeric_stats(train: pd.Series, test: pd.Series) -> dict[str, Any]:
    train_values = _clean_numeric(train).dropna()
    test_values = _clean_numeric(test).dropna()
    result: dict[str, Any] = {
        "train_missing_rate": float(1.0 - len(train_values) / max(1, len(train))),
        "test_missing_rate": float(1.0 - len(test_values) / max(1, len(test))),
        "train_unique_count": int(train_values.nunique()),
        "test_unique_count": int(test_values.nunique()),
    }
    for prefix, values in (("train", train_values), ("test", test_values)):
        if len(values):
            result.update({
                f"{prefix}_mean": float(values.mean()),
                f"{prefix}_std": float(values.std(ddof=0)),
                f"{prefix}_min": float(values.min()),
                f"{prefix}_max": float(values.max()),
            })
        else:
            result.update({f"{prefix}_{name}": None for name in ("mean", "std", "min", "max")})
    if result["train_mean"] is not None and result["test_mean"] is not None:
        result["mean_difference"] = result["test_mean"] - result["train_mean"]
    return result


def _category_key(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return ("missing", None)
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return (type(value).__name__, value)
    return (type(value).__name__, str(value))


def _top_categories(values: pd.Series, counts: Counter, only_keys: set[Any] | None = None) -> list[dict[str, Any]]:
    kept = [(key, count) for key, count in counts.items() if only_keys is None or key in only_keys]
    kept.sort(key=lambda item: (-item[1], str(item[0])))
    top = []
    labels = {}
    for raw_value in values.tolist():
        key = _category_key(raw_value)
        if key not in labels:
            labels[key] = None if key[0] == "missing" else _json_value(raw_value)
    for key, count in kept[:_MAX_TOP_CATEGORIES]:
        value = labels.get(key)
        top.append({"value": value, "count": int(count)})
    return top


def _categorical_stats(train: pd.Series, test: pd.Series) -> dict[str, Any]:
    train_counts = Counter(_category_key(value) for value in train.tolist())
    test_counts = Counter(_category_key(value) for value in test.tolist())
    train_only = set(train_counts) - set(test_counts)
    test_only = set(test_counts) - set(train_counts)
    return {
        "train_missing_rate": float(train.isna().mean()),
        "test_missing_rate": float(test.isna().mean()),
        "train_unique_count": int(train.nunique(dropna=True)),
        "test_unique_count": int(test.nunique(dropna=True)),
        "category_shift": {
            "train_only_count": len(train_only),
            "test_only_count": len(test_only),
            "train_only_top": _top_categories(train, train_counts, train_only),
            "test_only_top": _top_categories(test, test_counts, test_only),
            "train_top": _top_categories(train, train_counts),
            "test_top": _top_categories(test, test_counts),
        },
    }


def _temporal_values(values: pd.Series) -> tuple[pd.Series, str] | None:
    present = values.notna()
    if not present.any():
        return None
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric[present].notna().all() and np.isfinite(numeric[present].astype(float)).all():
        result = numeric.astype(float)
        return result, "numeric"
    try:
        dates = pd.to_datetime(values, errors="coerce", utc=True)
    except (ValueError, TypeError, OverflowError):
        return None
    if dates[present].notna().all():
        seconds = dates.astype("int64", copy=False).astype(float) / 1_000_000_000
        seconds = seconds.where(dates.notna(), np.nan)
        return seconds, "datetime"
    return None


def _temporal_candidates(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for column in columns:
        if not _TEMPORAL_WORDS.search(re.sub(r"([a-z])([A-Z])", r"\1_\2", column)):
            continue
        parsed_train = _temporal_values(train[column])
        parsed_test = _temporal_values(test[column])
        if parsed_train is None or parsed_test is None:
            continue
        train_values, kind = parsed_train
        test_values, test_kind = parsed_test
        if kind != test_kind or train_values.notna().sum() != len(train_values) or test_values.notna().sum() != len(test_values):
            continue
        train_groups = int(train_values.nunique())
        train_max = float(train_values.max())
        if train_groups < 3 or not bool((test_values > train_max).all()):
            continue
        def display(value: float) -> Any:
            if kind == "datetime":
                return pd.to_datetime(value, unit="s", utc=True).isoformat()
            return float(value) if not value.is_integer() else int(value)
        candidates.append({
            "feature": column,
            "kind": kind,
            "future_only": True,
            "direction": "future",
            "train_group_count": train_groups,
            "train_min": display(float(train_values.min())),
            "train_max": display(train_max),
            "test_min": display(float(test_values.min())),
            "test_max": display(float(test_values.max())),
        })
    return candidates


def _cv_columns(columns: list[str], temporal: list[dict[str, Any]]) -> list[str]:
    temporal_names = [item["feature"] for item in temporal]
    ordered = [*temporal_names, *(name for name in columns if name not in temporal_names)]
    return ordered[:_MAX_FEATURES_IN_CV]


def _semantic_temporal_columns(columns: list[str]) -> list[str]:
    return [
        column for column in columns
        if _TEMPORAL_WORDS.search(re.sub(r"([a-z])([A-Z])", r"\1_\2", column))
    ]


def _adversarial_cv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    *,
    n_jobs: int,
    seed: int,
    categorical_columns: set[str],
) -> dict[str, Any]:
    if len(train) < _MIN_DOMAIN_ROWS_FOR_CV or len(test) < _MIN_DOMAIN_ROWS_FOR_CV:
        return {
            "status": "skipped",
            "reason": "Three-fold adversarial CV requires at least 100 sampled rows per domain.",
            "rows_per_domain": {"train": int(len(train)), "test": int(len(test))},
        }
    if not columns:
        return {"status": "skipped", "reason": "No eligible feature columns remain."}

    combined = pd.concat([train.loc[:, columns], test.loc[:, columns]], ignore_index=True)
    domain = np.concatenate([np.zeros(len(train), dtype=np.int8), np.ones(len(test), dtype=np.int8)])
    numeric_columns = [
        name for name in columns
        if _numeric_pair(train[name], test[name], force_category=name in categorical_columns)
    ]
    category_columns = [name for name in columns if name not in numeric_columns]

    for name in numeric_columns:
        combined[name] = pd.to_numeric(combined[name], errors="coerce").astype(float)
        combined[name] = combined[name].where(np.isfinite(combined[name]), np.nan)
    for name in category_columns:
        values = combined[name].astype("object")
        combined[name] = values.where(values.notna(), "__missing__").astype(str)

    branches = []
    if numeric_columns:
        branches.append(("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric_columns))
    if category_columns:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=2,
            max_categories=_MAX_CATEGORY_LEVELS,
            sparse_output=True,
            dtype=np.float32,
        )
        branches.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("encode", encoder),
        ]), category_columns))
    preprocessor = ColumnTransformer(branches, remainder="drop", sparse_threshold=1.0)
    estimator = ExtraTreesClassifier(
        n_estimators=60,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=max(1, min(int(n_jobs), _MAX_THREADS)),
        random_state=int(seed),
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(seed))
    scores = cross_val_score(
        pipeline, combined, domain, scoring="roc_auc", cv=folds, n_jobs=1, error_score="raise"
    )
    return {
        "status": "ok",
        "metric": "roc_auc",
        "fold_roc_auc": [float(score) for score in scores],
        "roc_auc_mean": float(np.mean(scores)),
        "roc_auc_std": float(np.std(scores)),
        "folds": 3,
        "rows_per_domain": {"train": int(len(train)), "test": int(len(test))},
        "feature_names": list(columns),
        "feature_count": len(columns),
    }


def dataset_diagnostics(task: Any, n_jobs: int = 4, max_rows: int = 10000) -> dict:
    """Measure feature drift and bounded adversarial train/test separability."""
    if not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    train, test, columns = _read_feature_frames(task)
    if not columns:
        return {
            "features": {},
            "total_features": 0,
            "features_partial": False,
            "temporal_candidates": [],
            "adversarial_cv": {"status": "skipped", "reason": "No eligible feature columns."},
        }

    temporal = _temporal_candidates(train, test, columns)
    forced_categorical = set(_value(task, "categorical_columns", []) or [])
    profile_names = columns[:_MAX_FEATURES_IN_PROFILE]
    feature_stats = {}
    for column in profile_names:
        if _numeric_pair(train[column], test[column], column in forced_categorical):
            feature_stats[column] = {"kind": "numeric", **_numeric_stats(train[column], test[column])}
        else:
            feature_stats[column] = {"kind": "categorical", **_categorical_stats(train[column], test[column])}

    row_limit = min(int(max_rows), _MAX_ROWS_PER_DOMAIN)
    seed = _value(task, "seed", 42)
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = 42
    train_sample = train if len(train) <= row_limit else train.sample(n=row_limit, random_state=seed)
    test_sample = test if len(test) <= row_limit else test.sample(n=row_limit, random_state=seed + 1)
    auc_names = _cv_columns(columns, temporal)
    forced_categorical = set(_value(task, "categorical_columns", []) or [])
    try:
        thread_count = max(1, min(int(n_jobs), _MAX_THREADS))
    except (TypeError, ValueError):
        thread_count = 1
    cv = _adversarial_cv(
        train_sample,
        test_sample,
        auc_names,
        n_jobs=thread_count,
        seed=seed,
        categorical_columns=forced_categorical,
    )
    cv["total_feature_count"] = len(columns)
    cv["features_truncated"] = len(auc_names) < len(columns)
    temporal_names = _semantic_temporal_columns(auc_names)
    temporal_name_set = set(temporal_names)
    without_temporal_names = [name for name in auc_names if name not in temporal_name_set]
    if cv.get("status") == "ok" and temporal_names and without_temporal_names:
        cv["without_temporal"] = _adversarial_cv(
            train_sample,
            test_sample,
            without_temporal_names,
            n_jobs=thread_count,
            seed=seed,
            categorical_columns=forced_categorical,
        )
        cv["temporal_features_excluded"] = temporal_names
    else:
        cv["without_temporal"] = {
            "status": "skipped",
            "reason": "No other semantic temporal feature or the primary CV was skipped.",
        }
    return {
        "features": feature_stats,
        "total_features": len(columns),
        "features_partial": len(columns) > _MAX_FEATURES_IN_PROFILE,
        "temporal_candidates": temporal,
        "adversarial_cv": cv,
    }


def feature_importance(model: Any, context: Any) -> dict:
    """Return normalized importances only when a trusted mapping is available."""
    unsupported = {"status": "unsupported", "items": []}
    original_features = list(_value(context, "feature_columns", []) or [])
    candidate = model
    visited = set()
    while not isinstance(candidate, Pipeline):
        if id(candidate) in visited:
            return {**unsupported, "reason": "cyclic model wrapper"}
        visited.add(id(candidate))
        module = type(candidate).__module__
        if (module, type(candidate).__name__) in (_KNOWN_TREE_ESTIMATORS | _KNOWN_LINEAR_ESTIMATORS):
            break
        if module.startswith("automl_agent."):
            try:
                inner_model = vars(candidate).get("model")
            except TypeError:
                inner_model = None
            if inner_model is not None:
                candidate = inner_model
                continue
        return {**unsupported, "reason": "model type is not in the trusted estimator allowlist"}

    if isinstance(candidate, Pipeline):
        if not candidate.steps:
            return {**unsupported, "reason": "empty sklearn Pipeline"}
        estimator = candidate.steps[-1][1]
        preprocessor = next(
            (step for _, step in candidate.steps[:-1] if isinstance(step, ColumnTransformer)),
            None,
        )
    else:
        estimator = candidate
        preprocessor = None
    # Host XGBoost label decoder preserves the original class axis.
    if type(estimator).__module__ == 'automl_agent.training' and type(estimator).__name__ == '_XGBLabelDecodingClassifier':
        estimator = vars(estimator).get('_model', vars(estimator).get('estimator'))
    estimator_key = (type(estimator).__module__, type(estimator).__name__)
    try:
        if estimator_key in _KNOWN_TREE_ESTIMATORS:
            raw_importance = getattr(estimator, "feature_importances_", None)
            method = "native_feature_importances_"
        elif estimator_key in _KNOWN_LINEAR_ESTIMATORS:
            raw_importance = getattr(estimator, "coef_", None)
            method = "absolute_coefficient"
        else:
            return {**unsupported, "reason": "Pipeline final estimator is not in the trusted estimator allowlist"}
    except Exception:
        return {**unsupported, "reason": "fitted model does not expose supported importance values"}
    if raw_importance is None:
        return {**unsupported, "reason": "fitted model does not expose supported importance values"}
    try:
        values = np.asarray(raw_importance, dtype=float)
    except (TypeError, ValueError):
        return {**unsupported, "reason": "importance values are not numeric"}
    if values.ndim == 2:
        values = np.mean(np.abs(values), axis=0)
    elif method == "absolute_coefficient":
        values = np.abs(values.reshape(-1))
    else:
        values = values.reshape(-1)
    if values.ndim != 1 or not np.isfinite(values).all():
        return {**unsupported, "reason": "importance values are not a finite vector"}

    if not original_features:
        source = preprocessor if preprocessor is not None else estimator
        try:
            original_features = list(source.feature_names_in_)
        except (AttributeError, TypeError):
            original_features = []
    try:
        mapping = _map_to_original_features(preprocessor, len(values), original_features)
    except (AttributeError, KeyError, TypeError, ValueError):
        mapping = None
    if mapping is None:
        return {**unsupported, "reason": "cannot reliably map transformed features to original columns"}
    total = float(np.sum(values))
    if total <= 0:
        return {**unsupported, "reason": "all supported feature importances are zero"}
    normalized = values / total
    aggregated: dict[str, float] = {feature: 0.0 for feature in original_features}
    for feature, importance in zip(mapping, normalized, strict=True):
        aggregated[feature] = aggregated.get(feature, 0.0) + float(importance)
    items = [
        {"feature": feature, "importance": float(value)}
        for feature, value in aggregated.items()
    ]
    items.sort(key=lambda item: (-item["importance"], item["feature"]))
    return {
        "status": "ok",
        "items": items,
        "method": method,
        "mapping_status": "original_features",
        "normalization": "sum_to_one",
        "interpretation": "Model association only; feature importance is not causal.",
    }


def _map_to_original_features(
    preprocessor: ColumnTransformer | None,
    importance_count: int,
    original_features: list[str],
) -> list[str] | None:
    if preprocessor is None:
        return list(original_features) if len(original_features) == importance_count else None
    if not isinstance(preprocessor, ColumnTransformer) or not original_features:
        return None
    output_indices = preprocessor.output_indices_
    mapping: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or (isinstance(transformer, str) and transformer == "drop"):
            continue
        try:
            source_columns = list(columns)
        except TypeError:
            return None
        if any(not isinstance(column, str) for column in source_columns):
            if all(isinstance(column, (int, np.integer)) for column in source_columns):
                source_columns = [original_features[int(column)] for column in source_columns]
            else:
                return None
        output_slice = output_indices.get(name)
        if not isinstance(output_slice, slice) or output_slice.start is None or output_slice.stop is None:
            return None
        output_count = output_slice.stop - output_slice.start
        if output_count == 0:
            continue
        branch_mapping = _branch_feature_mapping(transformer, source_columns, output_count)
        if branch_mapping is None:
            return None
        mapping.extend(branch_mapping)
    return mapping if len(mapping) == importance_count else None


def _branch_feature_mapping(transformer: Any, columns: list[str], output_count: int) -> list[str] | None:
    if isinstance(transformer, str) and transformer == "passthrough":
        return list(columns) if len(columns) == output_count else None
    if isinstance(transformer, OneHotEncoder):
        try:
            output_names = transformer.get_feature_names_out(columns).tolist()
        except (AttributeError, TypeError, ValueError):
            return None
        return _assign_encoded_names(output_names, columns) if len(output_names) == output_count else None
    steps = transformer.steps if isinstance(transformer, Pipeline) else []
    encoders = [step for _, step in steps if isinstance(step, OneHotEncoder)]
    if encoders:
        try:
            output_names = encoders[-1].get_feature_names_out(columns).tolist()
        except (AttributeError, TypeError, ValueError):
            return None
        return _assign_encoded_names(output_names, columns) if len(output_names) == output_count else None
    imputers = [step for _, step in steps if isinstance(step, SimpleImputer)]
    if isinstance(transformer, SimpleImputer):
        imputers = [transformer]
    if imputers:
        imputer = imputers[-1]
        if output_count == len(columns):
            return list(columns)
        if len(columns) != output_count:
            indicator = getattr(imputer, "indicator_", None)
            missing_indices = list(indicator.features_) if indicator is not None else []
            expected_count = len(columns) + len(missing_indices)
            if output_count != expected_count:
                return None
            return [*columns, *(columns[int(index)] for index in missing_indices)]
    return None


def _assign_encoded_names(output_names: list[str], columns: list[str]) -> list[str] | None:
    mapping = []
    for name in output_names:
        matches = [column for column in columns if name == column or name.startswith(f"{column}_")]
        if not matches:
            return None
        mapping.append(max(matches, key=len))
    return mapping
