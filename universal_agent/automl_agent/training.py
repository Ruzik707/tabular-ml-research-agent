"""Host-owned model fitting with train-only early stopping and refitting."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


_DEFAULT_OPTIONS = {
    "early_stopping_rounds": 0,
    "inner_validation_fraction": 0.15,
    "sample_weight": None,
}
_VALID_SAMPLE_WEIGHT_MODES = (None, "balanced", "sqrt_balanced")
_MAX_JSON_STRING = 160
_MAX_JSON_LIST = 64
_MAX_JSON_NUMBER = 1_000_000_000_000


class _XGBLabelDecodingClassifier(ClassifierMixin, BaseEstimator):
    """Keep XGBoost's numeric labels behind a sklearn-compatible wrapper."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None, **fit_params):
        encoder = LabelEncoder()
        encoded_y = encoder.fit_transform(np.asarray(y).reshape(-1))
        prepared_params = dict(fit_params)
        eval_set = prepared_params.get("eval_set")
        if eval_set is not None:
            if type(eval_set) not in (list, tuple):
                raise TypeError("XGBoost eval_set must be a sequence of (X, y) pairs")
            encoded_eval_set = []
            for pair in eval_set:
                if type(pair) not in (list, tuple) or len(pair) != 2:
                    raise TypeError("XGBoost eval_set entries must be (X, y) pairs")
                try:
                    encoded_valid_y = encoder.transform(np.asarray(pair[1]).reshape(-1))
                except ValueError as exc:
                    raise ValueError("XGBoost inner validation labels must match inner train classes") from exc
                encoded_eval_set.append((pair[0], encoded_valid_y))
            prepared_params["eval_set"] = encoded_eval_set

        self.classes_ = encoder.classes_
        self._encoder = encoder
        self._model = self.estimator
        if sample_weight is None:
            self._model.fit(X, encoded_y, **prepared_params)
        else:
            self._model.fit(X, encoded_y, sample_weight=sample_weight, **prepared_params)
        return self

    def predict(self, X):
        codes = np.asarray(self._model.predict(X)).reshape(-1).astype(np.int64)
        if np.any(codes < 0) or np.any(codes >= len(self.classes_)):
            raise ValueError("XGBoost predicted a label outside the fitted class mapping")
        return self.classes_[codes]

    def predict_proba(self, X):
        # XGBoost's encoded integer axis uses the same order as LabelEncoder.classes_.
        return np.asarray(self._model.predict_proba(X))


def _normalise_options(options: dict[str, Any] | None) -> dict[str, Any]:
    if options is None:
        options = {}
    if type(options) is not dict:
        raise TypeError("options must be a dict")
    unknown = set(options) - set(_DEFAULT_OPTIONS)
    if unknown:
        raise ValueError(f"unknown fit option(s): {', '.join(sorted(map(str, unknown)))}")
    result = dict(_DEFAULT_OPTIONS)
    result.update(options)

    rounds = result["early_stopping_rounds"]
    if type(rounds) is not int or not 0 <= rounds <= 300:
        raise ValueError("early_stopping_rounds must be an integer from 0 through 300")
    fraction = result["inner_validation_fraction"]
    if type(fraction) not in (int, float) or type(fraction) is bool:
        raise ValueError("inner_validation_fraction must be a number from 0.1 through 0.3")
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0.1 <= fraction <= 0.3:
        raise ValueError("inner_validation_fraction must be a number from 0.1 through 0.3")
    result["inner_validation_fraction"] = fraction
    weight_mode = result["sample_weight"]
    if weight_mode is not None and (type(weight_mode) is not str or weight_mode not in _VALID_SAMPLE_WEIGHT_MODES):
        raise ValueError("sample_weight must be None, 'balanced', or 'sqrt_balanced'")
    return result


def _native_component(estimator):
    if isinstance(estimator, Pipeline):
        if not estimator.steps:
            return None
        return _native_component(estimator.steps[-1][1])
    if isinstance(estimator, _XGBLabelDecodingClassifier):
        return _native_component(estimator.estimator)
    return estimator


def _backend_info(estimator) -> tuple[str | None, str | None]:
    """Identify exact supported native classes without importing optional ML packages."""
    native = _native_component(estimator)
    if native is None:
        return None, None
    try:
        cls = type(native)
        module = type.__getattribute__(cls, "__module__")
        name = type.__getattribute__(cls, "__name__")
    except Exception:
        return None, None
    exact = {
        ("catboost.core", "CatBoostClassifier"): ("catboost", "classification"),
        ("catboost.core", "CatBoostRegressor"): ("catboost", "regression"),
        ("lightgbm.sklearn", "LGBMClassifier"): ("lightgbm", "classification"),
        ("lightgbm.sklearn", "LGBMRegressor"): ("lightgbm", "regression"),
        ("xgboost.sklearn", "XGBClassifier"): ("xgboost", "classification"),
        ("xgboost.sklearn", "XGBRegressor"): ("xgboost", "regression"),
    }
    return exact.get((module, name), (None, None))


def _final_step(estimator):
    if isinstance(estimator, Pipeline):
        if not estimator.steps:
            raise ValueError("Pipeline must contain a final estimator")
        name, final = estimator.steps[-1]
        if final in ("passthrough", None):
            raise ValueError("Pipeline must end with a fitted estimator")
        return name, final
    return None, estimator


def _unwrap_decoder(estimator):
    return estimator.estimator if isinstance(estimator, _XGBLabelDecodingClassifier) else estimator


def _maybe_wrap_xgboost_classifier(estimator, backend):
    if backend != "xgboost" or _backend_info(estimator)[1] != "classification":
        return estimator
    _name, final = _final_step(estimator)
    if isinstance(final, _XGBLabelDecodingClassifier):
        return estimator
    decoder = _XGBLabelDecodingClassifier(final)
    if isinstance(estimator, Pipeline):
        steps = list(estimator.steps)
        steps[-1] = (steps[-1][0], decoder)
        estimator.steps = steps
        return estimator
    return decoder


def _estimator_params(estimator, backend):
    native = _native_component(estimator)
    if native is None:
        return {}
    if backend == "catboost":
        state = vars(native).get("_init_params")
        return state if type(state) is dict else {}
    try:
        params = native.get_params(deep=False)
    except Exception:
        params = {}
    return params if type(params) is dict else {}


def _setting_enabled(key, value):
    if value is None:
        return False
    if key == "scale_pos_weight":
        return value != 1
    if key == "is_unbalance":
        return value is True
    if type(value) is bool:
        return value
    if type(value) in (int, float):
        return value != 0
    if type(value) is str:
        return bool(value)
    if type(value) in (dict, list, tuple, set):
        return len(value) > 0
    return True


def _reject_double_weighting(estimator, backend, sample_weight_mode):
    if sample_weight_mode is None:
        return
    params = _estimator_params(estimator, backend)
    keys = ("class_weights", "auto_class_weights") if backend == "catboost" else (
        "class_weight", "class_weights", "auto_class_weights", "is_unbalance", "scale_pos_weight"
    )
    enabled = [key for key in keys if key in params and _setting_enabled(key, params[key])]
    if enabled:
        raise ValueError(
            "sample_weight would cause double weighting with constructor "
            + ", ".join(enabled)
            + "; disable constructor weighting first"
        )


def _row_weights(y, mode):
    if mode is None:
        return None
    labels, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    if len(labels) < 2:
        raise ValueError("balanced sample_weight requires at least two classes")
    per_class = len(y) / (len(labels) * counts.astype(float))
    if mode == "sqrt_balanced":
        per_class = np.sqrt(per_class)
        row_values = per_class[inverse]
        row_values = row_values / float(np.mean(row_values))
        return row_values.astype(float, copy=False)
    return per_class[inverse].astype(float, copy=False)


def _take_rows(value, indices):
    if hasattr(value, "iloc"):
        return value.iloc[indices]
    return value[indices]


def _indices(value, name, row_count):
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ValueError(f"inner_split {name} must be a one-dimensional integer index sequence")
    array = array.astype(np.int64, copy=False)
    if len(array) == 0:
        raise ValueError(f"inner_split {name} cannot be empty")
    if len(np.unique(array)) != len(array):
        raise ValueError(f"inner_split {name} cannot contain duplicate indices")
    if int(array.min()) < 0 or int(array.max()) >= row_count:
        raise ValueError(f"inner_split {name} indices are outside outer training rows")
    return array


def _normalise_inner_split(inner_split, y, task_type, fraction, seed):
    row_count = len(y)
    if inner_split is None:
        if row_count < 4:
            return None, "inner_split_skipped_too_few_rows"
        stratify = None
        if task_type == "classification":
            classes, counts = np.unique(y, return_counts=True)
            if len(classes) < 2:
                return None, "inner_split_skipped_rare_class"
            if int(counts.min()) < 2:
                return None, "inner_split_skipped_rare_class"
            validation_count = int(math.ceil(row_count * fraction))
            if validation_count < len(classes) or row_count - validation_count < len(classes):
                return None, "inner_split_skipped_too_few_rows"
            stratify = y
        train_indices, valid_indices = train_test_split(
            np.arange(row_count),
            test_size=fraction,
            random_state=seed,
            stratify=stratify,
        )
    else:
        if type(inner_split) is dict:
            if "train_indices" in inner_split and "valid_indices" in inner_split:
                train_values = inner_split["train_indices"]
                valid_values = inner_split["valid_indices"]
            elif "train" in inner_split and "valid" in inner_split:
                train_values = inner_split["train"]
                valid_values = inner_split["valid"]
            else:
                raise ValueError("inner_split dict must contain train_indices/valid_indices")
        elif type(inner_split) in (tuple, list) and len(inner_split) == 2:
            train_values, valid_values = inner_split
        else:
            raise ValueError("inner_split must be a pair of local index sequences")
        train_indices = _indices(train_values, "train_indices", row_count)
        valid_indices = _indices(valid_values, "valid_indices", row_count)
        if np.intersect1d(train_indices, valid_indices).size:
            raise ValueError("inner_split train and valid indices must be disjoint")

    if len(train_indices) < 2 or len(valid_indices) < 2:
        return None, "inner_split_skipped_too_few_rows"
    if task_type == "classification":
        all_classes = set(np.unique(y).tolist())
        train_classes = set(np.unique(y[train_indices]).tolist())
        valid_classes = set(np.unique(y[valid_indices]).tolist())
        if len(all_classes) < 2 or train_classes != all_classes or valid_classes != all_classes:
            return None, "inner_split_skipped_rare_class"
    return (np.asarray(train_indices), np.asarray(valid_indices)), None


def _fit_pipeline_prefix(pipeline, X_train, y_train, X_valid):
    transformed_train = X_train
    transformed_valid = X_valid
    for _name, transformer in pipeline.steps[:-1]:
        if transformer in ("passthrough", None):
            continue
        if callable(getattr(transformer, "fit_transform", None)):
            transformed_train = transformer.fit_transform(transformed_train, y_train)
        else:
            transformer.fit(transformed_train, y_train)
            transformed_train = transformer.transform(transformed_train)
        transformed_valid = transformer.transform(transformed_valid)
    return transformed_train, transformed_valid


def _fit_estimator_with_eval(estimator, backend, X_train, y_train, X_valid, y_valid, weights, rounds):
    if backend == "catboost":
        kwargs = {"eval_set": (X_valid, y_valid), "early_stopping_rounds": rounds, "verbose": False}
    elif backend == "lightgbm":
        from lightgbm import early_stopping, log_evaluation

        kwargs = {
            "eval_set": [(X_valid, y_valid)],
            "callbacks": [early_stopping(rounds, verbose=False), log_evaluation(period=0)],
        }
    elif backend == "xgboost":
        native = _unwrap_decoder(estimator)
        native.set_params(early_stopping_rounds=rounds)
        kwargs = {"eval_set": [(X_valid, y_valid)], "verbose": False}
    else:
        raise ValueError("early stopping is only supported for native CatBoost, LightGBM, and XGBoost")
    if weights is not None:
        kwargs["sample_weight"] = weights
    estimator.fit(X_train, y_train, **kwargs)


def _fit_probe(estimator, backend, X_train, y_train, X_valid, y_valid, weights, rounds):
    if isinstance(estimator, Pipeline):
        transformed_train, transformed_valid = _fit_pipeline_prefix(estimator, X_train, y_train, X_valid)
        _name, final = _final_step(estimator)
        _fit_estimator_with_eval(
            final, backend, transformed_train, y_train, transformed_valid, y_valid, weights, rounds
        )
    else:
        _fit_estimator_with_eval(estimator, backend, X_train, y_train, X_valid, y_valid, weights, rounds)


def _set_iteration_count(estimator, backend, count, *, clear_early_stopping=False):
    native = _native_component(estimator)
    if backend not in ("catboost", "lightgbm", "xgboost") or native is None:
        raise ValueError("iteration_override requires a supported native boosting estimator")
    parameter = "iterations" if backend == "catboost" else "n_estimators"
    settings = {parameter: int(count)}
    if backend == "xgboost" and clear_early_stopping:
        settings["early_stopping_rounds"] = None
    native.set_params(**settings)


def _requested_iterations(estimator, backend):
    native = _native_component(estimator)
    if native is None:
        return None
    if backend == "catboost":
        state = vars(native).get("_init_params")
        if type(state) is dict:
            value = state.get("iterations", state.get("n_estimators"))
            if type(value) is int and value > 0:
                return value
        return 1000
    params = _estimator_params(estimator, backend)
    value = params.get("n_estimators")
    return int(value) if type(value) is int and value > 0 else 100


def _best_iteration(estimator, backend, fallback):
    native = _native_component(estimator)
    if native is None:
        return fallback
    try:
        if backend == "catboost":
            best = native.get_best_iteration()
            if type(best) is int and best >= 0:
                return best + 1
            count = getattr(native, "tree_count_", None)
        elif backend == "lightgbm":
            best = getattr(native, "best_iteration_", 0)
            if type(best) is int and best > 0:
                return best
            count = getattr(native, "n_estimators_", None)
        else:
            best = getattr(native, "best_iteration", None)
            if type(best) is int and best >= 0:
                return best + 1
            count = native.get_booster().num_boosted_rounds()
        if type(count) is int and count > 0:
            return count
    except Exception:
        pass
    return fallback


def _tree_count(estimator, backend):
    return _best_iteration(estimator, backend, None)


def _json_parameter(value):
    if value is None or type(value) in (bool, int, str):
        if type(value) is int and abs(value) > _MAX_JSON_NUMBER:
            return None
        if type(value) is str:
            return value[:_MAX_JSON_STRING]
        return value
    if type(value) is float:
        if math.isfinite(value) and abs(value) <= _MAX_JSON_NUMBER:
            return value
        return None
    if type(value) in (tuple, list):
        source = list.__iter__(value) if type(value) is list else tuple.__iter__(value)
        result = []
        for item in list(source)[:_MAX_JSON_LIST]:
            safe = _json_parameter(item)
            if safe is None and item is not None:
                return None
            result.append(safe)
        return result
    return None


def _trained_parameters(estimator, backend):
    native = _native_component(estimator)
    if native is None or backend is None:
        return {}, "unsupported_estimator"
    try:
        if backend == "catboost":
            params = native.get_all_params()
            fields = (
                "iterations", "learning_rate", "depth", "l2_leaf_reg", "border_count",
                "auto_class_weights", "class_weights",
            )
            source = "catboost.get_all_params"
        else:
            params = native.get_params(deep=False)
            if backend == "lightgbm":
                fields = ("n_estimators", "learning_rate", "num_leaves", "max_depth", "class_weight", "n_jobs")
            else:
                fields = ("n_estimators", "learning_rate", "max_depth", "max_leaves", "tree_method", "scale_pos_weight", "n_jobs")
            source = backend + ".get_params"
        if type(params) is not dict:
            return {}, source + "_unavailable"
        result = {}
        for field in fields:
            if field in params:
                value = _json_parameter(params[field])
                if value is not None or params[field] is None:
                    result[field] = value
        return result, source
    except Exception:
        return {}, backend + ".parameters_unavailable"


def _fit_outer(estimator, X, y, weights):
    if isinstance(estimator, Pipeline):
        if weights is None:
            estimator.fit(X, y)
        else:
            final_name, _final = _final_step(estimator)
            estimator.fit(X, y, **{final_name + "__sample_weight": weights})
    elif weights is None:
        estimator.fit(X, y)
    else:
        estimator.fit(X, y, sample_weight=weights)
    return estimator


def fit_model(model, X, y, *, context, options=None, iteration_override=None, inner_split=None):
    """Fit a fresh estimator, using only its supplied outer-training data.

    For boosting, early stopping trains an inner probe and then clones the
    original model again for a full outer-training refit. Outer validation
    features and labels are deliberately absent from this API.
    """
    started = time.perf_counter()
    fit_options = _normalise_options(options)
    if type(context) is not dict:
        raise TypeError("context must be a dict")
    task_type = context.get("task_type")
    if task_type not in ("classification", "regression"):
        raise ValueError("context.task_type must be classification or regression")
    y_array = np.asarray(y).reshape(-1)
    if len(y_array) == 0 or len(X) != len(y_array):
        raise ValueError("X and y must have the same non-zero number of rows")
    if iteration_override is not None and (
        type(iteration_override) is not int or not 1 <= iteration_override <= 100_000
    ):
        raise ValueError("iteration_override must be an integer from 1 through 100000")

    backend, backend_task = _backend_info(model)
    if backend_task is not None and backend_task != task_type:
        raise ValueError(f"{backend} estimator task type does not match context.task_type")
    sample_weight_mode = fit_options["sample_weight"]
    if sample_weight_mode is not None and task_type != "classification":
        raise ValueError("balanced sample_weight is only supported for classification")
    _reject_double_weighting(model, backend, sample_weight_mode)

    # The host may pass a previous CV iteration count to every candidate. Keep
    # that control meaningful for native boosters while allowing ordinary
    # sklearn estimators to ignore it without inventing a tree count.
    effective_iteration_override = iteration_override if backend is not None else None
    requested_iterations = (
        effective_iteration_override or _requested_iterations(model, backend)
        if backend is not None
        else None
    )
    rounds = fit_options["early_stopping_rounds"]
    reason = None
    selected_iteration = effective_iteration_override
    inner_train_rows = None
    inner_valid_rows = None
    early_stopping_used = False
    probe_seconds = 0.0

    if effective_iteration_override is not None:
        reason = "iteration_override_supplied"
    elif rounds == 0:
        reason = (
            "iteration_override_ignored_non_booster"
            if iteration_override is not None
            else "early_stopping_disabled"
        )
        params = _estimator_params(model, backend)
        if backend == "xgboost" and params.get("early_stopping_rounds") not in (None, 0):
            raise ValueError("XGBoost constructor early stopping requires host fit options and an inner split")
        if backend == "catboost":
            init_params = vars(_native_component(model)).get("_init_params", {})
            if type(init_params) is dict and init_params.get("early_stopping_rounds") not in (None, 0):
                raise ValueError("CatBoost constructor early stopping requires host fit options and an inner split")
    else:
        if backend is None:
            raise ValueError("early_stopping_rounds requires native CatBoost, LightGBM, or XGBoost")
        seed = context.get("seed", 42)
        if type(seed) is not int:
            seed = 42
        split, split_reason = _normalise_inner_split(
            inner_split,
            y_array,
            task_type,
            fit_options["inner_validation_fraction"],
            seed,
        )
        if split is None:
            reason = split_reason
        else:
            train_indices, valid_indices = split
            X_inner_train = _take_rows(X, train_indices)
            y_inner_train = y_array[train_indices]
            X_inner_valid = _take_rows(X, valid_indices)
            y_inner_valid = y_array[valid_indices]
            inner_weights = _row_weights(y_inner_train, sample_weight_mode)
            probe = clone(model)
            probe = _maybe_wrap_xgboost_classifier(probe, backend)
            probe_start = time.perf_counter()
            _fit_probe(
                probe,
                backend,
                X_inner_train,
                y_inner_train,
                X_inner_valid,
                y_inner_valid,
                inner_weights,
                rounds,
            )
            probe_seconds = time.perf_counter() - probe_start
            selected_iteration = _best_iteration(probe, backend, requested_iterations)
            if selected_iteration is None or selected_iteration < 1:
                selected_iteration = requested_iterations
            early_stopping_used = True
            inner_train_rows = int(len(train_indices))
            inner_valid_rows = int(len(valid_indices))

    # The public code interface also accepts plain fit/predict classes. Builders
    # provide fresh objects per fold; only native early-stop probes require clone.
    try:
        final_model = clone(model)
    except TypeError:
        if backend is not None:
            raise
        final_model = model
    final_model = _maybe_wrap_xgboost_classifier(final_model, backend)
    if backend is not None and selected_iteration is not None:
        _set_iteration_count(final_model, backend, selected_iteration, clear_early_stopping=True)
    elif backend == "xgboost":
        _native_component(final_model).set_params(early_stopping_rounds=None)

    outer_weights = _row_weights(y_array, sample_weight_mode)
    refit_start = time.perf_counter()
    fitted_model = _fit_outer(final_model, X, y_array, outer_weights)
    refit_seconds = time.perf_counter() - refit_start
    actual_iteration = _tree_count(fitted_model, backend)
    if actual_iteration is None and backend is not None:
        actual_iteration = selected_iteration or requested_iterations
    parameters, parameters_source = _trained_parameters(fitted_model, backend)
    summary = {
        "fit_seconds": float(time.perf_counter() - started),
        "refit_seconds": float(refit_seconds),
        "best_iteration": int(actual_iteration) if actual_iteration is not None else None,
        "requested_iterations": int(requested_iterations) if requested_iterations is not None else None,
        "early_stopping_used": bool(early_stopping_used),
        "inner_train_rows": inner_train_rows,
        "inner_valid_rows": inner_valid_rows,
        "train_rows": int(len(y_array)),
        "sample_weight_mode": sample_weight_mode,
        "reason": reason,
        "trained_parameters": parameters,
        "parameters_source": parameters_source,
    }
    return fitted_model, summary
