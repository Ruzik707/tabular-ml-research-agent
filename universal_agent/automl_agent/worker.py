"""Isolated model-fitting subprocess; validation labels stay in the host."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import cloudpickle


_MAX_ESTIMATOR_DEPTH = 6
_MAX_ESTIMATOR_COUNT = 32
_MAX_PIPELINE_STEPS = 32
_MAX_METADATA_NAME_LENGTH = 160
_ESTIMATOR_LINKS = (
    # Prefer fitted links. Following both estimator_ and the original estimator
    # would incorrectly present an unfitted template as a trained backend.
    "best_estimator_", "estimator_", "regressor_", "classifier_", "final_estimator_",
    "model", "_model", "_estimator", "estimator", "base_estimator_",
    "base_estimator", "final_estimator",
)
_NATIVE_PARAMETER_TYPES: dict[type, tuple[str, tuple[str, ...], Any]] | None = None
_MAX_PARAMETER_DEPTH = 1
_MAX_PARAMETER_LIST_ITEMS = 64
_MAX_PARAMETER_STRING_LENGTH = 160
_MAX_PARAMETER_NUMBER = 1_000_000_000_000
_UNSAFE_PARAMETER = object()


def _class_name(value: Any) -> str:
    """Return a bounded fully-qualified class name without inspecting the value."""
    try:
        cls = type(value)
        module = type.__getattribute__(cls, "__module__")
        name = type.__getattribute__(cls, "__qualname__")
        if not isinstance(module, str) or not isinstance(name, str):
            raise TypeError("class name is not text")
        result = f"{module}.{name}"
    except Exception:
        result = "builtins.object"
    return result[:_MAX_METADATA_NAME_LENGTH]


def _instance_dict(value: Any) -> dict:
    """Read instance state while bypassing custom __getattribute__ methods."""
    try:
        attributes = object.__getattribute__(value, "__dict__")
    except Exception:
        return {}
    return attributes if isinstance(attributes, dict) else {}


def _is_sklearn_pipeline(value: Any) -> bool:
    try:
        mro = type.__getattribute__(type(value), "__mro__")
        return any(
            type.__getattribute__(base, "__module__") == "sklearn.pipeline"
            and type.__getattribute__(base, "__name__") == "Pipeline"
            for base in mro
        )
    except Exception:
        return False


def _pipeline_step_pairs(value: Any) -> list[tuple[str, Any]]:
    raw_steps = _instance_dict(value).get("steps")
    if not isinstance(raw_steps, (list, tuple)):
        return []
    pairs = []
    if isinstance(raw_steps, list):
        step_iterator = list.__iter__(raw_steps)
    else:
        step_iterator = tuple.__iter__(raw_steps)
    for index, pair in enumerate(itertools.islice(step_iterator, _MAX_PIPELINE_STEPS)):
        if type(pair) not in (list, tuple) or len(pair) != 2:
            continue
        if type(pair) is list:
            name, step = list.__getitem__(pair, 0), list.__getitem__(pair, 1)
        else:
            name, step = tuple.__getitem__(pair, 0), tuple.__getitem__(pair, 1)
        # sklearn step names are strings. Avoid calling custom __str__ methods.
        if type(name) is str:
            safe_name = name[:_MAX_METADATA_NAME_LENGTH]
        else:
            safe_name = f"step_{index}"
        pairs.append((safe_name, step))
    return pairs


def _linked_estimators(value: Any) -> list[Any]:
    # sklearn ensembles expose original base-estimator templates as well as
    # fitted members. The ensemble is the predictor; neither is a next step.
    if _class_name(value).startswith("sklearn.ensemble."):
        return []
    attributes = _instance_dict(value)
    for key in _ESTIMATOR_LINKS:
        child = attributes.get(key)
        # Ensemble members are parallel branches, not a chain of algorithms.
        # Keep the fitted ensemble class, not dozens of individual trees.
        if child is None or isinstance(child, (str, bytes, int, float, bool, dict, list, tuple)):
            continue
        return [child]
    return []


def _capture_native_parameter_types() -> dict[type, tuple[str, tuple[str, ...], Any]]:
    """Capture exact supported native classes and CatBoost's original getter.

    The worker calls this before importing candidate code. That lets metadata
    recognize the real classes by identity and avoids trusting methods or
    module attributes that candidate code might replace later.
    """
    global _NATIVE_PARAMETER_TYPES
    if _NATIVE_PARAMETER_TYPES is not None:
        return _NATIVE_PARAMETER_TYPES

    known: dict[type, tuple[str, tuple[str, ...], Any]] = {}
    catboost_fields = (
        "iterations", "learning_rate", "depth", "l2_leaf_reg", "border_count",
        "auto_class_weights", "class_weights",
    )
    try:
        import catboost

        package_values = vars(catboost)
        for class_name in ("CatBoostClassifier", "CatBoostRegressor"):
            estimator_class = package_values.get(class_name)
            if not isinstance(estimator_class, type):
                continue
            try:
                get_all_params = type.__getattribute__(estimator_class, "get_all_params")
            except Exception:
                continue
            known[estimator_class] = ("catboost", catboost_fields, get_all_params)
    except Exception:
        pass

    sklearn_tree_fields = (
        "n_estimators", "max_depth", "max_leaf_nodes", "min_samples_leaf",
        "max_features", "class_weight", "n_jobs",
    )
    sklearn_linear_fields = ("C", "max_iter", "solver", "class_weight", "n_jobs")
    sklearn_regression_fields = ("n_jobs", "fit_intercept", "positive")
    try:
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        from sklearn.linear_model import LinearRegression, LogisticRegression

        for estimator_class in (ExtraTreesClassifier, ExtraTreesRegressor):
            known[estimator_class] = ("sklearn", sklearn_tree_fields, None)
        known[LogisticRegression] = ("sklearn", sklearn_linear_fields, None)
        known[LinearRegression] = ("sklearn", sklearn_regression_fields, None)
    except Exception:
        pass

    _NATIVE_PARAMETER_TYPES = known
    return known


def _bounded_parameter_value(value: Any, name: str, depth: int = 0) -> Any:
    """Convert expected native parameter values to bounded JSON primitives."""
    if value is None or type(value) in (bool, int, str):
        if type(value) is int and abs(value) > _MAX_PARAMETER_NUMBER:
            return _UNSAFE_PARAMETER
        if type(value) is str:
            return value[:_MAX_PARAMETER_STRING_LENGTH]
        return value
    if type(value) is float:
        if not math.isfinite(value) or abs(value) > _MAX_PARAMETER_NUMBER:
            return _UNSAFE_PARAMETER
        return value
    if type(value) in (list, tuple, dict) and depth >= _MAX_PARAMETER_DEPTH:
        return _UNSAFE_PARAMETER
    if type(value) in (list, tuple):
        iterator = list.__iter__(value) if type(value) is list else tuple.__iter__(value)
        result = []
        for item in itertools.islice(iterator, _MAX_PARAMETER_LIST_ITEMS):
            safe_item = _bounded_parameter_value(item, name, depth + 1)
            if safe_item is _UNSAFE_PARAMETER:
                return _UNSAFE_PARAMETER
            result.append(safe_item)
        return result
    if type(value) is dict and name in ("class_weight", "class_weights"):
        # Mapping keys are class labels (target values), so retain only bounded
        # numeric weights and explicitly omit the labels.
        weights = []
        for _, item in itertools.islice(dict.items(value), _MAX_PARAMETER_LIST_ITEMS):
            safe_item = _bounded_parameter_value(item, name, depth + 1)
            if type(safe_item) not in (int, float) or type(safe_item) is bool:
                return _UNSAFE_PARAMETER
            weights.append(safe_item)
        return {"weights": weights, "class_labels_omitted": True}
    return _UNSAFE_PARAMETER


def _realized_parameters(value: Any) -> tuple[dict[str, Any], str]:
    """Return whitelisted parameters for one exact supported estimator class."""
    native = _capture_native_parameter_types().get(type(value))
    if native is None:
        return {}, "unsupported_estimator"

    family, fields, getter = native
    try:
        if family == "catboost":
            raw = getter(value)
            if type(raw) is not dict:
                return {}, "catboost.get_all_params_unavailable"
            result = {}
            for name in fields:
                if not dict.__contains__(raw, name):
                    continue
                raw_value = dict.__getitem__(raw, name)
                safe_value = _bounded_parameter_value(raw_value, name)
                if safe_value is not _UNSAFE_PARAMETER:
                    result[name] = safe_value

            source = "catboost.get_all_params"
            native_thread_count = dict.get(raw, "thread_count")
            thread_count_source = source
            if native_thread_count is None:
                init_params = _instance_dict(value).get("_init_params")
                if type(init_params) is dict and dict.__contains__(init_params, "thread_count"):
                    native_thread_count = dict.__getitem__(init_params, "thread_count")
                    thread_count_source = "catboost._init_params"
            if native_thread_count is not None:
                safe_thread_count = _bounded_parameter_value(native_thread_count, "thread_count")
                if safe_thread_count is not _UNSAFE_PARAMETER:
                    result["thread_count"] = safe_thread_count
                    if thread_count_source != source:
                        source += ";thread_count=" + thread_count_source
            return result, source

        attributes = _instance_dict(value)
        if type(attributes) is not dict:
            return {}, "sklearn.constructor_params_unavailable"
        result = {}
        for name in fields:
            if name not in attributes:
                continue
            safe_value = _bounded_parameter_value(attributes[name], name)
            if safe_value is not _UNSAFE_PARAMETER:
                result[name] = safe_value
        return result, "sklearn.constructor_params"
    except Exception:
        # Metadata is best-effort and must not interfere with model evaluation.
        source = "catboost.get_all_params_unavailable" if family == "catboost" else "sklearn.constructor_params_unavailable"
        return {}, source


def _estimator_details(estimator: Any) -> dict[str, Any]:
    """Describe a fitted estimator graph using only bounded, known links.

    This deliberately avoids get_params(), repr(), arbitrary properties, and
    training attributes such as feature names or target values. Metadata is
    best-effort and must never make model evaluation fail.
    """
    root_class = _class_name(estimator)
    chain: list[str] = []
    pipeline_steps: list[dict[str, str]] = []
    realized_parameters: dict[str, Any] = {}
    parameters_source = "unsupported_estimator"
    seen: set[int] = set()
    pending: list[tuple[Any, int]] = [(estimator, 0)]
    try:
        while pending and len(seen) < _MAX_ESTIMATOR_COUNT:
            current, depth = pending.pop(0)
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            current_class = _class_name(current)
            chain.append(current_class)
            current_parameters, current_source = _realized_parameters(current)
            if current_source != "unsupported_estimator":
                realized_parameters = current_parameters
                parameters_source = current_source
            if depth >= _MAX_ESTIMATOR_DEPTH:
                continue

            if _is_sklearn_pipeline(current):
                pairs = _pipeline_step_pairs(current)
                remaining_steps = _MAX_PIPELINE_STEPS - len(pipeline_steps)
                for name, step in pairs[:remaining_steps]:
                    pipeline_steps.append({"name": name, "class": _class_name(step)})
                    pending.append((step, depth + 1))
            else:
                for child in _linked_estimators(current):
                    # Ignore scalar bookkeeping values; wrapper links should
                    # lead to estimator objects, not target or feature data.
                    if child is None or isinstance(child, (str, bytes, int, float, bool)):
                        continue
                    pending.append((child, depth + 1))
    except Exception:
        # A malformed or unusual third-party wrapper must not affect the fit.
        pass
    details = {
        "class": root_class,
        "estimator_chain": chain[:_MAX_ESTIMATOR_COUNT] or [root_class],
        "pipeline_steps": pipeline_steps[:_MAX_ESTIMATOR_COUNT],
        "parameters": realized_parameters,
        "parameters_source": parameters_source,
    }
    fallback = _instance_dict(estimator).get("fallback_reason")
    if isinstance(fallback, str) and fallback:
        details["fallback_reason"] = fallback[:1000]
    return details


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False, na_rep="")
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npy(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=path.parent)
    os.close(fd)
    try:
        np.save(temporary, np.asarray(array), allow_pickle=False)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_features(path: str, columns: list[str], categorical: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns in {path}: {missing}")
    frame = frame.reindex(columns=columns)
    for name in columns:
        if name in categorical:
            continue
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan)


def _load_model(code_path: Path):
    if not code_path.is_file():
        raise FileNotFoundError(f"Не найден код модели: {code_path}")
    spec = importlib.util.spec_from_file_location(f"automl_candidate_{os.getpid()}", code_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось импортировать код модели: {code_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    cloudpickle.register_pickle_by_value(module)
    builder = getattr(module, "build_model", None)
    if not callable(builder):
        raise TypeError("Код модели должен определять build_model(context: dict)")
    return module, builder


def _merge_training_parameters(details, summary):
    """Retain evidence for native parameters and host-observed fit details."""
    if summary.get('trained_parameters'):
        details['parameters'] = {**details.get('parameters', {}), **summary['trained_parameters']}
        sources = []
        for source in (details.get('parameters_source'), summary.get('parameters_source')):
            for part in (source or '').split(';'):
                if part and part not in sources:
                    sources.append(part)
        details['parameters_source'] = ';'.join(sources)


def _run_manifest(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Manifest должен быть JSON-объектом")
    mode = manifest.get("mode")
    if mode == "finalize_ensemble":
        from .final_ensemble import run_final_ensemble
        return run_final_ensemble(manifest)
    if mode == "evaluate_cv":
        from .cv_worker import run_cv_manifest
        return run_cv_manifest(manifest)
    if mode not in {"evaluate", "finalize"}:
        raise ValueError("mode должен быть evaluate или finalize")
    inputs = manifest.get("inputs")
    context = manifest.get("context")
    if not isinstance(inputs, dict) or not isinstance(context, dict):
        raise ValueError("Manifest должен содержать объекты inputs и context")
    if "validation_target" in inputs or "validation_target" in context:
        raise ValueError("Worker не принимает validation_target: его хранит и оценивает host")
    required = {"train_features", "train_target", "test_features"}
    if mode == "evaluate":
        required.add("validation_features")
    missing = required - set(inputs)
    if missing:
        raise ValueError(f"В manifest не хватает файлов: {sorted(missing)}")
    columns = context.get("feature_columns")
    target = context.get("target")
    task_type = context.get("task_type")
    if not isinstance(columns, list) or not columns or not isinstance(target, str):
        raise ValueError("context должен задавать target и непустой feature_columns")
    if task_type not in {"regression", "classification"}:
        raise ValueError("context.task_type должен быть regression или classification")
    categorical = set(context.get("categorical_columns", []))
    train_x = _load_features(inputs["train_features"], columns, categorical)
    train_y_frame = pd.read_csv(inputs["train_target"], low_memory=False)
    if train_y_frame.shape[1] != 1 or train_y_frame.columns[0] != target:
        raise ValueError(f"train_target.csv должен содержать один столбец {target!r}")
    if len(train_x) != len(train_y_frame) or train_x.empty:
        raise ValueError("Число строк X/y не совпадает или train пуст")
    train_y = train_y_frame[target]
    if train_y.isna().any():
        raise ValueError("train_target.csv содержит пропуски")

    validation_x = None
    if mode == "evaluate":
        validation_x = _load_features(inputs["validation_features"], columns, categorical)
    test_x = _load_features(inputs["test_features"], columns, categorical)
    _capture_native_parameter_types()
    _, builder = _load_model(Path(manifest["code_path"]).expanduser().resolve())
    model = builder(context)
    if not callable(getattr(model, "fit", None)) or not callable(getattr(model, "predict", None)):
        raise TypeError("build_model(context) должен вернуть estimator с fit и predict")
    from .training import fit_model
    model, training_summary = fit_model(model, train_x, train_y.to_numpy(), context=context,
                options=manifest.get("fit_options"), iteration_override=manifest.get("iteration_override"))
    estimator_details = _estimator_details(model)
    if training_summary is not None:
        estimator_details['training'] = training_summary
        _merge_training_parameters(estimator_details, training_summary)

    artifact_dir = Path(manifest["artifact_dir"]).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if validation_x is not None:
        predictions = np.asarray(model.predict(validation_x)).reshape(-1)
        if len(predictions) != len(validation_x):
            raise ValueError("Число validation predictions не совпадает с validation_features")
        _validate_predictions(predictions, task_type, "validation")
        _atomic_csv(pd.DataFrame({target: predictions}), artifact_dir / "valid_predictions.csv")

    test_predictions = np.asarray(model.predict(test_x)).reshape(-1)
    if len(test_predictions) != len(test_x):
        raise ValueError("Число test predictions не совпадает с test_features")
    _validate_predictions(test_predictions, task_type, "test")
    _atomic_csv(pd.DataFrame({target: test_predictions}), artifact_dir / "test_predictions.csv")

    probability_written = False
    test_probability_written = False
    if task_type == "classification" and callable(getattr(model, "predict_proba", None)):
        probability_method = getattr(model, "predict_proba")
        if callable(probability_method):
            classes = getattr(model, "classes_", None)
            if classes is None:
                raise ValueError("Для вероятностей модель должна предоставить classes_")
            classes = list(classes)
            for stage, frame, filename in (("valid", validation_x, "valid_probabilities.npy"),
                                           ("test", test_x, "test_probabilities.npy")):
                if frame is None:
                    continue
                probabilities = np.asarray(probability_method(frame), dtype=float)
                if probabilities.ndim != 2 or probabilities.shape != (len(frame), len(classes)):
                    raise ValueError(f"predict_proba вернул матрицу неподходящей формы для {stage}")
                if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                        or (probabilities > 1).any()
                        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6)):
                    raise ValueError(f"predict_proba вернул невалидные вероятности для {stage}")
                _atomic_npy(probabilities, artifact_dir / filename)
                if stage == "valid":
                    probability_written = True
                else:
                    test_probability_written = True
            _atomic_json({"classes": [_json_value(item) for item in classes]}, artifact_dir / "probability_classes.json")
    elif task_type == "classification" and context.get("output_kind") in {"binary_probability", "class_probabilities"}:
        raise ValueError("Probability submission requires a classifier with predict_proba")

    model_path = None
    if bool(manifest.get("save_model", False)):
        model_path = artifact_dir / "model.joblib"
        fd, temporary = tempfile.mkstemp(prefix=".model.", suffix=".joblib", dir=artifact_dir)
        os.close(fd)
        try:
            # cloudpickle embeds dynamically loaded candidate classes/functions;
            # the resulting pickle remains readable through joblib.load().
            with open(temporary, "wb") as stream:
                cloudpickle.dump(model, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, model_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    result = {
        "status": "ok",
        "mode": mode,
        "train_rows": int(len(train_x)),
        "validation_rows": int(len(validation_x)) if validation_x is not None else None,
        "test_rows": int(len(test_x)),
        "target": target,
        "model_class": estimator_details["class"],
        "estimator_details": estimator_details,
        "valid_predictions": "valid_predictions.csv" if validation_x is not None else None,
        "test_predictions": "test_predictions.csv",
        "probabilities": "valid_probabilities.npy" if probability_written else None,
        "test_probabilities": "test_probabilities.npy" if test_probability_written else None,
        "model_path": str(model_path) if model_path else None,
    }
    _atomic_json(result, artifact_dir / "worker_result.json")
    return result


def _validate_predictions(predictions: np.ndarray, task_type: str, label: str) -> None:
    if pd.isna(predictions).any():
        raise ValueError(f"{label} predictions содержат пропуски")
    if task_type == "regression":
        try:
            numeric = predictions.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} regression predictions должны быть числовыми") from exc
        if not np.isfinite(numeric).all():
            raise ValueError(f"{label} regression predictions содержат NaN или бесконечность")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and run one AutoML candidate")
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        _run_manifest(args.manifest.expanduser().resolve())
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
