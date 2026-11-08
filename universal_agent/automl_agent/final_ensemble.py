"""Full-data refit for the selected candidate ensemble."""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd

from .prediction import PredictionEnsemble


def _class_labels(context: dict[str, Any], y: pd.Series) -> list[Any]:
    classes = context.get("classes")
    if type(classes) is not list or not classes:
        raise ValueError("classification context must define a non-empty global classes list")
    normalized = []
    for value in classes:
        if isinstance(value, np.generic):
            value = value.item()
        if type(value) not in (str, int, float, bool) or (type(value) is float and not math.isfinite(value)):
            raise ValueError("classes must contain only finite JSON scalar labels")
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise ValueError("context.classes must not contain duplicate labels")
    observed = set(y.tolist())
    if not observed.issubset(set(normalized)):
        raise ValueError("train_target contains labels outside context.classes")
    return normalized


def _validated_manifest(manifest: dict[str, Any]) -> tuple[dict, dict, list, np.ndarray, list | None, float | None, Any]:
    if type(manifest) is not dict:
        raise TypeError("manifest must be a dict")
    context, inputs = manifest.get("context"), manifest.get("inputs")
    if type(context) is not dict or type(inputs) is not dict:
        raise TypeError("manifest.context and manifest.inputs must be dicts")
    if set(inputs) != {"train_features", "train_target", "test_features"}:
        raise ValueError("inputs must contain only train_features, train_target, and test_features")
    if any(key in context for key in ("validation_target", "oof_target", "cv_labels")):
        raise ValueError("outer validation and OOF labels must not be passed to final refit")

    task_type, target = context.get("task_type"), context.get("target")
    columns, categorical = context.get("feature_columns"), context.get("categorical_columns", [])
    if task_type not in ("classification", "regression"):
        raise ValueError("context.task_type must be classification or regression")
    if type(target) is not str or not target:
        raise ValueError("context.target must be a non-empty string")
    if type(columns) is not list or not columns or any(type(name) is not str or not name for name in columns):
        raise ValueError("context.feature_columns must be a non-empty list of column names")
    if len(columns) != len(set(columns)) or target in columns:
        raise ValueError("feature columns must be unique and must not include the target")
    if type(categorical) is not list or any(name not in columns for name in categorical):
        raise ValueError("context.categorical_columns must be a list of known features")
    if len(categorical) != len(set(categorical)):
        raise ValueError("context.categorical_columns must not contain duplicates")

    members = manifest.get("members")
    if type(members) is not list or not 1 <= len(members) <= 3:
        raise ValueError("members must contain between one and three candidates")
    for member in members:
        if type(member) is not dict:
            raise TypeError("each member must be a dict")
        if type(member.get("code_path")) is not str or not member["code_path"]:
            raise ValueError("each member requires a code_path")
        if type(member.get("trial_id")) is not str or not member["trial_id"]:
            raise ValueError("each member requires a trial_id")
        if type(member.get("fit_options", {})) is not dict:
            raise TypeError("member.fit_options must be a dict")
        override = member.get("iteration_override")
        if override is not None and (type(override) is not int or not 1 <= override <= 100_000):
            raise ValueError("member.iteration_override must be a positive integer or null")

    raw_weights = manifest.get("weights")
    if type(raw_weights) is not list or len(raw_weights) != len(members):
        raise ValueError("weights length must match members")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in raw_weights):
        raise ValueError("weights must be positive finite numbers")
    if not math.isclose(sum(raw_weights), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("weights must sum to one")
    weights = np.asarray(raw_weights, dtype=float)

    if task_type == "classification":
        classes = context.get("classes")
        if type(classes) is not list or not classes:
            raise ValueError("classification context must define classes")
        raw_multipliers = manifest.get("class_multipliers")
        if raw_multipliers is not None:
            if type(raw_multipliers) is not list or len(raw_multipliers) != len(classes):
                raise ValueError("class_multipliers length must match context.classes")
            if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in raw_multipliers):
                raise ValueError("class_multipliers must be finite non-negative numbers")
            if not any(value > 0 for value in raw_multipliers):
                raise ValueError("class_multipliers must have at least one positive value")
            multipliers = [float(value) for value in raw_multipliers]
        else:
            multipliers = None
    else:
        if manifest.get("class_multipliers") is not None:
            raise ValueError("class_multipliers are only valid for classification")
        multipliers = None

    threshold = manifest.get("binary_threshold")
    positive_class = manifest.get("positive_class", context.get("positive_class"))
    if threshold is not None:
        if task_type != "classification" or len(context.get("classes", [])) != 2:
            raise ValueError("binary_threshold requires exactly two classification classes")
        if context.get("metric") not in (None, "f1"):
            raise ValueError("binary_threshold is only supported for binary f1")
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("binary_threshold must be a finite number in [0, 1]")
        if positive_class is None or positive_class not in context["classes"]:
            raise ValueError("binary_threshold requires a known positive_class")
        if context.get("positive_class") is not None and positive_class != context["positive_class"]:
            raise ValueError("binary threshold positive_class conflicts with context")
        if multipliers is not None:
            raise ValueError("binary_threshold cannot be combined with class_multipliers")
        threshold = float(threshold)

    for key in ("train_features", "train_target", "test_features"):
        if type(inputs[key]) is not str or not inputs[key]:
            raise ValueError(f"inputs.{key} must be a file path")
    if type(manifest.get("artifact_dir")) is not str or not manifest["artifact_dir"]:
        raise ValueError("artifact_dir must be a path")
    if type(manifest.get("save_model", False)) is not bool:
        raise TypeError("save_model must be a bool")
    return context, inputs, members, weights, multipliers, threshold, positive_class


def _atomic_npy(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_prediction_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write enough float digits for a CSV reload to round-trip predictions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False, na_rep="", float_format="%.17g")
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".model.", suffix=".joblib", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as stream:
            cloudpickle.dump(model, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_final_ensemble(manifest: dict[str, Any]) -> dict[str, Any]:
    """Refit selected members on all train rows and persist ensemble outputs.

    The manifest carries only outer-train data and test features. Member fits
    are performed in order, using the host-selected iteration override and the
    shared train-only fitting API.
    """
    context, inputs, members, weights, multipliers, threshold, positive_class = _validated_manifest(manifest)
    from .worker import (
        _atomic_json,
        _capture_native_parameter_types,
        _class_name,
        _estimator_details,
        _json_value,
        _load_features,
        _load_model,
        _validate_predictions,
    )
    from .training import fit_model

    feature_columns = context["feature_columns"]
    train_x = _load_features(inputs["train_features"], feature_columns, set(context.get("categorical_columns", [])))
    train_target_frame = pd.read_csv(inputs["train_target"], low_memory=False)
    target = context["target"]
    if list(train_target_frame.columns) != [target]:
        raise ValueError(f"train_target.csv must contain exactly one {target!r} column")
    if len(train_x) != len(train_target_frame) or train_x.empty:
        raise ValueError("train feature and target row counts must match and be non-zero")
    train_y = train_target_frame[target]
    if train_y.isna().any():
        raise ValueError("train_target.csv contains missing labels")
    if context["task_type"] == "classification":
        classes = _class_labels(context, train_y)
    else:
        try:
            train_y = pd.to_numeric(train_y, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("regression train_target must be numeric") from exc
        if not np.isfinite(train_y.to_numpy(dtype=float)).all():
            raise ValueError("regression train_target must contain finite values")
        classes = []
    test_x = _load_features(inputs["test_features"], feature_columns, set(context.get("categorical_columns", [])))

    # Capture exact trusted native types before importing any candidate code.
    _capture_native_parameter_types()
    fitted_members = []
    member_records = []
    for member in members:
        _, builder = _load_model(Path(member["code_path"]).expanduser().resolve())
        model = builder(context)
        if not callable(getattr(model, "fit", None)) or not callable(getattr(model, "predict", None)):
            raise TypeError("build_model(context) must return an estimator with fit and predict")
        fit_options = dict(member.get("fit_options", {}))
        # The final refit uses CV's chosen iteration count. Do not launch a new
        # inner probe on the full outer-train set while refitting the ensemble.
        fit_options["early_stopping_rounds"] = 0
        fitted, training = fit_model(
            model,
            train_x,
            train_y.to_numpy(),
            context=context,
            options=fit_options,
            iteration_override=member.get("iteration_override"),
        )
        details = _estimator_details(fitted)
        from .worker import _merge_training_parameters
        _merge_training_parameters(details, training)
        fitted_members.append(fitted)
        member_records.append({"trial_id": member["trial_id"], "details": details, "training": training})

    ensemble = PredictionEnsemble(
        fitted_members,
        context["task_type"],
        classes=classes,
        weights=weights.tolist(),
        class_multipliers=multipliers,
        binary_threshold=threshold,
        positive_class=positive_class,
    )
    prediction = np.asarray(ensemble.predict(test_x)).reshape(-1)
    if len(prediction) != len(test_x):
        raise ValueError("test prediction row count does not match test_features")
    _validate_predictions(prediction, context["task_type"], "test")
    if context["task_type"] == "classification":
        if pd.isna(prediction).any() or any(value not in classes for value in prediction.tolist()):
            raise ValueError("classification predictions must be present labels from context.classes")
        if not all(callable(getattr(model, "predict_proba", None)) for model in fitted_members):
            raise ValueError("final classification ensemble members must provide predict_proba")
        probabilities = np.asarray(ensemble.predict_proba(test_x), dtype=float)
        if probabilities.shape != (len(test_x), len(classes)) or not np.isfinite(probabilities).all():
            raise ValueError("test probabilities have an invalid shape or non-finite values")
        if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
            raise ValueError("test probabilities must be non-negative and sum to one")
    else:
        probabilities = None

    member_chains = []
    for record in member_records:
        for name in record["details"].get("estimator_chain", []):
            if name not in member_chains:
                member_chains.append(name)
    ensemble_name = _class_name(ensemble)
    estimator_details = {
        "class": ensemble_name,
        "estimator_chain": [ensemble_name, *member_chains],
        "pipeline_steps": [],
        "parameters": {
            "weights": weights.tolist(),
            "class_multipliers": multipliers,
            "binary_threshold": threshold,
            "positive_class": positive_class,
        },
        "parameters_source": "final_ensemble.manifest",
        "members": member_records,
        "training": {
            "train_rows": int(len(train_x)),
            "final_iterations": [record["training"].get("best_iteration") for record in member_records],
        },
    }

    artifact_dir = Path(manifest["artifact_dir"]).expanduser().resolve()
    test_predictions_name = "test_predictions.csv"
    _atomic_prediction_csv(pd.DataFrame({target: prediction}), artifact_dir / test_predictions_name)
    probabilities_name = None
    if probabilities is not None:
        probabilities_name = "test_probabilities.npy"
        _atomic_npy(probabilities, artifact_dir / probabilities_name)
        _atomic_json({"classes": [_json_value(value) for value in classes]}, artifact_dir / "probability_classes.json")

    save_model = manifest.get("save_model", False)
    model_name = None
    if save_model:
        model_name = "model.joblib"
        _atomic_model(ensemble, artifact_dir / model_name)

    result = {
        "status": "ok",
        "mode": "finalize_ensemble",
        "train_rows": int(len(train_x)),
        "test_rows": int(len(test_x)),
        "target": target,
        "model_class": ensemble_name,
        "estimator_details": estimator_details,
        "test_predictions": test_predictions_name,
        "probabilities": probabilities_name,
        "test_probabilities": probabilities_name,
        "probability_classes": "probability_classes.json" if probabilities is not None else None,
        "model_path": str(artifact_dir / model_name) if model_name else None,
    }
    _atomic_json(result, artifact_dir / "worker_result.json")
    return result
