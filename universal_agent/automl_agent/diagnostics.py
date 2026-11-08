"""Host-side validation diagnostics for candidate predictions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix as sklearn_confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)

from .evaluation import _as_float_array, _read_one_column


_MAX_CLASSES = 30
_MAX_SEGMENTS = 20
_MIN_SEGMENT_SUPPORT = 20


def _json_label(value: Any) -> str | int:
    """Normalize a supported class label to a JSON-safe Python scalar."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Validation labels must be finite")
        if value.is_integer():
            return int(value)
        return str(value)
    if isinstance(value, bool):
        return str(value)
    return str(value)


def _label_key(value: str | int) -> tuple[type, str | int]:
    return type(value), value


def validation_diagnostics(task: Any, split: dict, trial_dir: Path) -> dict:
    """Return bounded, host-side aggregate validation diagnostics."""
    task_type = getattr(task, "task_type", None)
    if task_type == "regression":
        if "validation_features" not in split or not getattr(task, "categorical_columns", []):
            return {}
        return _regression_segments(task, split, trial_dir)
    if task_type != "classification":
        return {}

    target = getattr(task, "target", None)
    if not isinstance(target, str) or not target:
        raise ValueError("Task target must be a non-empty column name")

    actual = _read_one_column(
        Path(split["validation_target"]), target, "validation_target.csv"
    )
    predicted = _read_one_column(
        Path(trial_dir) / "valid_predictions.csv", target, "valid_predictions.csv"
    )
    if len(actual) != len(predicted) or len(actual) == 0:
        raise ValueError("Размер validation predictions не совпадает с validation target")
    if actual.isna().any() or predicted.isna().any():
        raise ValueError("Validation labels и predictions не должны содержать пропуски")

    actual_labels = [_json_label(value) for value in actual.tolist()]
    predicted_labels = [_json_label(value) for value in predicted.tolist()]
    declared_labels = [
        _json_label(value)
        for value in (getattr(task, "classes", []) or [])
        if not pd.isna(value)
    ]

    # Keep declared class order, then append any validation or prediction label
    # not represented in the task metadata. This also reports zero-support labels.
    labels: list[str | int] = []
    label_indices: dict[tuple[type, str | int], int] = {}
    for label in [*declared_labels, *actual_labels, *predicted_labels]:
        key = _label_key(label)
        if key not in label_indices:
            label_indices[key] = len(labels)
            labels.append(label)

    if not labels:
        return {"per_class": [], "total_classes": 0, "partial": False}

    y_true = [label_indices[_label_key(label)] for label in actual_labels]
    y_pred = [label_indices[_label_key(label)] for label in predicted_labels]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        average=None,
        zero_division=0,
    )

    per_class = [
        {
            "label": labels[index],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in range(len(labels))
    ]
    per_class.sort(key=lambda item: item["f1"])

    total_classes = len(per_class)
    matrix_class_ids = sorted(range(len(labels)), key=lambda index: f1[index])[:_MAX_CLASSES]
    matrix = sklearn_confusion_matrix(y_true, y_pred, labels=matrix_class_ids)
    return {
        "per_class": per_class[:_MAX_CLASSES],
        "total_classes": total_classes,
        "partial": total_classes > _MAX_CLASSES,
        "confusion_matrix": {
            "labels": [labels[index] for index in matrix_class_ids],
            "matrix": matrix.astype(int).tolist(),
            "total_classes": total_classes,
            "partial": total_classes > _MAX_CLASSES,
            "selection": "lowest_f1_first",
        },
    }


def _regression_segments(task: Any, split: dict, trial_dir: Path) -> dict:
    """Summarize errors by declared categorical validation feature only."""
    target = getattr(task, "target", None)
    if not isinstance(target, str) or not target:
        raise ValueError("Task target must be a non-empty column name")
    actual = _read_one_column(
        Path(split["validation_target"]), target, "validation_target.csv"
    )
    predicted = _read_one_column(
        Path(trial_dir) / "valid_predictions.csv", target, "valid_predictions.csv"
    )
    if len(actual) != len(predicted) or len(actual) == 0:
        raise ValueError("Размер validation predictions не совпадает с validation target")
    if actual.isna().any() or predicted.isna().any():
        raise ValueError("Validation labels и predictions не должны содержать пропуски")
    y_true = _as_float_array(actual, "validation_target")
    y_pred = _as_float_array(predicted, "valid_predictions")

    features_path = Path(split["validation_features"])
    categorical_columns = list(getattr(task, "categorical_columns", []) or [])
    task_features = getattr(task, "feature_columns", None)
    if task_features:
        categorical_columns = [name for name in categorical_columns if name in task_features]
    if not categorical_columns:
        return {"segments": [], "total_segments": 0, "partial": False}
    if not features_path.is_file():
        raise ValueError(f"Не найден файл validation_features: {features_path}")
    features = pd.read_csv(features_path, low_memory=False)
    if any(column not in features.columns for column in categorical_columns):
        missing = [column for column in categorical_columns if column not in features.columns]
        raise ValueError(f"validation_features отсутствуют categorical columns: {missing}")
    if len(features) != len(actual):
        raise ValueError("Размер validation_features не совпадает с validation target")

    all_segments: list[dict[str, Any]] = []
    for column in categorical_columns:
        groups: dict[tuple[type, str | int | None], list[int]] = {}
        raw_labels: dict[tuple[type, str | int | None], str | int | None] = {}
        for index, raw_value in enumerate(features[column].tolist()):
            if pd.isna(raw_value):
                value: str | int | None = None
            else:
                value = _json_label(raw_value)
            key = (type(value), value)
            groups.setdefault(key, []).append(index)
            raw_labels[key] = value
        for key, indices in groups.items():
            count = len(indices)
            if count < _MIN_SEGMENT_SUPPORT:
                continue
            all_segments.append({
                "feature": column,
                "category": raw_labels[key],
                "count": count,
                "rmse": float(np.sqrt(mean_squared_error(y_true[indices], y_pred[indices]))),
                "mae": float(mean_absolute_error(y_true[indices], y_pred[indices])),
            })

    all_segments.sort(key=lambda item: (item["rmse"], item["mae"], item["count"]), reverse=True)
    total_segments = len(all_segments)
    return {
        "segments": all_segments[:_MAX_SEGMENTS],
        "total_segments": total_segments,
        "partial": total_segments > _MAX_SEGMENTS,
    }
