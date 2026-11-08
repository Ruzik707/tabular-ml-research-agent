"""Host-side scoring and independent, atomic submission validation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from .contracts import TaskSpec


def _read_one_column(path: Path, expected: str, label: str) -> pd.Series:
    if not Path(path).is_file():
        raise ValueError(f"Не найден файл {label}: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if list(frame.columns) != [expected]:
        raise ValueError(f"{label} должен содержать один столбец {expected!r}, получено: {list(frame.columns)}")
    return frame[expected]


def _metric_name(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"root_mean_squared_error": "rmse", "macro_f1_score": "macro_f1", "f1_macro": "macro_f1",
               "weighted_f1_score": "weighted_f1", "auc": "roc_auc",
               "balanced_accuracy_score": "balanced_accuracy", "mean_squared_error": "mse",
               "mean_absolute_percentage_error": "mape", "root_mean_squared_log_error": "rmsle"}
    return aliases.get(normalized, normalized)


def _as_float_array(values: pd.Series, label: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} содержит нечисловые значения, NaN или бесконечность")
    return numeric


def _classes(task: TaskSpec, actual: pd.Series, predicted: pd.Series) -> list[Any]:
    values = list(task.classes) if task.classes else list(pd.unique(pd.concat([actual, predicted], ignore_index=True)))
    values = [value for value in values if not pd.isna(value)]
    return values


def _same_label(left: Any, right: Any) -> bool:
    if pd.isna(left) or pd.isna(right):
        return False
    try:
        return bool(left == right)
    except Exception:
        return str(left) == str(right)


def _positive_class(labels: list[Any], declared: Any = None) -> Any:
    """Use sklearn's conventional label 1, or the last ordered class."""
    if declared is not None and any(_same_label(declared, label) for label in labels):
        return next(label for label in labels if _same_label(declared, label))
    for label in labels:
        if _same_label(label, 1):
            return label
    try:
        return sorted(labels)[-1]
    except TypeError:
        return labels[-1]


def _read_probability_artifacts(trial_dir: Path, stage: str = "valid") -> tuple[np.ndarray, list[Any]] | None:
    if stage not in {"valid", "test"}:
        raise ValueError("probability stage must be valid or test")
    probability_path = trial_dir / f"{stage}_probabilities.npy"
    classes_path = trial_dir / "probability_classes.json"
    if not probability_path.exists() and not classes_path.exists():
        return None
    if not probability_path.exists() or not classes_path.exists():
        raise ValueError("Вероятности должны сопровождаться probability_classes.json")
    try:
        probabilities = np.load(probability_path, allow_pickle=False)
        class_payload = json.loads(classes_path.read_text(encoding="utf-8"))
        classes = class_payload["classes"]
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать {stage} probabilities: {exc}") from exc
    if type(classes) is not list or not classes or len(classes) != len(set(map(str, classes))):
        raise ValueError("probability_classes.json must contain a non-empty unique classes list")
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("Размер вероятностей не совпадает со списком classes")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Вероятности должны быть конечными и находиться в диапазоне [0, 1]")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("Сумма вероятностей по каждой строке должна быть равна 1")
    return probabilities, classes


def score_trial(task: TaskSpec, split: dict, trial_dir: Path) -> dict[str, float]:
    """Compute metrics from host-owned labels and candidate prediction files."""
    trial_dir = Path(trial_dir)
    if split.get('mode') == 'evaluate_cv':
        try:
            worker_result = json.loads((trial_dir/'worker_result.json').read_text())
            indices = np.load(trial_dir/'oof_indices.npy', allow_pickle=False)
            plan = json.loads(Path(split['cv_plan']).read_text())
            fold_ids = np.load(trial_dir/'oof_fold_ids.npy', allow_pickle=False)
            expected_ids = np.full(len(pd.read_csv(split['train_target'])), -1)
            for fold in plan['folds']:
                expected_ids[fold['validation']] = fold['fold']
            aligned = (worker_result.get('split_fingerprint') == split['split_fingerprint']
                       and np.array_equal(indices, split['validation_indices'])
                       and np.array_equal(fold_ids, expected_ids[split['validation_indices']]))
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError('OOF alignment artifacts are missing or invalid') from exc
        if not aligned:
            raise ValueError('OOF indices, fold IDs or split fingerprint do not match the host plan')
    validation_path = Path(split["validation_target"])
    actual = _read_one_column(validation_path, task.target, "validation_target.csv")
    prediction_path = trial_dir / "valid_predictions.csv"
    predicted = _read_one_column(prediction_path, task.target, "valid_predictions.csv")
    if len(actual) != len(predicted) or len(actual) == 0:
        raise ValueError("Размер validation predictions не совпадает с validation target")
    if actual.isna().any() or predicted.isna().any():
        raise ValueError("Validation labels и predictions не должны содержать пропуски")
    primary = _metric_name(task.metric)
    scores: dict[str, float] = {}

    if task.task_type == "regression":
        y_true = _as_float_array(actual, "validation_target")
        y_pred = _as_float_array(predicted, "valid_predictions")
        candidates = {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        }
        if primary == "rmsle":
            if (y_true < 0).any():
                raise ValueError("RMSLE требует неотрицательный validation target")
            clipped = np.maximum(y_pred, 0.0)
            candidates["rmsle"] = float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(clipped))))
        if len(y_true) < 2:
            if primary == "r2":
                raise ValueError("Primary metric r2 is undefined with one validation row")
        else:
            candidates["r2"] = float(r2_score(y_true, y_pred))
        scores.update({name: value for name, value in candidates.items() if np.isfinite(value)})
    elif task.task_type == "classification":
        labels = _classes(task, actual, predicted)
        if primary == "f1" and len(labels) != 2:
            raise ValueError("Метрика f1 поддерживается только для бинарной задачи; для нескольких классов используйте macro_f1")
        unknown_predictions = [
            value for value in pd.unique(predicted)
            if not any(_same_label(value, label) for label in labels)
        ]
        if unknown_predictions:
            raise ValueError(f"Predictions содержат неизвестные классы: {unknown_predictions[:10]}")
        scores.update({
            "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(actual, predicted, labels=labels, average="weighted", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
            "accuracy": float(accuracy_score(actual, predicted)),
        })
        if len(labels) == 2:
            scores["f1"] = float(f1_score(
                actual, predicted, average="binary", pos_label=_positive_class(labels, getattr(task, "positive_class", None)), zero_division=0
            ))
        if primary in {"roc_auc", "log_loss"}:
            artifacts = _read_probability_artifacts(trial_dir)
            if artifacts is None:
                raise ValueError(f"Метрика {primary} требует valid_probabilities.npy и probability_classes.json")
            probabilities, probability_classes = artifacts
            if len(probabilities) != len(actual):
                raise ValueError("Число строк вероятностей не совпадает с validation target")
            if len(probability_classes) != len(labels) or any(
                not any(_same_label(label, found) for found in probability_classes) for label in labels
            ):
                raise ValueError("Классы вероятностей не совпадают с классами validation задачи")
            if any(not any(_same_label(found, label) for label in labels) for found in probability_classes):
                raise ValueError("Артефакты вероятностей содержат неизвестные классы")
            try:
                ordered_labels = sorted(labels)
            except TypeError:
                ordered_labels = sorted(labels, key=lambda value: str(value))
            aligned = np.column_stack([
                probabilities[:, next(i for i, found in enumerate(probability_classes) if _same_label(label, found))]
                                        for label in ordered_labels])
            if primary == "log_loss":
                scores["log_loss"] = float(log_loss(actual, aligned, labels=ordered_labels))
            elif len(ordered_labels) == 2:
                positive = _positive_class(ordered_labels, getattr(task, "positive_class", None))
                positive_index = next(i for i, label in enumerate(ordered_labels) if _same_label(label, positive))
                y_binary = np.asarray([_same_label(value, positive) for value in actual])
                scores["roc_auc"] = float(roc_auc_score(y_binary, aligned[:, positive_index]))
            else:
                scores["roc_auc"] = float(roc_auc_score(actual, aligned, labels=ordered_labels, multi_class="ovr", average="macro"))
    else:
        raise ValueError(f"Неподдерживаемый task_type: {task.task_type}")

    if primary not in scores:
        raise ValueError(f"Основная метрика {task.metric!r} не поддерживается или не определена для validation данных")
    scores[task.metric] = float(scores[primary])
    return scores


def _expected_output_columns(task: TaskSpec) -> list[str]:
    columns = list(task.output_columns or [*task.id_columns, task.target])
    output_kind = getattr(task, "output_kind", "labels")
    if output_kind not in {"labels", "binary_probability", "class_probabilities"}:
        raise ValueError(f"Неподдерживаемый output_kind: {output_kind!r}")
    if output_kind in {"labels", "binary_probability"} and columns.count(task.target) != 1:
        raise ValueError(f"target должен ровно один раз входить в output_columns: {columns}")
    if output_kind == "class_probabilities":
        mapping = getattr(task, "probability_column_classes", {}) or {}
        if columns.count(task.target) or set(mapping) != set(name for name in columns if name not in task.id_columns):
            raise ValueError("Probability output columns must map one-to-one to classes and omit the label target")
        mapped = list(mapping.values())
        if len(mapped) != len(task.classes) or any(
            not any(_same_label(label, value) for label in task.classes) for value in mapped
        ) or any(
            sum(_same_label(value, candidate) for candidate in mapped) != 1 for value in task.classes
        ):
            raise ValueError("Probability output columns do not map uniquely to all task classes")
    if len(columns) != len(set(columns)):
        raise ValueError(f"output_columns содержит повторяющиеся имена: {columns}")
    return columns


def _read_test(task: TaskSpec) -> pd.DataFrame:
    id_dtypes = {name: "string" for name in task.id_columns}
    test = pd.read_csv(task.test_path, low_memory=False, dtype=id_dtypes)
    if not task.id_columns:
        return test
    missing = [name for name in task.id_columns if name not in test]
    if missing:
        raise ValueError(f"В test.csv отсутствуют id-столбцы: {missing}")
    if test.loc[:, task.id_columns].isna().any().any():
        raise ValueError("ID в test.csv содержит пропуски")
    if test.duplicated(subset=task.id_columns).any():
        raise ValueError("ID в test.csv не уникален")
    return test


def _compare_ids(expected: pd.DataFrame, actual: pd.DataFrame, columns: list[str]) -> bool:
    for name in columns:
        left, right = expected[name], actual[name]
        if pd.api.types.is_numeric_dtype(left.dtype) and pd.api.types.is_numeric_dtype(right.dtype):
            if not np.array_equal(pd.to_numeric(left).to_numpy(), pd.to_numeric(right).to_numpy(), equal_nan=False):
                return False
        elif not left.astype("string").equals(right.astype("string")):
            return False
    return True


def _validate_frame(task: TaskSpec, frame: pd.DataFrame, test: pd.DataFrame) -> dict:
    expected_columns = _expected_output_columns(task)
    if list(frame.columns) != expected_columns:
        raise ValueError(f"Неверные колонки/порядок submission: ожидаются {expected_columns}, получены {list(frame.columns)}")
    if len(frame) != len(test):
        raise ValueError(f"Неверное число строк submission: ожидалось {len(test)}, получено {len(frame)}")
    id_columns = list(task.id_columns)
    if any(name not in test.columns for name in id_columns):
        raise ValueError("Output содержит столбец, отсутствующий в test.csv")
    if frame[expected_columns].isna().any().any():
        raise ValueError("Submission содержит пропуски")
    if id_columns and not _compare_ids(test.loc[:, id_columns], frame.loc[:, id_columns], id_columns):
        raise ValueError("ID в submission не совпадают с test.csv или изменён их порядок")

    output_kind = getattr(task, "output_kind", "labels")
    if output_kind == "binary_probability":
        if task.task_type != "classification" or len(task.classes) != 2:
            raise ValueError("binary_probability output requires a binary classification task")
        values = _as_float_array(frame[task.target], "submission probabilities")
        if (values < 0).any() or (values > 1).any():
            raise ValueError("Binary submission probabilities must be in [0, 1]")
    elif output_kind == "class_probabilities":
        if task.task_type != "classification":
            raise ValueError("class_probabilities output requires classification")
        columns = list((getattr(task, "probability_column_classes", {}) or {}).keys())
        probability_values = np.column_stack([_as_float_array(frame[column], f"probability column {column}") for column in columns])
        if (probability_values < 0).any() or (probability_values > 1).any():
            raise ValueError("Class submission probabilities must be in [0, 1]")
        if not np.allclose(probability_values.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
            raise ValueError("Class submission probabilities must sum to 1 for each row")
    elif task.task_type == "regression":
        values = _as_float_array(frame[task.target], "submission target")
        if _metric_name(task.metric) == "rmsle" and (values < 0).any():
            raise ValueError("RMSLE submission predictions must be clipped to non-negative values")
    elif task.task_type == "classification":
        values = frame[task.target]
        allowed = list(task.classes)
        if not allowed:
            train = pd.read_csv(task.train_path, usecols=[task.target], low_memory=False)[task.target]
            allowed = list(pd.unique(train.dropna()))
        unknown = [value for value in pd.unique(values) if not any(_same_label(value, label) for label in allowed)]
        if unknown:
            raise ValueError(f"Submission содержит неизвестные классы: {unknown[:10]}")
    else:
        raise ValueError(f"Неподдерживаемый task_type: {task.task_type}")

    if task.sample_path:
        sample = pd.read_csv(task.sample_path, nrows=0)
        if list(sample.columns) != expected_columns:
            raise ValueError("Колонки sample_submission не совпадают с output_columns")
    return {"valid": True, "rows": int(len(frame)), "columns": expected_columns, "target": task.target}


def _atomic_publish(frame: pd.DataFrame, destination: Path) -> None:
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False, na_rep="")
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Directory fsync is not available on every supported filesystem.
            pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publish_submission(task: TaskSpec, predictions_path: Path, destination: Path) -> dict:
    """Validate model predictions, attach output IDs and atomically publish."""
    predictions_path = Path(predictions_path)
    if not predictions_path.is_file():
        raise ValueError(f"Не найден файл predictions: {predictions_path}")
    id_dtypes = {name: "string" for name in task.id_columns}
    predictions = pd.read_csv(predictions_path, low_memory=False, dtype=id_dtypes)
    expected_columns = _expected_output_columns(task)
    test = _read_test(task)
    output_kind = getattr(task, "output_kind", "labels")
    if output_kind in {"binary_probability", "class_probabilities"}:
        if list(predictions.columns) != [task.target]:
            raise ValueError(f"Probability conversion requires label predictions [{task.target!r}] as row alignment")
        if len(predictions) != len(test):
            raise ValueError(f"Неверное число prediction строк: ожидалось {len(test)}, получено {len(predictions)}")
        artifacts = _read_probability_artifacts(predictions_path.parent, stage="test")
        if artifacts is None:
            raise ValueError("Probability submission requires test_probabilities.npy and probability_classes.json")
        matrix, artifact_classes = artifacts
        if len(matrix) != len(test):
            raise ValueError("Число строк test probabilities не совпадает с test.csv")
        if len(artifact_classes) != len(task.classes) or any(
            not any(_same_label(label, found) for found in artifact_classes) for label in task.classes
        ) or any(not any(_same_label(label, found) for found in task.classes) for label in artifact_classes):
            raise ValueError("test probability classes do not match TaskSpec.classes")
        aligned = np.column_stack([
            matrix[:, next(i for i, found in enumerate(artifact_classes) if _same_label(label, found))]
            for label in task.classes
        ])
        if not np.allclose(aligned.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
            raise ValueError("Aligned test probability rows must sum to 1")
        output = pd.DataFrame(index=range(len(test)))
        for name in expected_columns:
            if name in task.id_columns:
                output[name] = test[name].to_numpy()
            elif output_kind == "binary_probability" and name == task.target:
                positive = _positive_class(task.classes, getattr(task, "positive_class", None))
                index = next(i for i, label in enumerate(task.classes) if _same_label(label, positive))
                output[name] = aligned[:, index]
            elif output_kind == "class_probabilities":
                mapping = task.probability_column_classes
                label = mapping[name]
                index = next(i for i, candidate in enumerate(task.classes) if _same_label(candidate, label))
                output[name] = aligned[:, index]
            else:
                raise ValueError(f"Unexpected output column for {output_kind}: {name}")
    elif list(predictions.columns) == [task.target]:
        if len(predictions) != len(test):
            raise ValueError(f"Неверное число prediction строк: ожидалось {len(test)}, получено {len(predictions)}")
        output = pd.DataFrame(index=range(len(test)))
        for name in expected_columns:
            output[name] = predictions[task.target].to_numpy() if name == task.target else test[name].to_numpy()
    elif list(predictions.columns) == expected_columns:
        output = predictions
    else:
        raise ValueError(f"predictions должны содержать [{task.target!r}] или все колонки {expected_columns}")
    if task.task_type == "regression" and _metric_name(task.metric) == "rmsle":
        output[task.target] = np.maximum(_as_float_array(output[task.target], "test predictions"), 0.0)
    _validate_frame(task, output, test)
    _atomic_publish(output.loc[:, expected_columns], Path(destination))
    return {"valid": True, "rows": int(len(output)), "columns": expected_columns,
            "destination": str(Path(destination).expanduser().resolve()), "target": task.target}


def validate_submission(task: TaskSpec, path: Path) -> dict:
    """Independently check an already-written submission against the task files."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Не найден submission: {path}")
    id_dtypes = {name: "string" for name in task.id_columns}
    frame = pd.read_csv(path, low_memory=False, dtype=id_dtypes)
    return _validate_frame(task, frame, _read_test(task))
