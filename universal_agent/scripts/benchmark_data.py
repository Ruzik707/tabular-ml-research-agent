"""Build isolated outer-holdout tasks and score agent submissions independently."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


_REGRESSION_NAME = "Регрессия"
_MULTICLASS_NAME = "Многоклассовая классификация"
_INPUT_NAMES = (
    "train.csv",
    "test.csv",
    "sample_submission.csv",
    "data_dictionary.csv",
    "task.md",
)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_split(protocol: str, train_indices: np.ndarray, test_indices: np.ndarray, seed: int) -> str:
    payload = {
        "protocol": protocol,
        "seed": int(seed),
        "train_indices": [int(x) for x in train_indices],
        "test_indices": [int(x) for x in test_indices],
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dataset_dir(source_root: Path, dirname: str) -> Path:
    source_root = Path(source_root).expanduser().resolve()
    candidates = [source_root / dirname]
    if source_root.name == dirname:
        candidates.insert(0, source_root)
    for candidate in candidates:
        if (candidate / "train.csv").is_file():
            return candidate
    # Permit callers to provide an enclosing directory with an extra project level.
    matches = sorted(path.parent for path in source_root.glob(f"**/{dirname}/train.csv"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Несколько каталогов источника {dirname!r} под {source_root}")
    raise FileNotFoundError(f"Не найден {dirname}/train.csv под {source_root}")


def _read_source_train(path: Path, id_columns: list[str]) -> pd.DataFrame:
    # IDs are strings throughout the evaluator so that leading zeroes are meaningful.
    return pd.read_csv(path, low_memory=False, dtype={name: "string" for name in id_columns})


def _dictionary_for_frame(
    source_dictionary: Path | None,
    columns: list[str],
    id_columns: list[str],
    target: str,
    target_description: str,
    source_frame: pd.DataFrame,
) -> pd.DataFrame:
    if source_dictionary is not None and source_dictionary.is_file():
        dictionary = pd.read_csv(source_dictionary, low_memory=False)
    else:
        dictionary = pd.DataFrame(columns=["feature", "role", "dtype", "description_ru"])
    if "feature" not in dictionary.columns:
        dictionary = pd.DataFrame(columns=["feature", "role", "dtype", "description_ru"])
    for column in ("role", "dtype", "description_ru"):
        if column not in dictionary.columns:
            dictionary[column] = ""

    indexed = {str(row["feature"]): idx for idx, row in dictionary.iterrows() if pd.notna(row["feature"])}
    for name in columns:
        role = "target" if name == target else "id" if name in id_columns else "feature"
        dtype = "category" if name == target and not pd.api.types.is_numeric_dtype(source_frame[name]) else (
            "numeric" if pd.api.types.is_numeric_dtype(source_frame[name]) else "category"
        )
        description = target_description if name == target else ("Идентификатор строки" if name in id_columns else "Признак")
        if name in indexed:
            idx = indexed[name]
            dictionary.loc[idx, "role"] = role
            if name == target:
                dictionary.loc[idx, "dtype"] = dtype
                dictionary.loc[idx, "description_ru"] = description
        else:
            row = {column: "" for column in dictionary.columns}
            row.update({"feature": name, "role": role, "dtype": dtype, "description_ru": description})
            dictionary = pd.concat([dictionary, pd.DataFrame([row])], ignore_index=True)
    return dictionary.loc[dictionary["feature"].astype(str).isin(columns)].reset_index(drop=True)


def _write_case(
    *,
    work_dir: Path,
    name: str,
    train: pd.DataFrame,
    test_features: pd.DataFrame,
    truth: pd.DataFrame,
    target: str,
    task_type: str,
    metric: str,
    maximize: bool,
    positive_class: Any,
    classes: list[Any],
    id_columns: list[str],
    source_dictionary: Path | None,
    split_protocol: str,
    split_fingerprint: str,
) -> dict:
    task_dir = work_dir / "inputs" / name
    evaluator_dir = work_dir / "evaluator" / name
    if task_dir.exists():
        shutil.rmtree(task_dir)
    if evaluator_dir.exists():
        shutil.rmtree(evaluator_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    evaluator_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = [column for column in train.columns if column not in {*id_columns, target}]
    test_columns = [*id_columns, *feature_columns]
    train.to_csv(task_dir / "train.csv", index=False)
    test_features.loc[:, test_columns].to_csv(task_dir / "test.csv", index=False)
    truth.loc[:, [*id_columns, target]].to_csv(evaluator_dir / "truth.csv", index=False)

    placeholder: Any = 0.0 if task_type == "regression" else (classes[0] if classes else "")
    sample = test_features.loc[:, id_columns].copy()
    sample[target] = placeholder
    sample.to_csv(task_dir / "sample_submission.csv", index=False)

    target_description = f"Целевая переменная; основной критерий {metric}."
    if positive_class is not None:
        target_description += f" Положительный класс: {positive_class}."
    dictionary = _dictionary_for_frame(
        source_dictionary, list(train.columns), id_columns, target, target_description, train
    )
    dictionary.to_csv(task_dir / "data_dictionary.csv", index=False)

    task_lines = [
        f"# External heldout benchmark: {name}",
        "",
        f"Тип задачи / Task type: {task_type}",
        f"Целевая переменная / Target: {target}",
        f"target={target}",
        f"Основная метрика / Primary metric: {metric.upper()}",
        f"primary_metric={metric}",
        f"Строк train / Train rows: {len(train)}",
        f"Строк test / Test rows: {len(test_features)}",
        f"Столбцов признаков / Feature columns: {len(feature_columns)}",
        "Входы: train.csv, test.csv, data_dictionary.csv, sample_submission.csv рядом с task.md.",
        "Сохраните submission.csv рядом с task.md по схеме sample_submission.csv; порядок и значения ID сохраняются.",
        "Используйте только предоставленный train для обучения; ID не является предиктивным признаком.",
        f"Протокол разбиения / Split protocol: {split_protocol}",
        "Файл test.csv — фиксированная внешняя контрольная выборка; её метки target отсутствуют во входном каталоге.",
        "The provided test.csv is the fixed external holdout. Its target labels are not included in this task directory.",
    ]
    if positive_class is not None:
        task_lines.extend([
            f"Положительный класс / Positive class: {positive_class}",
            f"positive_class={positive_class}",
        ])
    (task_dir / "task.md").write_text("\n".join(task_lines) + "\n", encoding="utf-8")

    input_hashes = {filename: _sha256_file(task_dir / filename) for filename in _INPUT_NAMES}
    return {
        "name": name,
        "target": target,
        "task_type": task_type,
        "metric": metric,
        "maximize": bool(maximize),
        "positive_class": _json_scalar(positive_class) if positive_class is not None else None,
        "train_rows": int(len(train)),
        "test_rows": int(len(test_features)),
        "feature_columns": feature_columns,
        "id_columns": id_columns,
        "classes": [_json_scalar(value) for value in classes],
        "split_protocol": split_protocol,
        "task_dir": str(task_dir.resolve()),
        "truth_path": str((evaluator_dir / "truth.csv").resolve()),
        "input_hashes": input_hashes,
        "split_fingerprint": split_fingerprint,
    }


def _prepare_public_regression(source_root: Path, work_dir: Path, seed: int) -> dict:
    dataset_dir = _dataset_dir(source_root, _REGRESSION_NAME)
    target = "site_eui"
    columns = pd.read_csv(dataset_dir / "train.csv", nrows=0).columns
    ids = ["id"] if "id" in columns else []
    source = _read_source_train(dataset_dir / "train.csv", ids)
    required = {target, "Year_Factor"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"В regression train.csv отсутствуют обязательные колонки: {sorted(missing)}")
    if not ids:
        raise ValueError("Публичная regression задача должна содержать id")
    latest_period = source["Year_Factor"].max()
    test_mask = source["Year_Factor"].eq(latest_period)
    train_indices = np.flatnonzero(~test_mask.to_numpy())
    test_indices = np.flatnonzero(test_mask.to_numpy())
    if not len(train_indices) or not len(test_indices):
        raise ValueError("Не удалось отделить последний Year_Factor как внешний holdout")
    train = source.iloc[train_indices].copy()
    test_rows = source.iloc[test_indices].copy()
    features = [column for column in source.columns if column not in {*ids, target}]
    protocol = f"temporal Year_Factor holdout: {latest_period}; all previous periods train"
    fingerprint = _fingerprint_split(protocol, train_indices, test_indices, seed)
    return _write_case(
        work_dir=work_dir, name="public-regression", train=train,
        test_features=test_rows.loc[:, [*ids, *features]], truth=test_rows.loc[:, [*ids, target]],
        target=target, task_type="regression", metric="rmse", maximize=False, positive_class=None,
        classes=[], id_columns=ids, source_dictionary=dataset_dir / "data_dictionary.csv",
        split_protocol=protocol, split_fingerprint=fingerprint,
    )


def _prepare_public_multiclass(source_root: Path, work_dir: Path, seed: int) -> dict:
    dataset_dir = _dataset_dir(source_root, _MULTICLASS_NAME)
    raw = pd.read_csv(dataset_dir / "train.csv", low_memory=False, dtype={"id": "string"})
    target = "attack_cat"
    ids = ["id"] if "id" in raw.columns else []
    if target not in raw.columns or not ids:
        raise ValueError("Публичная multiclass задача должна содержать attack_cat и id")
    if raw[target].isna().any():
        raise ValueError("attack_cat содержит пропуски; benchmark требует размеченные классы")
    features = [column for column in raw.columns if column not in {*ids, target}]
    if not features:
        raise ValueError("Не найдены multiclass признаки")
    y = raw[target]
    classes = sorted(_json_scalar(value) for value in pd.unique(y))
    groups = pd.util.hash_pandas_object(raw.loc[:, features], index=False, categorize=True).to_numpy(dtype="uint64")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=int(seed))
    try:
        train_indices, test_indices = next(splitter.split(raw.loc[:, features], y, groups))
    except (ValueError, StopIteration) as exc:
        raise ValueError(f"Не удалось построить первый stratified group fold: {exc}") from exc
    train_classes = set(pd.unique(y.iloc[train_indices]))
    test_classes = set(pd.unique(y.iloc[test_indices]))
    expected_classes = set(classes)
    if train_classes != expected_classes or test_classes != expected_classes:
        raise ValueError(
            "Первый seed-fixed StratifiedGroupKFold не содержит все классы в train и holdout; "
            f"train={sorted(train_classes)}, holdout={sorted(test_classes)}, expected={classes}"
        )
    if set(groups[train_indices]).intersection(groups[test_indices]):
        raise RuntimeError("Одинаковые feature groups пересекаются между train и внешним holdout")
    train = raw.iloc[train_indices].copy()
    test_rows = raw.iloc[test_indices].copy()
    protocol = f"first StratifiedGroupKFold(n_splits=5, shuffle=True, random_state={seed}); groups=exact feature rows excluding id/target"
    fingerprint = _fingerprint_split(protocol, train_indices, test_indices, seed)
    return _write_case(
        work_dir=work_dir, name="public-multiclass", train=train,
        test_features=test_rows.loc[:, [*ids, *features]], truth=test_rows.loc[:, [*ids, target]],
        target=target, task_type="classification", metric="macro_f1", maximize=True, positive_class=None,
        classes=classes, id_columns=ids, source_dictionary=dataset_dir / "data_dictionary.csv",
        split_protocol=protocol, split_fingerprint=fingerprint,
    )


def _prepare_synthetic_regression(work_dir: Path, seed: int) -> dict:
    x, y, _ = make_regression(
        n_samples=800, n_features=12, n_informative=8, noise=20.0,
        coef=True, random_state=int(seed),
    )
    ids = np.asarray([f"lin-{index:04d}" for index in range(len(y))], dtype=object)
    feature_names = [f"x{index}" for index in range(x.shape[1])]
    frame = pd.DataFrame(x, columns=feature_names)
    frame.insert(0, "row_id", ids)
    frame["target"] = y
    train_indices, test_indices = train_test_split(
        np.arange(len(frame)), train_size=600, test_size=200, random_state=int(seed), shuffle=True
    )
    protocol = f"synthetic make_regression(n_samples=800); train_test_split(600/200, random_state={seed})"
    return _write_case(
        work_dir=work_dir, name="synthetic-signed-linear-regression",
        train=frame.iloc[train_indices].copy(),
        test_features=frame.iloc[test_indices].loc[:, ["row_id", *feature_names]].copy(),
        truth=frame.iloc[test_indices].loc[:, ["row_id", "target"]].copy(),
        target="target", task_type="regression", metric="rmse", maximize=False,
        positive_class=None, classes=[], id_columns=["row_id"], source_dictionary=None,
        split_protocol=protocol, split_fingerprint=_fingerprint_split(protocol, train_indices, test_indices, seed),
    )


def _prepare_synthetic_binary(work_dir: Path, seed: int) -> dict:
    x, y = make_classification(
        n_samples=1000, n_features=20, n_informative=8, n_redundant=2,
        n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=[0.9, 0.1],
        flip_y=0.01, class_sep=1.0, random_state=int(seed),
    )
    ids = np.asarray([f"bin-{index:04d}" for index in range(len(y))], dtype=object)
    feature_names = [f"x{index}" for index in range(x.shape[1])]
    frame = pd.DataFrame(x, columns=feature_names)
    frame.insert(0, "row_id", ids)
    # Explicit positive-first naming is independent of sklearn's encoded class order.
    frame["label"] = np.where(y == 1, "alert", "normal")
    train_indices, test_indices = train_test_split(
        np.arange(len(frame)), train_size=800, test_size=200, random_state=int(seed),
        shuffle=True, stratify=frame["label"],
    )
    protocol = f"synthetic make_classification(weights=90/10); stratified train_test_split(800/200, random_state={seed})"
    return _write_case(
        work_dir=work_dir, name="synthetic-imbalanced-binary-f1",
        train=frame.iloc[train_indices].copy(),
        test_features=frame.iloc[test_indices].loc[:, ["row_id", *feature_names]].copy(),
        truth=frame.iloc[test_indices].loc[:, ["row_id", "label"]].copy(),
        target="label", task_type="classification", metric="f1", maximize=True,
        positive_class="alert", classes=["alert", "normal"], id_columns=["row_id"],
        source_dictionary=None, split_protocol=protocol,
        split_fingerprint=_fingerprint_split(protocol, train_indices, test_indices, seed),
    )


def prepare_cases(source_data_root: Path, work_dir: Path, seed: int = 42) -> list[dict]:
    """Create four fixed cases, keeping each outer holdout's labels out of agent inputs."""
    source_root = Path(source_data_root).expanduser().resolve()
    destination = Path(work_dir).expanduser().resolve()
    if destination == source_root or source_root in destination.parents:
        raise ValueError("work_dir должен находиться вне каталога исходных данных")
    destination.mkdir(parents=True, exist_ok=True)
    return [
        _prepare_public_regression(source_root, destination, int(seed)),
        _prepare_public_multiclass(source_root, destination, int(seed)),
        _prepare_synthetic_regression(destination, int(seed)),
        _prepare_synthetic_binary(destination, int(seed)),
    ]


def _invalid(error: str, rows: int = 0) -> dict:
    return {"valid": False, "score": None, "metrics": {}, "error": error, "rows": int(rows)}


def _strict_label_match(value: Any, expected: Any) -> bool:
    value = _json_scalar(value)
    expected = _json_scalar(expected)
    if isinstance(value, (bool, np.bool_)) or isinstance(expected, (bool, np.bool_)):
        return isinstance(value, (bool, np.bool_)) and isinstance(expected, (bool, np.bool_)) and bool(value) == bool(expected)
    if isinstance(value, (int, float, np.integer, np.floating)) and isinstance(expected, (int, float, np.integer, np.floating)):
        return float(value) == float(expected)
    return type(value) is type(expected) and value == expected


def score_submission(case: dict, submission: Path) -> dict:
    """Validate a final CSV against evaluator-owned IDs/truth and compute its primary metric."""
    submission = Path(submission)
    rows = 0
    try:
        target = str(case["target"])
        id_columns = list(case["id_columns"])
        if not id_columns:
            return _invalid("id contract must define at least one ID column")
        truth_path = Path(case["truth_path"])
        if not truth_path.is_file():
            return _invalid(f"missing evaluator truth file: {truth_path}")
        classes = list(case.get("classes") or [])
        truth_dtype = "string" if classes and all(isinstance(label, str) for label in classes) else None
        truth_dtypes = {column: "string" for column in id_columns}
        if truth_dtype:
            truth_dtypes[target] = "string"
        truth = pd.read_csv(truth_path, low_memory=False, dtype=truth_dtypes)
        expected_columns = [*id_columns, target]
        if truth.columns.tolist() != expected_columns:
            return _invalid(f"evaluator truth columns must be exactly {expected_columns}, got {truth.columns.tolist()}")
        if not submission.is_file():
            return _invalid(f"submission CSV does not exist: {submission}")
        submission_dtypes = {column: "string" for column in id_columns}
        if truth_dtype:
            submission_dtypes[target] = "string"
        predicted = pd.read_csv(submission, low_memory=False, dtype=submission_dtypes)
        rows = len(predicted)
        if predicted.columns.tolist() != expected_columns:
            return _invalid(f"submission columns/order must be exactly {expected_columns}, got {predicted.columns.tolist()}", rows)
        expected_rows = int(case.get("test_rows", len(truth)))
        if len(predicted) != expected_rows or len(truth) != expected_rows:
            return _invalid(
                f"row count mismatch: submission={len(predicted)}, truth={len(truth)}, expected={expected_rows}", rows
            )
        if predicted[id_columns].isna().any().any() or truth[id_columns].isna().any().any():
            return _invalid("ID values must not be missing", rows)
        expected_ids = truth[id_columns].astype("string").to_numpy(dtype=object)
        actual_ids = predicted[id_columns].astype("string").to_numpy(dtype=object)
        if not np.array_equal(actual_ids, expected_ids):
            mismatch = np.argwhere(actual_ids != expected_ids)
            row = int(mismatch[0][0]) if mismatch.size else -1
            return _invalid(f"ID/order mismatch at row {row}; submission IDs must match evaluator keys exactly in order", rows)
        if predicted[target].isna().any() or truth[target].isna().any():
            return _invalid("target values must not be missing", rows)

        task_type = str(case["task_type"])
        metric = str(case["metric"]).strip().lower()
        if task_type == "regression":
            if metric != "rmse":
                return _invalid(f"unsupported regression metric: {metric}", rows)
            y_true = pd.to_numeric(truth[target], errors="coerce").to_numpy(dtype=float)
            y_pred = pd.to_numeric(predicted[target], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
                return _invalid("regression target and predictions must be finite numeric values", rows)
            score = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics = {"rmse": score, "mae": float(mean_absolute_error(y_true, y_pred)),
                       "absolute_error_quantiles": np.quantile(np.abs(y_true-y_pred), [.5, .9, .99]).tolist()}
        elif task_type == "classification":
            if not classes:
                return _invalid("classification case must declare classes", rows)
            y_true = truth[target].tolist()
            y_pred = predicted[target].tolist()
            for label in y_true + y_pred:
                if not any(_strict_label_match(label, allowed) for allowed in classes):
                    return _invalid(f"unknown class label: {label!r}", rows)
            if metric == "f1":
                if len(classes) != 2:
                    return _invalid("binary F1 requires exactly two declared classes", rows)
                positive = case.get("positive_class")
                if positive is None or not any(_strict_label_match(positive, label) for label in classes):
                    return _invalid("binary F1 requires an explicit positive_class from classes", rows)
                score = float(f1_score(y_true, y_pred, pos_label=positive, average="binary", zero_division=0))
                metrics = {"f1": score}
            elif metric == "macro_f1":
                score = float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
                metrics = {"macro_f1": score}
            else:
                return _invalid(f"unsupported classification metric: {metric}", rows)
            precision, recall, per_f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=classes, zero_division=0)
            metrics.update(accuracy=float(accuracy_score(y_true, y_pred)),
                           macro_f1=float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
                           weighted_f1=float(f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)),
                           per_class={str(label): {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                                      for label, p, r, f, s in zip(classes, precision, recall, per_f1, support, strict=True)})
        else:
            return _invalid(f"unsupported task type: {task_type}", rows)
        return {"valid": True, "score": score, "metrics": metrics, "error": None, "rows": rows}
    except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
        return _invalid(f"unable to validate submission: {exc}", rows)
