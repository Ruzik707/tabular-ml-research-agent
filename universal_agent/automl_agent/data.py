"""Generic CSV task discovery, compact profiling and reproducible splitting."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .contracts import TaskSpec


_CSV_RE = re.compile(r"[^\s`<>\"'()]*?\.csv", re.IGNORECASE)
_KIND_NAMES = {
    "train": {"train.csv", "training.csv"},
    "test": {"test.csv", "testing.csv"},
    "dictionary": {"data_dictionary.csv", "dictionary.csv", "schema.csv"},
    "sample": {"sample_submission.csv", "submission_sample.csv", "sample.csv"},
}
_METRIC_PATTERNS = (
    (r"macro[\s_-]*f1|f1[_\s-]*score\s*\([^)]*macro", "macro_f1"),
    (r"\brmsle\b|root[\s_-]*mean[\s_-]*squared[\s_-]*log(?:arithmic)?[\s_-]*error", "rmsle"),
    (r"root[\s_-]*mean[\s_-]*square[\s_-]*error|\brmse\b", "rmse"),
    (r"\bweighted[\s_-]*f1\b", "weighted_f1"),
    (r"\broc[\s_-]*auc\b|\bauc\b", "roc_auc"),
    (r"\blog[\s_-]*loss\b", "log_loss"),
    (r"\bbalanced[\s_-]*accuracy\b", "balanced_accuracy"),
    (r"\baccuracy\b|точност[ья]", "accuracy"),
    (r"\bmape\b|mean[\s_-]*absolute[\s_-]*percentage[\s_-]*error", "mape"),
    (r"\bmse\b|mean[\s_-]*squared[\s_-]*error", "mse"),
    (r"\bmae\b|mean[\s_-]*absolute[\s_-]*error", "mae"),
    (r"\br2\b|r\^2|коэффициент детерминации", "r2"),
    (r"\bf1\b", "f1"),
)
_EXPLICIT_METRIC_MARKERS = (
    re.compile(r"\b(?:primary|main)\b[^\n.;]*?\bmetric\b\s*(?:is\b|[:=—-])?", re.IGNORECASE),
    re.compile(r"основн\w*[^\n.;]*?метрик\w*\s*(?:это\b|[:=—-])?", re.IGNORECASE),
    re.compile(r"(?:^|[.;]\s*)(?:[-*#]\s*)?(?:evaluation\s+)?metric(?:\s+(?:for|of)\s+evaluation)?\s*(?:is\b|[:=—-])", re.IGNORECASE),
    re.compile(r"(?:^|[.;]\s*)(?:[-*#]\s*)?метрик\w*(?:\s+оценки)?\s*(?:is\b|[:=—-])", re.IGNORECASE),
)


def _absolute(path: Path) -> str:
    return str(path.expanduser().resolve())


def _discover_files(task_path: Path, text: str) -> dict[str, Path | None]:
    directory = task_path.parent
    references: dict[str, list[Path]] = {key: [] for key in _KIND_NAMES}
    for match in _CSV_RE.findall(text):
        token = match.strip(".,;:!?[]{}")
        name = Path(token.replace("\\", "/")).name.lower()
        kind = next((k for k, names in _KIND_NAMES.items() if name in names), None)
        if kind is None:
            stem = Path(name).stem
            kind = next((k for k in _KIND_NAMES if k in stem), None)
        if kind is None:
            continue
        raw = Path(token).expanduser()
        possible = raw if raw.is_absolute() else task_path.parent / raw
        if possible.is_file():
            references[kind].append(possible.resolve())
        else:
            # Some task descriptions retain their original nested paths while
            # the platform places the referenced CSVs beside task.md.
            sibling = task_path.parent / Path(token.replace("\\", "/")).name
            if sibling.is_file():
                references[kind].append(sibling.resolve())

    available = [p.resolve() for p in directory.glob("*.csv") if p.is_file()]
    result: dict[str, Path | None] = {}
    for kind, names in _KIND_NAMES.items():
        matches = list(dict.fromkeys(references[kind]))
        if not matches:
            matches = [p for p in available if p.name.lower() in names]
        if not matches:
            matches = [p for p in available if kind in p.stem.lower()]
        if len(matches) > 1:
            exact = [p for p in matches if p.name.lower() in names]
            if len(exact) == 1:
                matches = exact
            else:
                raise ValueError(f"Неоднозначный путь к CSV типа {kind}: {matches}")
        result[kind] = matches[0] if matches else None

    if result["train"] is None or result["test"] is None:
        raise ValueError(f"Рядом с {task_path} не найдены однозначные train.csv и test.csv")
    return result


def _read_dictionary(path: Path | None) -> tuple[dict[str, str], dict[str, str]]:
    if path is None:
        return {}, {}
    frame = pd.read_csv(path, low_memory=False)
    by_lower = {str(column).strip().lower(): column for column in frame.columns}
    feature_column = next(
        (by_lower[name] for name in ("feature", "column", "field", "name", "признак", "поле") if name in by_lower),
        frame.columns[0] if len(frame.columns) else None,
    )
    role_column = next(
        (by_lower[name] for name in ("role", "роль", "назначение", "type", "тип") if name in by_lower),
        None,
    )
    dtype_column = next(
        (by_lower[name] for name in ("dtype", "data_type", "тип данных") if name in by_lower),
        None,
    )
    if feature_column is None:
        return {}, {}
    roles: dict[str, str] = {}
    dtypes: dict[str, str] = {}
    for _, row in frame.iterrows():
        field = row.get(feature_column)
        if pd.isna(field):
            continue
        name = str(field).strip()
        if role_column is not None and not pd.isna(row.get(role_column)):
            roles[name] = str(row[role_column]).strip().lower()
        if dtype_column is not None and not pd.isna(row.get(dtype_column)):
            dtypes[name] = str(row[dtype_column]).strip().lower()
    return roles, dtypes


def _role_has(role: str, tokens: tuple[str, ...]) -> bool:
    normalized = re.sub(r"[^a-zа-я0-9]+", " ", role.lower())
    return any(token in normalized for token in tokens)


def _metric_from_text(text: str) -> str | None:
    lower = text.lower().replace("‑", "-")
    lines = lower.splitlines()

    def first_mentioned_metric(segment: str) -> str | None:
        matches = []
        for pattern, metric in _METRIC_PATTERNS:
            for match in re.finditer(pattern, segment):
                matches.append((match.start(), -(match.end() - match.start()), metric))
        return min(matches)[2] if matches else None

    # An explicitly identified primary/main/assessment metric outranks any
    # reference metric elsewhere in the same line or document. Inspect text
    # after the marker so e.g. "RMSE diagnostic; primary metric: MAE" works too.
    for line_index, line in enumerate(lines):
        for marker in _EXPLICIT_METRIC_MARKERS:
            match = marker.search(line)
            if not match:
                continue
            tail = line[match.end():]
            metric = first_mentioned_metric(tail)
            if metric:
                return metric
            # Support common Markdown where the metric value wraps to the next line.
            for continuation in lines[line_index + 1:line_index + 3]:
                continuation = continuation.strip()
                if not continuation:
                    continue
                if continuation.startswith(("#", "-", "*")):
                    break
                metric = first_mentioned_metric(continuation)
                if metric:
                    return metric
                break

    # Descriptions without an explicit primary marker retain a deterministic
    # generic fallback using the recognized metric family priority.
    for pattern, metric in _METRIC_PATTERNS:
        if re.search(pattern, lower):
            return metric
    return None


def _normalize_metric(value: str) -> str:
    normalized = re.sub(r"[\s-]+", "_", str(value).strip().lower())
    aliases = {
        "root_mean_squared_error": "rmse",
        "mean_absolute_error": "mae",
        "macro_f1_score": "macro_f1",
        "f1_macro": "macro_f1",
        "weighted_f1_score": "weighted_f1",
        "f1_weighted": "weighted_f1",
        "auc": "roc_auc",
        "logloss": "log_loss",
        "balanced_accuracy_score": "balanced_accuracy",
        "mean_squared_error": "mse",
        "mean_absolute_percentage_error": "mape",
        "root_mean_squared_log_error": "rmsle",
    }
    return aliases.get(normalized, normalized)


def _infer_task_type(
    text: str,
    target_values: pd.Series,
    metric: str | None,
    target_is_categorical: bool = False,
) -> str:
    if metric in {"macro_f1", "weighted_f1", "balanced_accuracy", "accuracy", "f1", "roc_auc", "log_loss"}:
        return "classification"
    if metric in {"rmse", "rmsle", "mse", "mae", "mape", "r2"}:
        return "regression"
    lower = text.lower()
    if re.search(r"классификац|classification|multiclass|multi-class", lower):
        return "classification"
    if re.search(r"регресси|regression", lower):
        return "regression"
    if target_is_categorical:
        return "classification"
    return "regression" if pd.api.types.is_numeric_dtype(target_values.dtype) else "classification"


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if pd.isna(value):
        return None
    return value


def _same_label(left: Any, right: Any) -> bool:
    if pd.isna(left) or pd.isna(right):
        return False
    try:
        return bool(left == right)
    except Exception:
        return str(left) == str(right)


def _positive_class(text: str, classes: list[Any]) -> Any:
    patterns = (
        r"(?:positive\s+(?:class|label)|положительн\w*\s+(?:класс|метк\w*|таргет))\s*(?:is\s+|это\s+|[:=—-]\s*)[\"'`]?([^\"'`;,\n]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            declared = match.group(1).strip().strip(" .!?").strip("*_ ")
            matches = [label for label in classes if str(label).strip() == declared]
            if not matches:
                try:
                    numeric = float(declared)
                    matches = [label for label in classes if isinstance(label, (int, float, np.number)) and float(label) == numeric]
                except ValueError:
                    pass
            if len(matches) != 1:
                raise ValueError(f"Указанный positive class {declared!r} не совпадает однозначно с классами: {classes}")
            return matches[0]
    if any(_same_label(label, 1) for label in classes):
        return next(label for label in classes if _same_label(label, 1))
    try:
        return sorted(classes)[-1]
    except TypeError:
        return sorted(classes, key=lambda value: str(value))[-1]


def _probability_column_mapping(columns: list[str], classes: list[Any]) -> dict[str, Any] | None:
    """Map each probability output header to one label, rejecting collisions."""
    prefixes = ("probability_", "probabilities_", "prob_", "class_")
    mapping: dict[str, Any] = {}
    used: list[Any] = []
    for column in columns:
        name = str(column).strip()
        candidate = name
        lowered = name.lower()
        for prefix in prefixes:
            if lowered.startswith(prefix):
                candidate = name[len(prefix):]
                break
        matches = [label for label in classes if str(label).strip() == candidate.strip()]
        if len(matches) != 1 or any(_same_label(matches[0], item) for item in used):
            return None
        mapping[column] = matches[0]
        used.append(matches[0])
    return mapping if len(mapping) == len(classes) else None


def _fingerprint(paths: list[Path | None], schema: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(schema, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    for path in paths:
        if path is None:
            continue
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def discover_task(
    task_path: Path,
    seed: int = 42,
    validation_fraction: float = 0.2,
    overrides: dict | None = None,
) -> TaskSpec:
    """Resolve task inputs and infer their roles without dataset-specific names."""
    task_path = Path(task_path).expanduser().resolve()
    if not task_path.is_file():
        raise FileNotFoundError(f"Не найден файл описания задачи: {task_path}")
    if not 0 < float(validation_fraction) < 1:
        raise ValueError("validation_fraction должна быть между 0 и 1")
    overrides = dict(overrides or {})
    unknown_overrides = set(overrides) - {"target", "task_type", "metric"}
    if unknown_overrides:
        raise ValueError(f"Неизвестные overrides: {sorted(unknown_overrides)}")
    text = task_path.read_text(encoding="utf-8", errors="replace")
    files = _discover_files(task_path, text)
    train_path = files["train"]
    test_path = files["test"]
    assert train_path is not None and test_path is not None
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, nrows=5, low_memory=False)
    dictionary_path = files["dictionary"]
    sample_path = files["sample"]
    roles, dictionary_dtypes = _read_dictionary(dictionary_path)
    sample = pd.read_csv(sample_path, nrows=5, low_memory=False) if sample_path else None

    target_candidates = [name for name, role in roles.items() if _role_has(role, ("target", "label", "цель", "таргет"))]
    if "target" in overrides:
        target = str(overrides["target"])
    elif len(target_candidates) == 1:
        target = target_candidates[0]
    elif len(target_candidates) > 1:
        raise ValueError(f"В словаре несколько таргетов: {target_candidates}; укажите overrides.target")
    else:
        train_only = [column for column in train.columns if column not in test.columns]
        sample_names = set(sample.columns).intersection(test.columns) if sample is not None else set()
        choices = [column for column in train_only if column not in sample_names]
        if len(choices) != 1:
            raise ValueError(f"Не удалось однозначно определить таргет. Кандидаты: {choices}")
        target = choices[0]
    if target not in train.columns:
        raise ValueError(f"Таргет {target!r} отсутствует в train.csv")
    if train[target].isna().any():
        raise ValueError(f"Таргет {target!r} содержит пропуски")

    sample_output = list(sample.columns) if sample is not None else []
    dictionary_ids = [
        name for name, role in roles.items()
        if _role_has(role, ("identifier", "key", "идентификатор", "ключ"))
        or re.search(r"(?:^|\s)id(?:$|\s)", re.sub(r"[^a-zа-я0-9]+", " ", role.lower())) is not None
    ]
    ids_from_sample = [name for name in sample_output if name != target and name in test.columns]
    if dictionary_ids:
        id_columns = [name for name in train.columns if name in dictionary_ids and name in test.columns]
        # A valid sample schema may identify an output-only key omitted from the dictionary.
        id_columns.extend(name for name in ids_from_sample if name not in id_columns)
    elif ids_from_sample:
        id_columns = ids_from_sample
    else:
        id_columns = [
            name for name in train.columns if name in test.columns
            and re.search(r"(?:^|[_\W])(id|key|index|identifier)(?:$|[_\W])", str(name), re.IGNORECASE)
        ]
    feature_columns = [name for name in train.columns if name != target and name not in id_columns]
    missing_in_test = [name for name in feature_columns if name not in test.columns]
    if missing_in_test:
        raise ValueError(f"В test.csv отсутствуют признаки из train.csv: {missing_in_test}")
    unexpected_test = [name for name in test.columns if name not in feature_columns and name not in id_columns]
    if unexpected_test:
        raise ValueError(f"Не удалось назначить роли столбцам test.csv: {unexpected_test}")
    explicit_metric = _normalize_metric(overrides["metric"]) if "metric" in overrides else None
    detected_metric = explicit_metric or _metric_from_text(text)
    explicit_type = str(overrides["task_type"]).strip().lower() if "task_type" in overrides else None
    if explicit_type and explicit_type not in {"regression", "classification"}:
        raise ValueError("overrides.task_type должен быть regression или classification")
    target_is_categorical = target in dictionary_dtypes and _role_has(
        dictionary_dtypes[target], ("categor", "category", "string", "object", "номин", "категор")
    )
    task_type = explicit_type or _infer_task_type(
        text, train[target], detected_metric, target_is_categorical=target_is_categorical
    )
    if explicit_metric is None:
        compatible = {
            "regression": {"rmse", "rmsle", "mse", "mae", "mape", "r2"},
            "classification": {"macro_f1", "weighted_f1", "balanced_accuracy", "accuracy", "f1", "roc_auc", "log_loss"},
        }
        if "task_type" in overrides and detected_metric not in compatible[task_type]:
            metric = "rmse" if task_type == "regression" else "macro_f1"
        else:
            metric = detected_metric or ("rmse" if task_type == "regression" else "macro_f1")
    else:
        metric = explicit_metric
    if not metric:
        raise ValueError("Метрика не может быть пустой")
    allowed_metrics = (
        {"rmse", "rmsle", "mse", "mae", "mape", "r2"} if task_type == "regression"
        else {"macro_f1", "weighted_f1", "balanced_accuracy", "accuracy", "f1", "roc_auc", "log_loss"}
    )
    if metric not in allowed_metrics:
        raise ValueError(f"Метрика {metric!r} несовместима с task_type={task_type!r}")
    maximize = metric not in {"rmse", "rmsle", "mse", "mae", "mape", "log_loss"}
    if metric == "rmsle" and (pd.to_numeric(train[target], errors="coerce") < 0).any():
        raise ValueError("RMSLE требует неотрицательный target")

    categorical_set = {
        name for name, role in roles.items() if _role_has(role, ("categor", "category", "string", "номин", "категор"))
    }
    categorical_set.update(
        name for name, dtype in dictionary_dtypes.items()
        if _role_has(dtype, ("categor", "category", "string", "object", "номин", "категор"))
    )
    for name in feature_columns:
        if name in categorical_set:
            continue
        if name not in dictionary_dtypes and not pd.api.types.is_numeric_dtype(train[name].dtype):
            categorical_set.add(name)
    categorical_columns = [name for name in feature_columns if name in categorical_set]
    classes = [_json_scalar(value) for value in pd.unique(train[target])] if task_type == "classification" else []
    positive_class = _positive_class(text, classes) if task_type == "classification" and len(classes) == 2 else None
    output_kind = "labels"
    probability_column_classes: dict[str, Any] = {}
    if sample is not None:
        non_id_outputs = [name for name in sample_output if name not in id_columns and name != target]
        if task_type == "classification" and len(classes) > 2 and target not in sample_output and non_id_outputs:
            mapping = _probability_column_mapping(non_id_outputs, classes)
            if mapping is None:
                raise ValueError(f"Неоднозначная multiclass probability схема sample_submission: колонки {non_id_outputs} нельзя однозначно сопоставить классам {classes}")
            output_kind = "class_probabilities"
            probability_column_classes = mapping
        elif task_type == "classification" and len(classes) > 2 and len(non_id_outputs) == len(classes):
            raise ValueError("Multiclass probability схема должна содержать только ID и вероятности, без label target")
        elif metric in {"roc_auc", "log_loss"} and len(classes) == 2:
            output_kind = "binary_probability"
            if target not in sample_output:
                raise ValueError(f"Бинарный probability output должен содержать target-колонку {target!r}: {sample_output}")
        elif metric in {"roc_auc", "log_loss"} and len(classes) > 2:
            raise ValueError("Для multiclass ROC-AUC/LogLoss sample_submission должен задавать отдельную probability-колонку на каждый класс")
        elif target not in sample_output:
            raise ValueError(f"Схема sample_submission не содержит label target {target!r}: {sample_output}")
        invalid_output = [name for name in sample_output if name not in id_columns and name != target and name not in probability_column_classes]
        if invalid_output:
            raise ValueError(f"Схема sample_submission не соответствует target/id: {sample_output}")
        output_columns = sample_output
    else:
        output_columns = [*id_columns, target]
        if metric in {"roc_auc", "log_loss"} and len(classes) == 2:
            output_kind = "binary_probability"
        elif metric in {"roc_auc", "log_loss"} and len(classes) > 2:
            raise ValueError("Для multiclass ROC-AUC/LogLoss требуется sample_submission с явной probability-схемой классов")
    schema = {"target": target, "features": feature_columns, "ids": id_columns, "outputs": output_columns}
    fingerprint = _fingerprint([task_path, train_path, test_path, dictionary_path, sample_path], schema)

    return TaskSpec(
        task_path=_absolute(task_path),
        train_path=_absolute(train_path),
        test_path=_absolute(test_path),
        target=target,
        task_type=task_type,
        metric=metric,
        maximize=maximize,
        feature_columns=feature_columns,
        id_columns=id_columns,
        categorical_columns=categorical_columns,
        output_columns=output_columns,
        data_fingerprint=fingerprint,
        description=text,
        dictionary_path=_absolute(dictionary_path) if dictionary_path else None,
        sample_path=_absolute(sample_path) if sample_path else None,
        classes=classes,
        output_kind=output_kind,
        probability_column_classes=probability_column_classes,
        positive_class=positive_class,
        seed=int(seed),
        validation_fraction=float(validation_fraction),
    )


def profile_task(task: TaskSpec) -> dict:
    """Return a bounded summary suitable for a model prompt and run report."""
    train = pd.read_csv(task.train_path, low_memory=False)
    test = pd.read_csv(task.test_path, low_memory=False)
    features: dict[str, dict[str, Any]] = {}
    for name in task.feature_columns:
        series = train[name]
        item: dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "missing_fraction": float(series.isna().mean()),
            "unique": int(series.nunique(dropna=True)),
        }
        if name in task.categorical_columns or not pd.api.types.is_numeric_dtype(series.dtype):
            counts = series.astype("string").fillna("<missing>").value_counts().head(8)
            item["top_values"] = [{"value": str(value), "count": int(count)} for value, count in counts.items()]
            item["test_unique"] = int(test[name].nunique(dropna=True)) if name in test else None
        else:
            numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if not numeric.empty:
                item["numeric"] = {
                    "min": float(numeric.min()),
                    "median": float(numeric.median()),
                    "mean": float(numeric.mean()),
                    "max": float(numeric.max()),
                }
                item["non_finite"] = int(np.isinf(pd.to_numeric(series, errors="coerce")).sum())
        features[name] = item

    target_series = train[task.target]
    target: dict[str, Any] = {
        "name": task.target,
        "dtype": str(target_series.dtype),
        "missing": int(target_series.isna().sum()),
        "unique": int(target_series.nunique(dropna=True)),
    }
    if task.task_type == "classification":
        counts = target_series.astype("string").value_counts(dropna=False).head(30)
        target["class_counts"] = {str(label): int(count) for label, count in counts.items()}
    else:
        values = pd.to_numeric(target_series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        target["numeric"] = {
            "min": float(values.min()), "median": float(values.median()),
            "mean": float(values.mean()), "max": float(values.max()),
        }
    return {
        "task_type": task.task_type,
        "metric": task.metric,
        "train_shape": [int(train.shape[0]), int(train.shape[1])],
        "test_shape": [int(test.shape[0]), int(test.shape[1])],
        "feature_count": len(task.feature_columns),
        "categorical_columns": list(task.categorical_columns),
        "features": features,
        "target": target,
    }


def prepare_split(task: TaskSpec, run_dir: Path) -> dict:
    """Create deterministic train/validation CSVs; keep IDs outside model inputs."""
    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(task.train_path, low_memory=False)
    test = pd.read_csv(task.test_path, low_memory=False)
    if task.target not in train:
        raise ValueError(f"В train.csv отсутствует target {task.target!r}")
    absent = [column for column in task.feature_columns if column not in train or column not in test]
    if absent:
        raise ValueError(f"Неполная схема признаков в train/test: {absent}")
    if train[task.target].isna().any():
        raise ValueError(f"Таргет {task.target!r} содержит пропуски")
    features = train.loc[:, task.feature_columns]
    target = train.loc[:, [task.target]]
    n_rows = len(train)
    if n_rows < 2:
        raise ValueError("Для split требуется минимум две строки обучающих данных")
    indices = np.arange(n_rows)
    stratify = None
    if task.task_type == "classification":
        counts = target[task.target].value_counts(dropna=False)
        n_valid = max(1, int(math.ceil(n_rows * task.validation_fraction)))
        n_fit = n_rows - n_valid
        if len(counts) > 1 and int(counts.min()) >= 2 and n_valid >= len(counts) and n_fit >= len(counts):
            stratify = target[task.target]
    fit_indices, valid_indices = train_test_split(
        indices,
        test_size=task.validation_fraction,
        random_state=task.seed,
        shuffle=True,
        stratify=stratify,
    )
    fit_indices = [int(value) for value in fit_indices]
    valid_indices = [int(value) for value in valid_indices]
    split_hash = hashlib.sha256(
        json.dumps(
            {"data": task.data_fingerprint, "seed": task.seed, "fraction": task.validation_fraction,
             "train": fit_indices, "validation": valid_indices},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    (run_dir / "split_indices.json").write_text(
        json.dumps({"train": fit_indices, "validation": valid_indices, "fingerprint": split_hash}, indent=2),
        encoding="utf-8",
    )

    frames = {
        "train_features": features.iloc[fit_indices],
        "train_target": target.iloc[fit_indices],
        "validation_features": features.iloc[valid_indices],
        "validation_target": target.iloc[valid_indices],
        "test_features": test.loc[:, task.feature_columns],
        "full_train_features": features,
        "full_train_target": target,
    }
    paths: dict[str, str] = {}
    for name, frame in frames.items():
        destination = run_dir / f"{name}.csv"
        frame.to_csv(destination, index=False, na_rep="")
        paths[name] = _absolute(destination)
    paths["split_fingerprint"] = split_hash
    return paths
