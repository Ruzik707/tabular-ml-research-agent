"""Frozen, deterministic non-LLM references for the external agent benchmark.

The worker subprocess receives only task inputs and case metadata. Evaluator truth
is read by the parent process only after a submission has been written, through
``benchmark_data.score_submission``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Frozen before the timed agent runs. Do not tune or edit these specs based on
# outer-holdout scores; their purpose is a stable lower-bound reference.
REFERENCE_SPEC: dict[str, dict[str, dict[str, Any]]] = {
    "dummy": {
        "regression": {"estimator": "DummyRegressor", "strategy": "mean"},
        "classification": {"estimator": "DummyClassifier", "strategy": "most_frequent"},
    },
    "linear": {
        "regression": {
            "estimator": "Pipeline(StandardScaler+Ridge)",
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "one_hot_handle_unknown": "ignore",
            "ridge_alpha": 1.0,
        },
        "classification": {
            "estimator": "Pipeline(StandardScaler+LogisticRegression)",
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "one_hot_handle_unknown": "ignore",
            "logistic_C": 1.0,
            "max_iter": 1000,
            "random_state": 42,
            "solver": "lbfgs",
            "class_weight": None,
            "n_jobs_from_cli_threads": True,
        },
    },
}

_INPUT_NAMES = (
    "train.csv",
    "test.csv",
    "sample_submission.csv",
    "data_dictionary.csv",
    "task.md",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temp_path = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        stream.write("\n")
    os.replace(temp_path, path)


def _source_git_sha() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=3,
        )
        return completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _case_timeout_seconds(case: dict) -> int:
    return 60 if str(case.get("name", "")).startswith("synthetic-") else 180


def _verify_case_inputs(case: dict) -> tuple[bool, dict[str, str], str | None]:
    task_dir = Path(case["task_dir"])
    actual: dict[str, str] = {}
    expected = case.get("input_hashes") or {}
    if not expected:
        return False, actual, "case metadata has no input_hashes"
    for filename in _INPUT_NAMES:
        path = task_dir / filename
        if not path.is_file():
            return False, actual, f"missing task input {filename}"
        actual[filename] = _sha256_file(path)
        if filename not in expected or actual[filename] != expected[filename]:
            return False, actual, f"input hash mismatch for {filename}"
    return True, actual, None


def _model_spec(model_name: str, task_type: str) -> dict[str, Any]:
    try:
        return dict(REFERENCE_SPEC[model_name][task_type])
    except KeyError as exc:
        raise ValueError(f"unsupported reference model/task: {model_name}/{task_type}") from exc


def _make_estimator(model_name: str, task_type: str, threads: int):
    if model_name == "dummy":
        if task_type == "regression":
            return DummyRegressor(strategy="mean")
        if task_type == "classification":
            return DummyClassifier(strategy="most_frequent")
        raise ValueError(f"unsupported task_type: {task_type}")

    if model_name != "linear":
        raise ValueError(f"unknown reference estimator: {model_name}")
    if task_type not in {"regression", "classification"}:
        raise ValueError(f"unsupported task_type: {task_type}")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, "numeric"),
            ("categorical", categorical_pipeline, "categorical"),
        ],
        remainder="drop",
    )
    if task_type == "regression":
        estimator = Ridge(alpha=1.0)
    else:
        estimator = LogisticRegression(
            C=1.0,
            max_iter=1000,
            n_jobs=int(threads),
            random_state=42,
            solver="lbfgs",
            class_weight=None,
        )
    return Pipeline([("preprocess", preprocessor), ("estimator", estimator)])


def _load_task_data(case: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    task_dir = Path(case["task_dir"])
    target = str(case["target"])
    ids = list(case["id_columns"])
    features = list(case["feature_columns"])
    if not ids or not features:
        raise ValueError("reference requires non-empty id_columns and feature_columns")
    declared_classes = list(case.get("classes") or [])
    read_as_strings = bool(declared_classes) and all(isinstance(label, str) for label in declared_classes)
    train_dtypes = {column: "string" for column in ids}
    test_dtypes = {column: "string" for column in ids}
    sample_dtypes = {column: "string" for column in ids}
    if read_as_strings:
        train_dtypes[target] = "string"

    train = pd.read_csv(task_dir / "train.csv", low_memory=False, dtype=train_dtypes)
    test = pd.read_csv(task_dir / "test.csv", low_memory=False, dtype=test_dtypes)
    sample = pd.read_csv(task_dir / "sample_submission.csv", low_memory=False, dtype=sample_dtypes)
    expected_train = [*ids, *features, target]
    if len(train.columns) != len(expected_train) or set(train.columns) != set(expected_train):
        raise ValueError(f"train.csv must contain exactly {expected_train}, got {train.columns.tolist()}")
    expected_test = [*ids, *features]
    if test.columns.tolist() != expected_test:
        raise ValueError(f"test.csv columns/order must be {expected_test}, got {test.columns.tolist()}")
    expected_sample = [*ids, target]
    if sample.columns.tolist() != expected_sample:
        raise ValueError(f"sample_submission.csv columns/order must be {expected_sample}, got {sample.columns.tolist()}")
    if len(test) != int(case["test_rows"]) or len(sample) != len(test):
        raise ValueError(f"test/sample row count mismatch: test={len(test)}, sample={len(sample)}")
    if not np.array_equal(
        test.loc[:, ids].astype("string").to_numpy(dtype=object),
        sample.loc[:, ids].astype("string").to_numpy(dtype=object),
    ):
        raise ValueError("test and sample_submission IDs/order differ")
    X_train = train.loc[:, features].copy()
    X_test = test.loc[:, features].copy()
    y_train = train[target].copy()
    if y_train.isna().any():
        raise ValueError("training target contains missing values")
    if case["task_type"] == "regression":
        y_train = pd.to_numeric(y_train, errors="coerce")
        if not np.isfinite(y_train.to_numpy(dtype=float)).all():
            raise ValueError("training target must be finite numeric")
    return X_train, y_train, X_test, sample.loc[:, ids].copy()


def _fit_and_write(case: dict, model_name: str, threads: int, submission_path: Path) -> dict:
    started = time.perf_counter()
    X_train, y_train, X_test, sample_ids = _load_task_data(case)
    task_type = str(case["task_type"])
    if task_type == "classification" and model_name == "linear":
        numeric_features = [name for name in X_train.columns if pd.api.types.is_numeric_dtype(X_train[name].dtype)]
        categorical_features = [name for name in X_train.columns if name not in numeric_features]
        if not numeric_features and not categorical_features:
            raise ValueError("no usable features for preprocessing")
    elif task_type == "regression" and model_name == "linear":
        numeric_features = [name for name in X_train.columns if pd.api.types.is_numeric_dtype(X_train[name].dtype)]
        categorical_features = [name for name in X_train.columns if name not in numeric_features]
        if not numeric_features and not categorical_features:
            raise ValueError("no usable features for preprocessing")
    elif model_name == "linear":
        raise ValueError(f"unsupported task_type for linear reference: {task_type}")

    if model_name == "linear":
        for column in numeric_features:
            X_train[column] = pd.to_numeric(X_train[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            X_test[column] = pd.to_numeric(X_test[column], errors="coerce").replace([np.inf, -np.inf], np.nan)

    estimator = _make_estimator(model_name, task_type, threads)
    # ColumnTransformer receives concrete feature names instead of dynamic dtype selectors.
    if model_name == "linear":
        transformer = estimator.named_steps["preprocess"]
        configured = []
        if numeric_features:
            configured.append(("numeric", transformer.transformers[0][1], numeric_features))
        if categorical_features:
            configured.append(("categorical", transformer.transformers[1][1], categorical_features))
        estimator.set_params(preprocess__transformers=configured)

    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=int(threads)):
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
    if len(predictions) != len(sample_ids):
        raise ValueError(f"prediction row count mismatch: {len(predictions)} != {len(sample_ids)}")
    if task_type == "regression":
        predictions = np.asarray(predictions, dtype=float)
        if not np.isfinite(predictions).all():
            raise ValueError("reference produced non-finite predictions")
    output = sample_ids.copy()
    output[str(case["target"])] = predictions
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = submission_path.with_suffix(".tmp.csv")
    output.to_csv(temp_path, index=False)
    os.replace(temp_path, submission_path)
    return {
        "model": model_name,
        "model_spec": _model_spec(model_name, task_type),
        "submission_path": str(submission_path.resolve()),
        "fit_predict_seconds": float(time.perf_counter() - started),
        "rows": int(len(predictions)),
    }


def _child_environment(threads: int) -> dict[str, str]:
    env = os.environ.copy()
    value = str(int(threads))
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        env[name] = value
    return env


def _kill_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.communicate(timeout=1.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.communicate()


def _run_process(command: Sequence[str], cwd: Path, timeout_seconds: float, threads: int) -> dict:
    started = time.perf_counter()
    process = subprocess.Popen(
        list(command), cwd=str(cwd), env=_child_environment(threads),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=max(0.01, float(timeout_seconds)))
        return {
            "timed_out": False, "returncode": process.returncode,
            "stdout": stdout, "stderr": stderr,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        stdout, stderr = process.communicate()
        return {
            "timed_out": True, "returncode": process.returncode,
            "stdout": stdout, "stderr": stderr,
            "elapsed_seconds": float(time.perf_counter() - started),
        }


def _worker_command(case_path: Path, workspace: Path, model_name: str, threads: int) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "--workspace", str(workspace),
        "--worker-case", str(case_path), "--model", model_name, "--threads", str(threads),
    ]


def _run_one_model(case: dict, model_name: str, workspace: Path, threads: int, remaining_seconds: float) -> dict:
    task_dir = Path(case["task_dir"])
    case_dir = workspace / "reference" / str(case["name"])
    case_dir.mkdir(parents=True, exist_ok=True)
    submission_path = case_dir / f"{model_name}_submission.csv"
    worker_case_path = case_dir / f".{model_name}_worker_case.json"
    # Do not pass evaluator truth to the fitting subprocess. Only the parent scorer sees it.
    worker_case = {key: value for key, value in case.items() if key != "truth_path"}
    worker_case["submission_path"] = str(submission_path)
    _write_json_atomic(worker_case_path, worker_case)
    repo_root = Path(__file__).resolve().parents[2]
    try:
        process = _run_process(
            _worker_command(worker_case_path, workspace, model_name, threads),
            cwd=repo_root, timeout_seconds=remaining_seconds, threads=threads,
        )
    except OSError as exc:
        return {
            "model": model_name, "status": "worker_error", "valid": False, "score": None,
            "metrics": {}, "error": f"unable to start reference worker: {exc}", "rows": 0,
            "model_spec": _model_spec(model_name, str(case["task_type"])),
        }
    finally:
        worker_case_path.unlink(missing_ok=True)

    if process["timed_out"]:
        return {
            "model": model_name, "status": "timeout", "valid": False, "score": None,
            "metrics": {}, "error": f"reference worker exceeded case deadline ({remaining_seconds:.1f}s)",
            "rows": 0, "elapsed_seconds": process["elapsed_seconds"],
            "model_spec": _model_spec(model_name, str(case["task_type"])),
        }
    if process["returncode"] != 0:
        error = process["stderr"][-4000:] or process["stdout"][-4000:] or f"exit code {process['returncode']}"
        return {
            "model": model_name, "status": "worker_error", "valid": False, "score": None,
            "metrics": {}, "error": error, "rows": 0,
            "elapsed_seconds": process["elapsed_seconds"],
            "model_spec": _model_spec(model_name, str(case["task_type"])),
        }
    try:
        worker_result = json.loads(process["stdout"])
    except json.JSONDecodeError as exc:
        return {
            "model": model_name, "status": "worker_error", "valid": False, "score": None,
            "metrics": {}, "error": f"worker returned invalid JSON: {exc}", "rows": 0,
            "elapsed_seconds": process["elapsed_seconds"],
            "model_spec": _model_spec(model_name, str(case["task_type"])),
        }
    if worker_result.get("model") != model_name or not submission_path.is_file():
        return {
            "model": model_name, "status": "worker_error", "valid": False, "score": None,
            "metrics": {}, "error": "worker result/model/submission mismatch", "rows": 0,
            "elapsed_seconds": process["elapsed_seconds"],
            "model_spec": _model_spec(model_name, str(case["task_type"])),
        }
    from universal_agent.scripts.benchmark_data import score_submission

    evaluation = score_submission(case, submission_path)
    result = {
        **worker_result,
        "status": "completed" if evaluation.get("valid") else "invalid_submission",
        "valid": bool(evaluation.get("valid")),
        "score": evaluation.get("score"),
        "metrics": evaluation.get("metrics", {}),
        "error": evaluation.get("error"),
        "rows": int(evaluation.get("rows", worker_result.get("rows", 0))),
        "elapsed_seconds": float(process["elapsed_seconds"]),
    }
    if process["stderr"].strip():
        result["worker_stderr_tail"] = process["stderr"][-4000:]
    return result


def _source_metadata() -> dict:
    script = Path(__file__).resolve()
    data_module = script.with_name("benchmark_data.py")
    return {
        "source_git_sha": _source_git_sha(),
        "source_sha256": _sha256_file(script),
        "data_module_sha256": _sha256_file(data_module),
    }


def run_workspace(workspace: str | Path, threads: int = 2) -> dict:
    """Run the two frozen references sequentially for each cases.json task."""
    workspace = Path(workspace).expanduser().resolve()
    if int(threads) < 1:
        raise ValueError("threads must be >= 1")
    cases_path = workspace / "cases.json"
    if not cases_path.is_file():
        raise FileNotFoundError(f"cases.json not found: {cases_path}")
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases.json must contain a non-empty case list")
    names = [str(case.get("name", "")) for case in cases if isinstance(case, dict)]
    if len(names) != len(cases) or not all(names) or len(names) != len(set(names)):
        raise ValueError("each case must be a dict with a unique non-empty name")

    results = []
    output = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_status": "running",
        "threads": int(threads),
        "timeout_seconds": {"public": 180, "synthetic": 60},
        "reference_spec": REFERENCE_SPEC,
        **_source_metadata(),
        "cases": results,
    }
    output_path = workspace / "reference_results.json"

    for case in cases:
        case_name = str(case["name"])
        case_result = {
            "name": case_name,
            "task_type": case.get("task_type"),
            "metric": case.get("metric"),
            "split_fingerprint": case.get("split_fingerprint"),
            "input_hashes": case.get("input_hashes", {}),
            "runs": [],
        }
        valid_inputs, actual_hashes, hash_error = _verify_case_inputs(case)
        case_result["verified_input_hashes"] = actual_hashes
        case_result["input_hashes_valid"] = valid_inputs
        if not valid_inputs:
            for model_name in REFERENCE_SPEC:
                case_result["runs"].append({
                    "model": model_name,
                    "model_spec": _model_spec(model_name, str(case.get("task_type"))),
                    "status": "input_hash_mismatch",
                    "valid": False, "score": None, "metrics": {}, "error": hash_error, "rows": 0,
                })
            results.append(case_result)
            output["cases"] = results
            _write_json_atomic(output_path, output)
            continue

        budget = _case_timeout_seconds(case)
        deadline = time.monotonic() + budget
        for model_name in REFERENCE_SPEC:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                case_result["runs"].append({
                    "model": model_name,
                    "model_spec": _model_spec(model_name, str(case["task_type"])),
                    "status": "deadline_exhausted", "valid": False, "score": None,
                    "metrics": {}, "error": "case reference time budget exhausted before worker start", "rows": 0,
                })
                continue
            case_result["runs"].append(_run_one_model(case, model_name, workspace, int(threads), remaining))
            output["cases"] = [*results, case_result]
            _write_json_atomic(output_path, output)
        results.append(case_result)

    output["run_status"] = (
        "completed"
        if all(run.get("status") == "completed" and run.get("valid")
               for case_result in results for run in case_result["runs"])
        else "partial_failure"
    )
    output["cases"] = results
    _write_json_atomic(output_path, output)
    return output


def _worker_main(case_path: Path, model_name: str, threads: int) -> int:
    case = json.loads(case_path.read_text(encoding="utf-8"))
    submission_path = Path(case.pop("submission_path"))
    result = _fit_and_write(case, model_name, int(threads), submission_path)
    sys.stdout.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-case", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=tuple(REFERENCE_SPEC), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.worker_case is None or args.model is None:
            parser.error("worker mode requires --worker-case and --model")
        return _worker_main(args.worker_case, args.model, args.threads)
    run_workspace(args.workspace, args.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
