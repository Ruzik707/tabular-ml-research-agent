"""Human and structured projections of the redacted event journal.

The JSONL event journal remains authoritative. These two files are deliberately
rebuildable views and only expose allowlisted public facts and decision
summaries; provider messages, tool arguments, source code, and response bodies
never enter either projection.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterator
from contextlib import contextmanager

try:
    import fcntl
except ImportError:  # pragma: no cover - MVP platforms are macOS/Linux
    fcntl = None

from .journal import EventJournal


_PRIVATE_KEY_PARTS = (
    "prompt", "message", "code", "raw", "thinking", "thought", "reasoning",
    "chain_of_thought", "internal", "authorization", "credential", "api_key",
    "secret", "password", "argument", "response", "stderr", "stdout", "traceback",
)
_LLM_EVENTS = {
    "llm_request", "llm_response", "llm_error", "llm_protocol_error",
    "llm_response_consumed", "llm_response_recovered",
}
_WORKER_EVENTS = {"trial_completed", "final_fit_completed"}


class AuditJournal(EventJournal):
    """Append canonical events, then maintain Russian and JSONL audit views."""

    def __init__(self, path: Path, secrets=(), echo: bool = True) -> None:
        super().__init__(path, secrets=secrets)
        self.echo = bool(echo)
        self.agent_log_path = self.path.parent / "agent.log"
        self.decisions_path = self.path.parent / "decisions.jsonl"
        self._projection_lock_path = self.path.parent / ".audit-projections.lock"
        try:
            self._rebuild_projections()
        except Exception as exc:  # projections must not prevent canonical recovery/use
            self._warn(None, "проекции при запуске", exc)

    def append(self, event: str, **payload: Any) -> dict:
        # Make the durable event first. If a derived view, lock, or terminal
        # fails, callers still receive the canonical record and a clear warning.
        record = super().append(event, **payload)
        projection_error = None
        try:
            self._rebuild_projections()
        except Exception as exc:
            projection_error = exc
        echo_error = None
        if self.echo:
            try:
                line = _render_human(_project_event(record, {}))
                sys.stderr.write(line + "\n")
                sys.stderr.flush()
            except Exception as exc:
                echo_error = exc
        if projection_error is not None:
            self._warn(record["seq"], "обновление проекций", projection_error)
        if echo_error is not None:
            self._warn(record["seq"], "вывод в stderr", echo_error)
        return record

    @contextmanager
    def _projection_lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._projection_lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            with self._thread_lock:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _rebuild_projections(self) -> None:
        # Re-read under a cross-process projection lock. Rewriting from the
        # source of truth prevents duplicate lines after interrupted writes.
        with self._projection_lock():
            events = self.read_events()
            records = _project_events(events)
            human = "".join(_render_human(record) + "\n" for record in records)
            structured = "".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
                for record in records
            )
            _write_projection(self.agent_log_path, human)
            _write_projection(self.decisions_path, structured)

    @staticmethod
    def _warn(sequence: int | None, stage: str, exc: Exception) -> None:
        if sequence is None:
            message = (
                f"Предупреждение: не удалось завершить {stage} ({type(exc).__name__}). "
                "events.jsonl остаётся источником истины; повторно откройте AuditJournal "
                "для восстановления проекций."
            )
        else:
            message = (
                f"Предупреждение: событие №{sequence} сохранено в events.jsonl, "
                f"но не удалось завершить {stage} ({type(exc).__name__}). "
                "Повторно откройте AuditJournal для восстановления проекций из events.jsonl."
            )
        try:
            sys.stderr.write(message + "\n")
            sys.stderr.flush()
        except Exception:
            # Canonical durability takes precedence when stderr itself is broken.
            pass


def _project_events(events: list[dict]) -> list[dict]:
    baseline_trials: dict[str, bool] = {}
    projected = []
    for event in events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if event.get("event") == "trial_started" and isinstance(payload.get("trial_id"), str):
            baseline_trials[payload["trial_id"]] = bool(payload.get("is_baseline"))
        projected.append(_project_event(event, baseline_trials))
    return projected


def _project_event(event: dict, baseline_trials: dict[str, bool]) -> dict:
    event_name = _bounded_text(event.get("event", "unknown"), 100)
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    source = _source_for(event_name, payload, baseline_trials)
    details = _details_for(event_name, payload)
    return {
        "seq": event.get("seq"),
        "timestamp": _bounded_text(event.get("timestamp", ""), 80),
        "event": event_name,
        "source": source,
        "details": details,
    }


def _source_for(event: str, payload: dict, baseline_trials: dict[str, bool]) -> str:
    explicit = str(payload.get("source", "")).strip().lower()
    if explicit in {"llm", "model", "provider"}:
        return "LLM"
    if explicit == "worker":
        return "worker"
    if explicit == "harness":
        return "harness"
    if event == "decision_recorded":
        return "harness"
    if event in _LLM_EVENTS or event == "tool_call":
        return "LLM"
    if event == "trial_started":
        return "harness" if payload.get("is_baseline") else "LLM"
    if event == "trial_completed":
        result = payload.get("result")
        trial_id = result.get("trial_id") if isinstance(result, dict) else None
        if (isinstance(result, dict) and result.get("is_baseline")) or (
            trial_id in baseline_trials and baseline_trials[trial_id]
        ):
            return "harness"
        return "worker"
    if event in _WORKER_EVENTS:
        return "worker"
    return "harness"


def _details_for(event: str, payload: dict) -> dict:
    if event == 'family_comparison_completed':
        comparison = payload.get('comparison', {})
        if not isinstance(comparison, dict):
            return {}
        compact = {k:comparison[k] for k in ('metric','maximize','selected_trial','split_fingerprint') if k in comparison}
        compact['trials'] = []
        for row in comparison.get('trials', [])[:5]:
            if not isinstance(row, dict):
                continue
            item = _pick(row, ('family','trial_id','status','metric','seconds'))
            details = row.get('model_details', {})
            if isinstance(details, dict):
                item['model_details'] = _pick(details, ('class','parameters','parameters_source'))
            compact['trials'].append(item)
        return {'comparison':compact}
    if event == "family_comparison_started":
        return _selected(payload, ("families", "planned", "limited_by_trial_budget", "split_fingerprint", "comparison"))
    if event in {"batch_started", "batch_completed", "batch_interrupted"}:
        return _selected(payload, ("trial_ids", "origin", "parallel_trials", "threads_per_trial", "memory_limit_mb", "remaining_seconds"))
    if event == "recipe_rejected":
        return _selected(payload, ("base_trial_id", "hypothesis", "error"))
    if event == "research_mode_selected":
        return _selected(payload, ("requested", "effective"))
    if event == "reserve_updated":
        return _selected(payload, ("previous_reserve", "plan", "remaining_seconds"))
    if event == "data_profiled":
        return _selected(payload, ("summary",))
    if event == "validation_planned":
        return {"summary": _project_validation_plan(payload)}
    if event == "dataset_diagnosed":
        return _project_dataset_diagnosis(payload)
    if event == "postprocessing_selected":
        return _project_postprocessing(payload)
    if event == "research_strategy":
        return _selected(payload, ("strategy",))
    if event == "decision_recorded":
        return _selected(payload, (
            "trial_id", "hypothesis", "decision", "explanation_provided", "model_family",
            "observations", "rationale", "alternatives", "expected_effect", "risks",
            "previous_result_assessment",
        ))
    if event == "experiment_assessed":
        return _selected(payload, (
            "trial_id", "metric_name", "metric", "previous_best", "maximize",
            "selected", "delta", "reason", "model_details", "recovered", "completion_seq",
        ))
    if event == "research_finished":
        return _selected(payload, ("reason",))
    if event == "research_stopped":
        return _selected(payload, ("reason", "total_tokens", "elapsed_seconds"))
    if event == "submission_validated":
        return _selected(payload, ("validation", "submission_path"))
    if event in {"run_started", "run_resumed"}:
        config = payload.get("config")
        return {
            "data_fingerprint": _clean_public(payload.get("data_fingerprint")),
            "config": _pick(config, ("provider", "model", "offline", "max_trials", "time_budget")),
        }
    if event == "run_completed":
        report = payload.get("report")
        details = _pick(report, (
            "status", "stop_reason", "task_type", "metric", "best_metric", "selected_trial",
            "llm_calls", "llm_trials", "total_tokens", "elapsed_seconds", "decision_log", "decisions_path",
        ))
        if isinstance(report, dict) and isinstance(report.get("final_fit"), dict):
            final = _pick(report["final_fit"], ("status", "seconds"))
            if final:
                details["final_fit"] = final
        return details
    if event == "trial_started":
        return _selected(payload, ("trial_id", "hypothesis", "is_baseline", "recipe", "parent_trial_id", "recipe_hash"))
    if event == "trial_completed":
        result = payload.get("result")
        details = _pick(result, (
            "trial_id", "status", "metric", "metrics", "seconds", "hypothesis",
            "is_baseline", "model_details",
        ))
        if isinstance(result, dict):
            diagnostics = _project_trial_diagnostics(result.get("diagnostics"))
            if diagnostics:
                details["diagnostics"] = diagnostics
        if isinstance(result, dict) and result.get("status") != "ok":
            code = _short_error_code(result.get("error"), result.get("status"))
            if code:
                details["error_code"] = code
        return details
    if event == "candidate_selected":
        return _selected(payload, ("trial_id", "metric")) | {"selected": True}
    if event == "candidate_restored":
        return _selected(payload, ("trial_id", "metric")) | {"restored": True}
    if event == "fallback_written":
        return _selected(payload, ("strategy",))
    if event == "llm_request":
        return _selected(payload, ("request_id", "provider", "model", "max_tokens", "input_token_bound"))
    if event == "llm_response":
        response = payload.get("response")
        result = _selected(payload, ("request_id", "charged_tokens"))
        if isinstance(response, dict):
            usage = _pick(response, ("prompt_tokens", "completion_tokens", "total_tokens"))
            if usage:
                result["usage"] = usage
        return result
    if event == "llm_error":
        details = _selected(payload, ("request_id", "charged_tokens"))
        if isinstance(payload.get("code"), str):
            details["error_code"] = _bounded_text(payload["code"], 100)
        return details
    if event in {"llm_protocol_error", "llm_response_consumed", "llm_response_recovered"}:
        return _selected(payload, ("request_id",))
    if event == "context_compacted":
        return _selected(payload, ("reason",))
    if event == "tool_call":
        return _selected(payload, ("call_id", "name"))
    if event == "tool_result":
        return _tool_result_details(payload)
    if event == "final_fit_started":
        return _selected(payload, ("trial_id",))
    if event == "final_fit_completed":
        result = payload.get("result")
        details = _pick(result, ("status", "seconds", "model_details"))
        if isinstance(result, dict) and result.get("status") != "ok":
            code = _short_error_code(result.get("error"), result.get("status"))
            if code:
                details["error_code"] = code
            artifact_dir = result.get("artifact_dir")
            if isinstance(artifact_dir, str) and artifact_dir:
                details["process_ref"] = Path(artifact_dir).name + "/process.json"
        return details
    if event == "run_error":
        return {"error_code": _short_error_code(payload.get("error"), "orchestration_error")}
    if event == "run_interrupted":
        return {}
    # Unknown/legacy events retain their identity, but their unrecognized
    # payload is omitted until a public projection contract is defined.
    return {}


def _tool_result_details(payload: dict) -> dict:
    details = _selected(payload, ("call_id", "name"))
    result = payload.get("result")
    if not isinstance(result, dict):
        return details
    if "ok" in result:
        details["ok"] = bool(result["ok"])
    error = result.get("error")
    if isinstance(error, dict) and isinstance(error.get("code"), str):
        details["error_code"] = _bounded_text(error["code"], 100)
    value = result.get("value")
    if isinstance(value, dict):
        details["value"] = _pick(value, ("finished", "trial_id", "metrics", "seconds", "selected", "best_metric"))
    return details


def _public_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _public_text(value: Any, limit: int = 120) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return _bounded_text(text, limit) if text else None


def _public_label(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    number = _public_number(value)
    if number is not None:
        return number
    return _public_text(value)


def _bounded_string_list(value: Any, limit: int, item_chars: int = 180) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [text for item in value[:limit] if (text := _public_text(item, item_chars)) is not None]


def _project_validation_plan(payload: dict) -> dict:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else payload
    result = {}
    for key in ("seed", "fraction", "split_fingerprint", "policy", "support_policy", "fold_count"):
        if key not in summary:
            continue
        value = summary[key]
        if key in {"seed", "fraction", "fold_count"}:
            number = _public_number(value)
            if number is not None:
                result[key] = number
        else:
            text = _public_text(value, 180)
            if text is not None:
                result[key] = text
    coverage = summary.get("coverage")
    if isinstance(coverage, dict):
        allowed_coverage = (
            "eligible_rows", "oof_rows", "covered_rows", "total_rows", "fraction",
            "coverage", "folds", "train_rows", "validation_rows", "unique_groups",
        )
        compact_coverage = {
            key: _public_number(coverage[key])
            for key in allowed_coverage
            if key in coverage and _public_number(coverage[key]) is not None
        }
        if compact_coverage:
            result["coverage"] = compact_coverage
    elif (number := _public_number(coverage)) is not None:
        result["coverage"] = number
    warnings = _bounded_string_list(summary.get("warnings"), 5)
    if warnings:
        result["warnings"] = warnings
    return result


def _project_auc(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in ("status", "metric", "reason", "folds", "roc_auc_mean", "roc_auc_std", "seconds", "elapsed_seconds"):
        if key not in value:
            continue
        if key in {"status", "metric", "reason"}:
            text = _public_text(value[key], 180 if key == "reason" else 40)
            if text is not None:
                result[key] = text
        elif (number := _public_number(value[key])) is not None:
            result[key] = number
    return result


def _project_temporal_candidates(value: Any) -> list[dict]:
    if not isinstance(value, (list, tuple)):
        return []
    items = []
    for candidate in value[:5]:
        if not isinstance(candidate, dict):
            continue
        item = {}
        for key in ("feature", "kind", "direction"):
            text = _public_text(candidate.get(key), 100)
            if text is not None:
                item[key] = text
        for key in ("future_only",):
            if isinstance(candidate.get(key), bool):
                item[key] = candidate[key]
        for key in ("train_group_count", "train_min", "train_max", "test_min", "test_max"):
            raw = candidate.get(key)
            if isinstance(raw, str):
                text = _public_text(raw, 60)
                if text is not None:
                    item[key] = text
            elif (number := _public_number(raw)) is not None:
                item[key] = number
        if item:
            items.append(item)
    return items


def _feature_shift_item(name: Any, profile: Any) -> tuple[dict | None, float]:
    if not isinstance(profile, dict):
        return None, 0.0
    feature = _public_text(name, 100)
    kind = _public_text(profile.get("kind"), 30)
    item = {}
    if feature is not None:
        item["feature"] = feature
    if kind is not None:
        item["kind"] = kind
    score = 0.0
    for key in ("train_missing_rate", "test_missing_rate", "mean_difference"):
        if (number := _public_number(profile.get(key))) is not None:
            item[key] = number
            if key == "mean_difference":
                score = max(score, abs(float(number)))
    shift = profile.get("category_shift") if isinstance(profile.get("category_shift"), dict) else profile
    for key in ("train_only_count", "test_only_count"):
        if (number := _public_number(shift.get(key))) is not None:
            item[key] = number
            score = max(score, float(number))
    if "train_missing_rate" in item and "test_missing_rate" in item:
        score = max(score, abs(float(item["train_missing_rate"]) - float(item["test_missing_rate"])))
    return (item if item else None), score


def _project_feature_shift(raw: Any) -> tuple[list[dict], int]:
    if isinstance(raw, dict):
        entries = list(raw.items())
    elif isinstance(raw, (list, tuple)):
        entries = [
            (entry.get("feature", f"feature_{index}"), entry)
            for index, entry in enumerate(raw)
            if isinstance(entry, dict)
        ]
    else:
        return [], 0
    changed = []
    for name, profile in entries:
        item, score = _feature_shift_item(name, profile)
        if item is not None and score > 0:
            changed.append((score, item))
    changed.sort(key=lambda pair: (-pair[0], pair[1].get("feature", "")))
    return [item for _, item in changed[:8]], len(changed)


def _project_dataset_diagnosis(payload: dict) -> dict:
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else payload
    result = {}
    adversarial = diagnostics.get("adversarial_cv")
    if isinstance(adversarial, dict):
        primary = _project_auc(adversarial)
        if primary:
            result["adversarial_cv"] = primary
        without_temporal = _project_auc(adversarial.get("without_temporal"))
        if without_temporal:
            result.setdefault("adversarial_cv", {})["without_temporal"] = without_temporal
    temporal = _project_temporal_candidates(diagnostics.get("temporal_candidates"))
    if temporal:
        result["temporal_candidates"] = temporal
    shift, total = _project_feature_shift(diagnostics.get("feature_shift", diagnostics.get("features")))
    if shift or total:
        result["feature_shift"] = shift
        result["feature_shift_total"] = total
        result["feature_shift_partial"] = total > len(shift)
    return result


def _project_postprocessing(payload: dict) -> dict:
    plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
    payload = {**payload, **plan}
    accepted = plan.get("accepted", payload.get("accepted"))
    result = {"accepted": accepted} if isinstance(accepted, bool) else {}
    members = payload.get("member_trial_ids")
    if isinstance(members, (list, tuple)):
        result["member_trial_ids"] = [
            text for item in members[:3] if (text := _public_text(item, 80)) is not None
        ]
    weights = payload.get("weights")
    if isinstance(weights, (list, tuple)):
        result["weights"] = [number for item in weights[:3] if (number := _public_number(item)) is not None]
    threshold = _public_number(payload.get('binary_threshold'))
    if threshold is not None and 0 <= threshold <= 1:
        result['binary_threshold'] = threshold
        if (label := _public_label(payload.get('positive_class'))) is not None:
            result['positive_class'] = label
    multipliers = payload.get("class_multipliers")
    if isinstance(multipliers, dict):
        compact = {}
        for label, value in list(multipliers.items())[:20]:
            safe_label = _public_label(label)
            number = _public_number(value)
            if safe_label is not None and number is not None:
                compact[str(safe_label)[:80]] = number
        if compact:
            result["class_multipliers"] = compact
    elif isinstance(multipliers, (list, tuple)):
        compact = []
        axis = payload.get('classes', [])
        for index, item in enumerate(multipliers[:20]):
            if not isinstance(item, dict):
                if index < len(axis) and (number := _public_number(item)) is not None:
                    label = _public_label(axis[index])
                    if label is not None:
                        compact.append({'label':label,'multiplier':number})
                continue
            label = _public_label(item.get("label", item.get("class")))
            number = _public_number(item.get("multiplier"))
            if label is not None and number is not None:
                compact.append({"label": label, "multiplier": number})
        if compact:
            result["class_multipliers"] = compact
    if (reason := _public_text(payload.get('reason'), 240)) is not None:
        result['reason'] = reason
    for key in ("baseline_score", "assessed_score", "calibrated_oof_score"):
        source = payload.get(key)
        if (number := _public_number(source)) is not None:
            result[key] = number
    assessment = payload.get("assessment")
    if isinstance(assessment, str):
        text = _public_text(assessment, 240)
        if text is not None:
            result["assessment"] = text
    elif isinstance(assessment, dict):
        compact = {}
        for key in ("status", "reason_code", "reason", "metric_name", "selection_metric", "selected_members"):
            if key not in assessment:
                continue
            value = assessment[key]
            if key == "selected_members" and isinstance(value, (list, tuple)):
                compact[key] = [text for item in value[:3] if (text := _public_text(item, 80)) is not None]
            elif isinstance(value, str) and (text := _public_text(value, 180)) is not None:
                compact[key] = text
            elif (number := _public_number(value)) is not None:
                compact[key] = number
        if compact:
            result["assessment"] = compact
    warnings = _bounded_string_list(payload.get("warnings"), 4)
    if warnings:
        result["warnings"] = warnings
    return result


def _project_per_class(value: Any) -> tuple[list[dict], int, bool]:
    if not isinstance(value, (list, tuple)):
        return [], 0, False
    items = []
    for row in value[:8]:
        if not isinstance(row, dict):
            continue
        item = {}
        label = _public_label(row.get("label"))
        if label is not None:
            item["label"] = label
        for key in ("precision", "recall", "f1", "support"):
            if (number := _public_number(row.get(key))) is not None:
                item[key] = number
        if item:
            items.append(item)
    return items, len(value), len(value) > len(items)


def _project_confusion(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}
    labels = value.get("labels")
    matrix = value.get("matrix")
    if not isinstance(labels, (list, tuple)) or not isinstance(matrix, (list, tuple)):
        return {}
    size = min(10, len(labels), len(matrix))
    safe_labels = [_public_label(item) for item in labels[:size]]
    if not all(item is not None for item in safe_labels):
        return {}
    safe_matrix = []
    for row in matrix[:size]:
        if not isinstance(row, (list, tuple)) or len(row) < size:
            return {}
        safe_row = []
        for item in row[:size]:
            number = _public_number(item)
            if number is None:
                return {}
            safe_row.append(number)
        safe_matrix.append(safe_row)
    result = {"labels": safe_labels, "matrix": safe_matrix}
    for key in ("total_classes", "partial", "selection"):
        if key == "partial" and isinstance(value.get(key), bool):
            result[key] = value[key]
        elif key == "selection" and (text := _public_text(value.get(key), 60)) is not None:
            result[key] = text
        elif key == "total_classes" and (number := _public_number(value.get(key))) is not None:
            result[key] = number
    if len(labels) > size:
        result["partial"] = True
    return result


def _project_segments(value: Any) -> tuple[list[dict], int, bool]:
    if not isinstance(value, (list, tuple)):
        return [], 0, False
    items = []
    for row in value[:5]:
        if not isinstance(row, dict):
            continue
        item = {}
        for key in ("feature",):
            if (text := _public_text(row.get(key), 100)) is not None:
                item[key] = text
        if "category" in row:
            label = _public_label(row["category"])
            if label is not None:
                item["category"] = label
        for key in ("count", "rmse", "mae"):
            if (number := _public_number(row.get(key))) is not None:
                item[key] = number
        if item:
            items.append(item)
    return items, len(value), len(value) > len(items)


def _project_validation_metrics(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in ("status", "metric_name", "metric", "score", "seconds", "count", "rows", "fold_count"):
        if key not in value:
            continue
        if key in {"status", "metric_name", "metric"}:
            text = _public_text(value[key], 60)
            if text is not None:
                result[key] = text
        elif (number := _public_number(value[key])) is not None:
            result[key] = number
    return result


def _project_importance(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in ("status", "method", "mapping_status"):
        text = _public_text(value.get(key), 60)
        if text is not None:
            result[key] = text
    raw_items = value.get("items")
    if isinstance(raw_items, (list, tuple)):
        items = []
        for row in raw_items:
            if not isinstance(row, dict):
                continue
            feature = _public_text(row.get("feature"), 100)
            importance = _public_number(row.get("importance"))
            if feature is not None and importance is not None:
                items.append({"feature": feature, "importance": importance})
        items.sort(key=lambda item: (-float(item["importance"]), item["feature"]))
        result["items"] = items[:8]
        result["total_items"] = len(raw_items)
        result["partial"] = len(items) > 8 or len(raw_items) > len(items)
    return result


def _project_trial_diagnostics(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}
    result = {}
    per_class, total_classes, class_partial = _project_per_class(value.get("per_class"))
    if per_class or "total_classes" in value:
        result["per_class"] = per_class
        total = _public_number(value.get("total_classes"))
        result["total_classes"] = total if total is not None else total_classes
        result["partial"] = bool(value.get("partial", class_partial)) or class_partial
    confusion = _project_confusion(value.get("confusion_matrix"))
    if confusion:
        result["confusion_matrix"] = confusion
    segments, total_segments, segment_partial = _project_segments(value.get("segments"))
    if segments or "total_segments" in value:
        result["segments"] = segments
        total = _public_number(value.get("total_segments"))
        result["total_segments"] = total if total is not None else total_segments
        result["segments_partial"] = bool(value.get("partial", segment_partial)) or segment_partial
    validation = _project_validation_metrics(value.get("validation"))
    if validation:
        result["validation"] = validation
    importance = _project_importance(value.get("feature_importance"))
    if importance:
        result["feature_importance"] = importance
    return result


def _selected(payload: dict, fields: tuple[str, ...]) -> dict:
    return {key: _clean_public(payload[key]) for key in fields if key in payload and _is_public_key(key)}


def _pick(value: Any, fields: tuple[str, ...]) -> dict:
    if not isinstance(value, dict):
        return {}
    return {key: _clean_public(value[key]) for key in fields if key in value and _is_public_key(key)}


def _is_public_key(key: Any) -> bool:
    normalized = str(key).lower().replace("-", "_")
    if normalized == "error_code":
        return True
    return not any(part in normalized for part in _PRIVATE_KEY_PARTS)


def _clean_public(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "[truncated]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _bounded_text(value, 6000)
    if isinstance(value, dict):
        result = {}
        for key, child in list(value.items())[:60]:
            if _is_public_key(key):
                result[_bounded_text(str(key), 100)] = _clean_public(child, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        items = [_clean_public(item, depth + 1) for item in value[:40]]
        if len(value) > 40:
            items.append("[ещё элементы опущены]")
        return items
    return _bounded_text(str(value), 300)


def _bounded_text(value: Any, limit: int) -> str:
    text = str(value)
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def _render_human(record: dict) -> str:
    event = record.get("event", "unknown")
    source = record.get("source", "harness")
    details = record.get("details", {})
    timestamp = record.get("timestamp", "")
    seq = record.get("seq", "?")
    label = _event_text(event, details)
    if event == "trial_completed" and details.get("status") != "ok":
        trial_id = details.get("trial_id", "trial")
        label += (
            f" Код ошибки: {_human_value(details.get('error_code'))}; "
            f"диагностика: events.jsonl #{seq} и {trial_id}/process.json."
        )
    elif event == "final_fit_completed" and details.get("status") != "ok":
        label += (
            f" Код ошибки: {_human_value(details.get('error_code'))}; "
            f"диагностика: events.jsonl #{seq}"
        )
        if details.get("process_ref"):
            label += f" и {details['process_ref']}"
        label += "."
    return f"[{timestamp}] #{seq} [{source}] {label}"


def _event_text(event: str, details: dict) -> str:
    if event == "family_comparison_started":
        return f"Сравнение семейств на общем split: {_human_value(details.get('families'))}; новых опытов={details.get('planned')}; ограничено числом trials={details.get('limited_by_trial_budget')}."
    if event == "family_comparison_completed":
        return f"Результаты сравнения семейств: {_human_value(details.get('comparison'))}."
    if event == "batch_started":
        return f"Запущена группа {_human_value(details.get('trial_ids'))}: до {details.get('parallel_trials')} процессов по {details.get('threads_per_trial')} CPU; лимит RSS одного {details.get('memory_limit_mb')} MB."
    if event == "batch_completed":
        return f"Группа {_human_value(details.get('trial_ids'))} завершена; осталось на исследования {details.get('remaining_seconds')} с."
    if event == "batch_interrupted":
        return f"Группа прервана; процессы остановлены, неизвестные исходы будут отмечены при resume: {_human_value(details.get('trial_ids'))}."
    if event == "recipe_rejected":
        return f"Рецепт отклонён до обучения; основа={details.get('base_trial_id')}: {_human_value(details.get('error'))}."
    if event == "research_mode_selected":
        return f"Режим экспериментов: запрошен {details.get('requested')}, используется {details.get('effective')}."
    if event == "reserve_updated":
        return f"Пересчитан резерв финального обучения: {_human_value(details.get('plan'))}; осталось всего {details.get('remaining_seconds')} с."
    if event == "run_started":
        return "Начат новый запуск исследования."
    if event == "run_resumed":
        return "Возобновлён запуск исследования."
    if event == "data_profiled":
        return f"Профиль данных: {_human_value(details.get('summary'))}."
    if event == "validation_planned":
        summary = details.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        parts = []
        for key, label in (("support_policy", "политика"), ("policy", "схема"),
                           ("fold_count", "фолдов"), ("fraction", "доля")):
            if key in summary:
                parts.append(f"{label}={_human_value(summary[key])}")
        coverage = summary.get("coverage")
        if coverage is not None:
            parts.append(f"покрытие={_human_value(coverage)}")
        warnings = summary.get("warnings")
        if warnings:
            parts.append("предупреждения=" + _human_value(warnings[:2] if isinstance(warnings, list) else warnings))
        return "План проверки: " + ("; ".join(parts) if parts else "параметры не указаны") + "."
    if event == "dataset_diagnosed":
        return _dataset_diagnosis_text(details)
    if event == "postprocessing_selected":
        accepted = details.get("accepted")
        status = "принята" if accepted is True else "отклонена" if accepted is False else "статус не указан"
        text = f"План блендинга/постобработки {status}"
        if details.get("member_trial_ids"):
            text += f"; участники={_human_value(details['member_trial_ids'])}"
        if details.get("weights"):
            text += f"; веса={_human_value(details['weights'])}"
        if details.get("class_multipliers"):
            text += f"; множители классов={_human_value(details['class_multipliers'])}"
        if 'binary_threshold' in details:
            text += f"; бинарный порог={_human_value(details['binary_threshold'])}"
            if 'positive_class' in details:
                text += f"; положительный класс={_human_value(details['positive_class'])}"
        scores = []
        for key, label in (("baseline_score", "baseline"), ("assessed_score", "оценка"),
                           ("calibrated_oof_score", "OOF calibration")):
            if key in details:
                scores.append(f"{label}={_human_value(details[key])}")
        if scores:
            text += "; " + ", ".join(scores)
        if details.get("assessment") is not None:
            text += f"; вывод={_human_value(details['assessment'])}"
        if details.get("warnings"):
            text += f"; предупреждения={_human_value(details['warnings'][:2])}"
        return text + "."
    if event == "research_strategy":
        return f"Стратегия исследования: {_human_value(details.get('strategy'))}."
    if event == "decision_recorded":
        trial = details.get("trial_id", "без trial_id")
        hypothesis = details.get("hypothesis")
        decision = details.get("decision")
        text = f"Решение для {trial}"
        if hypothesis:
            text += f": гипотеза: {_human_value(hypothesis)}"
        fields = (
            ("model_family", "Семейство модели"),
            ("observations", "Наблюдения"),
            ("rationale", "Обоснование"),
            ("alternatives", "Рассмотренные альтернативы"),
            ("expected_effect", "Ожидаемый эффект"),
            ("risks", "Риски"),
            ("previous_result_assessment", "Оценка прошлого результата"),
        )
        for key, label in fields:
            value = details.get(key)
            if value is None:
                value = _get_nested(decision, key)
            if value is None:
                value = _get_nested(hypothesis, key)
            if value is not None:
                text += f"; {label}: {_human_value(value)}"
        if details.get("explanation_provided") is False and not any(
            key in details or key in (decision if isinstance(decision, dict) else {})
            for key in ("rationale", "observations", "expected_effect", "previous_result_assessment")
        ):
            text += "; объяснение не предоставлено"
        return text + "."
    if event == "experiment_assessed":
        metric = details.get("metric")
        metric_name = details.get("metric_name", "метрика")
        result = f"Фактическая валидация {details.get('trial_id', 'trial')}: {metric_name}={_human_value(metric)}"
        if details.get("recovered"):
            result += "; оценка восстановлена из сохранённого результата"
        if "previous_best" in details:
            result += f", прошлый лучший результат={_human_value(details['previous_best'])}"
        if "delta" in details:
            result += f", изменение={_human_value(details['delta'])}"
        if "maximize" in details:
            result += ", направление=максимизация" if details["maximize"] else ", направление=минимизация"
        result += "; кандидат выбран" if details.get("selected") else "; кандидат не выбран"
        if details.get("reason"):
            result += f". Причина: {_human_value(details['reason'])}"
        chain = _model_chain(details.get("model_details"))
        if chain:
            result += f". Фактическая цепочка модели: {chain}"
        parameters = _get_nested(details.get("model_details"), "parameters")
        if parameters:
            result += f". Фактические параметры: {_human_value(parameters)}"
        fallback = _get_nested(details.get("model_details"), "fallback_reason")
        if fallback:
            result += f". Причина перехода на запасную модель: {_human_value(fallback)}"
        return result + "."
    if event == "research_finished":
        return f"Исследование завершено: {_human_value(details.get('reason', 'причина не указана'))}."
    if event == "research_stopped":
        return (f"Исследование остановлено: {_human_value(details.get('reason', 'причина не указана'))}; "
                f"токены={_human_value(details.get('total_tokens'))}, время={_human_value(details.get('elapsed_seconds'))} с.")
    if event == "submission_validated":
        return f"Проверка submission: {_human_value(details.get('validation'))}; файл={_human_value(details.get('submission_path'))}."
    if event == "trial_started":
        trial = details.get("trial_id", "trial")
        if details.get("recipe"):
            return f"Запущен эксперимент {trial}: {_human_value(details.get('hypothesis'))}; основа={details.get('parent_trial_id')}; применённый рецепт={_human_value(details['recipe'])}."
        if details.get("is_baseline"):
            return f"Запущен базовый эксперимент {trial}: {_human_value(details.get('hypothesis'))}."
        return f"Запущен эксперимент {trial}: {_human_value(details.get('hypothesis'))}."
    if event == "trial_completed":
        result = f"Эксперимент {details.get('trial_id', 'trial')} завершён со статусом {details.get('status', 'unknown')}"
        if details.get("metric") is not None:
            result += f", метрика={_human_value(details['metric'])}"
        if details.get("seconds") is not None:
            result += f", время={_human_value(details['seconds'])} с"
        diagnostic_text = _trial_diagnostics_text(details.get("diagnostics"))
        return result + (f". {diagnostic_text}" if diagnostic_text else ".")
    if event == "candidate_selected":
        return f"Выбран лучший кандидат {details.get('trial_id', 'trial')}: метрика={_human_value(details.get('metric'))}."
    if event == "candidate_restored":
        return f"Восстановлен лучший кандидат {details.get('trial_id', 'trial')}: метрика={_human_value(details.get('metric'))}."
    if event == "fallback_written":
        return f"Записан резервный submission; стратегия={_human_value(details.get('strategy'))}."
    if event == "llm_request":
        return f"Отправлен запрос LLM {details.get('request_id', '')}: {_human_value(details.get('provider'))}/{_human_value(details.get('model'))}; полный запрос сохранён в events.jsonl."
    if event == "llm_response":
        return f"Получен ответ LLM {details.get('request_id', '')}; списано токенов={_human_value(details.get('charged_tokens'))}."
    if event == "llm_error":
        return f"Ошибка запроса LLM {details.get('request_id', '')}: код={_human_value(details.get('error_code', 'unknown'))}."
    if event == "llm_protocol_error":
        return f"Ошибка формата ответа LLM {details.get('request_id', '')}."
    if event == "llm_response_recovered":
        return f"Восстановлен ожидающий ответ LLM {details.get('request_id', '')}."
    if event == "llm_response_consumed":
        return f"Ответ LLM {details.get('request_id', '')} обработан."
    if event == "context_compacted":
        return f"Сокращён контекст запроса: {_human_value(details.get('reason'))}."
    if event == "tool_call":
        return f"LLM вызвала инструмент {details.get('name', 'unknown')} ({details.get('call_id', '')}); полные аргументы сохранены в events.jsonl."
    if event == "tool_result":
        status = "успешно" if details.get("ok") else "с ошибкой"
        code = f", код={details['error_code']}" if details.get("error_code") else ""
        return f"Инструмент {details.get('name', 'unknown')} завершился {status}{code}."
    if event == "final_fit_started":
        return f"Начат финальный fit для {details.get('trial_id', 'лучшего кандидата')}."
    if event == "final_fit_completed":
        result = f"Финальный fit завершён: статус={details.get('status', 'unknown')}, время={_human_value(details.get('seconds'))} с"
        chain = _model_chain(details.get("model_details"))
        if chain:
            result += f"; фактическая цепочка модели: {chain}"
        parameters = _get_nested(details.get("model_details"), "parameters")
        if parameters:
            result += f". Фактические параметры: {_human_value(parameters)}"
        fallback = _get_nested(details.get("model_details"), "fallback_reason")
        if fallback:
            result += f"; причина перехода на запасную модель: {_human_value(fallback)}"
        if details.get("error_code"):
            result += f"; код ошибки={details['error_code']}"
        return result + "."
    if event == "run_completed":
        return (f"Запуск завершён: статус={details.get('status', 'unknown')}, остановка={details.get('stop_reason', 'unknown')}, "
                f"лучшая метрика={_human_value(details.get('best_metric'))}, выбран={_human_value(details.get('selected_trial'))}, "
                f"журнал решений={_human_value(details.get('decisions_path') or details.get('decision_log'))}.")
    if event == "run_interrupted":
        return "Запуск прерван пользователем или системой."
    if event == "run_error":
        return "Ошибка оркестрации; подробности доступны только в каноническом диагностическом журнале."
    return f"Событие {event}."


def _format_number(value: Any) -> str:
    number = _public_number(value)
    if number is None:
        return _human_value(value)
    if isinstance(number, int):
        return str(number)
    return f"{number:.4g}"


def _dataset_diagnosis_text(details: dict) -> str:
    pieces = []
    cv = details.get("adversarial_cv")
    if isinstance(cv, dict):
        primary = cv.get("roc_auc_mean")
        if primary is not None:
            item = f"adversarial ROC-AUC={_format_number(primary)}"
            if cv.get("roc_auc_std") is not None:
                item += f"±{_format_number(cv['roc_auc_std'])}"
            if cv.get("folds") is not None:
                item += f" ({_format_number(cv['folds'])} фолда)"
            seconds = cv.get("seconds", cv.get("elapsed_seconds"))
            if seconds is not None:
                item += f"; CV time={_format_number(seconds)} с"
            pieces.append(item)
        elif cv.get("reason"):
            pieces.append(f"adversarial CV {cv.get('status', 'skipped')}: {_human_value(cv['reason'])}")
        without = cv.get("without_temporal")
        if isinstance(without, dict) and without.get("roc_auc_mean") is not None:
            pieces.append(f"без temporal-признаков={_format_number(without['roc_auc_mean'])}")
    candidates = details.get("temporal_candidates", [])
    temporal = []
    if isinstance(candidates, list):
        for item in candidates[:3]:
            if not isinstance(item, dict):
                continue
            name = item.get("feature", "признак")
            train_max = item.get("train_max")
            test_min = item.get("test_min")
            interval = ""
            if train_max is not None or test_min is not None:
                interval = f" (train max={_human_value(train_max)}, test min={_human_value(test_min)})"
            temporal.append(f"{_human_value(name)}{interval}")
    if temporal:
        pieces.append("future-only=" + ", ".join(temporal))
    shifts = details.get("feature_shift", [])
    drift = []
    if isinstance(shifts, list):
        for item in shifts[:3]:
            if not isinstance(item, dict):
                continue
            name = _human_value(item.get("feature", "feature"))
            facts = []
            if item.get("mean_difference") is not None:
                facts.append(f"Δmean={_format_number(item['mean_difference'])}")
            if item.get("train_only_count") is not None or item.get("test_only_count") is not None:
                facts.append(f"train/test-only={_format_number(item.get('train_only_count', 0))}/{_format_number(item.get('test_only_count', 0))}")
            if item.get("train_missing_rate") is not None or item.get("test_missing_rate") is not None:
                facts.append(f"missing train/test={_format_number(item.get('train_missing_rate', 0))}/{_format_number(item.get('test_missing_rate', 0))}")
            if facts:
                drift.append(f"{name} ({', '.join(facts)})")
    if drift:
        pieces.append("drift=" + "; ".join(drift))
        if details.get("feature_shift_partial"):
            pieces[-1] += f"; показано 3 из {details.get('feature_shift_total', '?')} признаков"
    return "Диагностика train/test: " + ("; ".join(pieces) if pieces else "нет доступных сводных фактов") + "."


def _trial_diagnostics_text(diagnostics: Any) -> str:
    if not isinstance(diagnostics, dict):
        return ""
    pieces = []
    per_class = diagnostics.get("per_class")
    if isinstance(per_class, list) and per_class:
        rows = []
        for item in per_class[:3]:
            if not isinstance(item, dict):
                continue
            label = _human_value(item.get("label", "class"))
            metrics = []
            for key in ("f1", "recall", "support"):
                if item.get(key) is not None:
                    metrics.append(f"{key}={_format_number(item[key])}")
            rows.append(f"{label} ({', '.join(metrics)})")
        if rows:
            pieces.append("худшие классы: " + "; ".join(rows))
    confusion = diagnostics.get("confusion_matrix")
    if isinstance(confusion, dict):
        labels, matrix = confusion.get("labels"), confusion.get("matrix")
        errors = []
        if isinstance(labels, list) and isinstance(matrix, list):
            for row_index, row in enumerate(matrix):
                if not isinstance(row, list):
                    continue
                for column_index, count in enumerate(row):
                    if row_index != column_index and column_index < len(labels) and _public_number(count) and count > 0:
                        errors.append((float(count), labels[row_index], labels[column_index]))
        errors.sort(key=lambda item: (-item[0], str(item[1]), str(item[2])))
        if errors:
            pieces.append("ошибки матрицы: " + "; ".join(
                f"{_human_value(actual)}→{_human_value(predicted)}={_format_number(count)}"
                for count, actual, predicted in errors[:3]
            ))
    segments = diagnostics.get("segments")
    if isinstance(segments, list) and segments:
        rendered = []
        for item in segments[:2]:
            if not isinstance(item, dict):
                continue
            label = f"{_human_value(item.get('feature', 'feature'))}={_human_value(item.get('category'))}"
            metrics = []
            for key in ("rmse", "mae", "count"):
                if item.get(key) is not None:
                    metrics.append(f"{key}={_format_number(item[key])}")
            rendered.append(f"{label} ({', '.join(metrics)})")
        if rendered:
            pieces.append("сегменты с высокой ошибкой: " + "; ".join(rendered))
    validation = diagnostics.get("validation")
    if isinstance(validation, dict):
        metric = validation.get("metric_name", validation.get("metric"))
        score = validation.get("score")
        if metric is not None or score is not None:
            pieces.append(f"validation {metric or 'score'}={_format_number(score)}")
        elif validation.get("seconds") is not None:
            pieces.append(f"validation time={_format_number(validation['seconds'])} с")
    importance = diagnostics.get("feature_importance")
    if isinstance(importance, dict) and isinstance(importance.get("items"), list):
        top = [item for item in importance["items"][:3] if isinstance(item, dict)]
        if top:
            pieces.append("важность признаков: " + ", ".join(
                f"{_human_value(item.get('feature'))}={_format_number(item.get('importance'))}"
                for item in top
            ))
    return "Диагностика: " + "; ".join(pieces) + "." if pieces else ""


def _get_nested(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


def _model_chain(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    chain = value.get("estimator_chain")
    if isinstance(chain, list) and chain:
        return " → ".join(_bounded_text(item, 160) for item in chain[:12])
    model_class = value.get("class")
    return _bounded_text(model_class, 240) if model_class else ""


def _human_value(value: Any) -> str:
    if value is None:
        return "не указано"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return _bounded_text(value, 6000)


def _short_error_code(error: Any, status: Any) -> str | None:
    """Expose only a stable error class/code; raw diagnostics stay canonical."""
    if isinstance(error, dict) and isinstance(error.get("code"), str):
        return _bounded_text(error["code"], 100)
    if isinstance(error, str):
        match = re.match(r"\s*([A-Za-z][A-Za-z0-9_]{0,59})\s*:", error)
        if match:
            return match.group(1)
    if isinstance(status, str) and status and status != "ok":
        return _bounded_text(status, 100)
    return None


def _write_projection(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        temp_path = None
        _fsync_directory(path.parent)
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
