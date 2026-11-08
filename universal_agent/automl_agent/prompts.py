"""Bounded model context derived from the durable experiment ledger."""

from __future__ import annotations

import copy
import json
from pathlib import Path


SYSTEM_PROMPT = """You are an autonomous ML researcher. Solve the supplied task by improving a real model, not by claiming a score.
The host owns a fixed train/validation split, evaluates predictions independently, and keeps the best candidate. Validation labels are not available to you. Test template values are placeholders. Never use IDs as predictive features.
When runtime.parallel_trials=2 and at least two trial slots remain, prefer run_experiments with an array of two independent focused hypotheses; the host trains them concurrently, each using context.n_jobs threads. Use run_experiment when only one hypothesis or trial slot remains. Never base one candidate on unobserved results of its companion.
Use run_experiment to propose one hypothesis and its complete Python source. The source must define build_model(context), returning an UNFITTED sklearn-compatible estimator or Pipeline with fit(X, y) and predict(X). The host passes raw pandas feature DataFrames and performs training, validation and final refit. Include preprocessing for categorical columns, missing values and infinity. context contains task_type (regression/classification), metric, feature_columns, categorical_columns, classes, seed and n_jobs. Use these keys; do not hardcode this dataset's target or columns when avoidable.
Allowed imports: sklearn, catboost, lightgbm, xgboost, numpy, pandas, scipy, math, functools, collections, typing, itertools, operator, statistics, warnings, re. No reading/writing files, networking, package installation, shell commands, dynamic eval, or introspection. Native ML dependencies may vary; CatBoost and sklearn are the recommended starting points. Use <=context['n_jobs'] threads, CPU only, bounded model sizes (typically 150-600 boosting iterations). Do not fit or predict at module import. Avoid lambdas/local functions if a named transformer can be used. A model must preserve row order; classification predictions are original class labels.
Inspect the previous results and errors. The profile includes available_packages: use packages with a real version, never those marked unavailable. Prefer a focused improvement to the current working code before changing an entire model family. Repair the actual traceback before proposing more complexity. Form a concrete hypothesis about preprocessing, feature engineering, model family, regularization, imbalance or model parameters. Change something meaningful; duplicate code is rejected. You can return an entirely new estimator/pipeline, including custom sklearn transformers. Use the supplied baseline code as a working starting point and keep preprocessing correct. Write hypothesis and finish reason in Russian.
For every run_experiment include decision with concise PUBLIC explanations in Russian: model_family, observations (facts from the profile), rationale (why this change), alternatives (considered models), expected_effect (an untested expectation), risks, previous_result_assessment (what the last measured result implies). Explain decisions for the user; do not supply private internal reasoning. Never invent measurements or call an expectation a result. For generated candidates let errors propagate instead of silently switching models. A deliberate fallback must expose a public fallback_reason attribute; otherwise a successful run may not test the stated hypothesis. The host has already measured the family_comparison before research; use those scores rather than repeating unchanged baselines. The research_strategy is a starting shortlist, not a guaranteed winner. Class labels may be strings: count with np.unique(y, return_counts=True) or pandas rather than np.bincount on raw labels. Use experiment model_details to distinguish the actual fitted backend from the hypothesis.
Return a tool call, not a plan for the human to execute. For a provider without native tool calls, return exactly JSON {"tool":"run_experiment","arguments":{"hypothesis":"...","code":"complete Python source"}}. When at least one generated candidate has succeeded and further experiments are unhelpful, call finish_research with a reason. Never fabricate execution results, score, files, or successful recovery.
"""


COMPACT_SYSTEM_PROMPT = """You are an autonomous tabular ML researcher. Host owns fixed validation labels, independent scoring and best-model selection. Test template values are placeholders; never use IDs as features. Inspect measured family_comparison, model_details and actual tracebacks; fix errors before adding complexity. Do not invent results.
Call run_experiments with experiments:[{hypothesis,code,decision},...] for two independent candidates when runtime.parallel_trials=2 and slots remain; otherwise call run_experiment. code must define build_model(context) returning an UNFITTED estimator/Pipeline with fit(X,y), predict(X); classification uses original labels (possibly strings). Handle NaN, infinity, categorical/unknown values and preserve row order. Count string labels with np.unique(return_counts=True). CPU only, <=context.n_jobs threads, bounded sizes. No fit at import. Host refits on all train.
Only modelling imports: sklearn,catboost,lightgbm,xgboost,numpy,pandas,scipy,math,functools,collections,typing,itertools,operator,statistics,warnings,re. Only use installed packages. No files, networking, shell, package installation, dynamic eval or introspection. Propagate errors; deliberate fallback must expose fallback_reason. Use complete code, no duplicate attempts.
In decision give concise public Russian explanations: model_family, observations, rationale, alternatives, expected_effect, risks, previous_result_assessment. Expectations are untested. Do not supply private internal reasoning. Write hypothesis and finish reason in Russian. After a successful generated trial call finish_research(reason) when more experiments are unhelpful. Return tool calls, or JSON {"tool":"run_experiments","arguments":{"experiments":[...]}} / {"tool":"run_experiment","arguments":{"hypothesis":"...","code":"..."}}.
"""


RECIPE_SYSTEM_PROMPT = """You are a tabular ML researcher. The host owns validation, scoring and best-model selection. Validation labels are hidden; test targets are placeholders. Never invent results.
Use run_model_experiments with a native JSON array of {hypothesis,base_trial_id,parameters,preprocessing,fit_options,decision}. Return up to runtime.proposal_batch_size independent candidates, even when runtime.parallel_trials=1: proposal_batch_size is the proposal limit; parallel_trials is the simultaneous local-worker limit. Prefer two distinct candidates when slots remain. The host queues proposals and trains with context.n_jobs CPU threads. Do not assume companion results. Return tool calls or exact JSON {"tool":"run_model_experiments","arguments":{"experiments":[...]}}; no Python code.
Edit successful measured bases only. Choose using family metrics/runtime and actual parameters when shown. Omitted fields inherit the exact measured recipe, including CatBoost's actual learning_rate; include only changes. Duplicates are rejected and base_trial_id fixes the family. Use model_catalog for valid parameters/ranges and real task.feature_columns names. preprocessing.feature_ops is an ordered replacement array; learned statistics fit on training folds only.
Write concise Russian hypotheses from observed evidence; separate facts from untested expectations and omit private reasoning. Inspect validation plan, diagnostics, per-class/confusion errors, regression segments and feature importance (associative, not causal). Add features when evidence supports them. Never rename and retry a failed recipe.
For boosting, fit_options.early_stopping_rounds uses a host-created train-only split; outer folds refit at its selected tree count. OOF labels never enter fitting. Classification may use one supported weighting method: fit_options.sample_weight for LightGBM/XGBoost; class_weight for ExtraTrees/Linear; auto_class_weights for CatBoost. Never combine row and constructor weights; class_weights, constructor sample_weight and y_train are invalid. For imbalanced macro_f1, compare weights with the unweighted base while keeping its learning rate and tree cap. Regression must not use class weights. CatBoost handles missing values; test ExtraTrees/Linear missing indicators only when numeric missingness supports it. Compare bounded tree capacity with measured values. More iterations and much lower learning_rate can underfit; one worse score does not prove overfitting. Respect measured runtimes. Continue proposing measured experiments until the reserved deadline; stop only at reserve or when no experiment fits. Temporal CV scores estimate transfer. Host evaluates OOF blending and Macro F1 calibration separately.
"""


def _profile_facts(profile):
    keys = ("task_type", "metric", "train_shape", "test_shape", "feature_count", "categorical_columns", "target", "available_packages", "validation_plan")
    result = {key: profile[key] for key in keys if key in profile}
    drift = profile.get('dataset_diagnostics', {})
    if drift:
        result['dataset_diagnostics'] = {k: drift[k] for k in ('temporal_candidates', 'adversarial_cv') if k in drift}
        raw_shift = drift.get('feature_shift', drift.get('features', {}))
        if isinstance(raw_shift, dict):
            result['dataset_diagnostics']['feature_shift'] = dict(list(raw_shift.items())[:8])
        elif isinstance(raw_shift, list):
            result['dataset_diagnostics']['feature_shift'] = raw_shift[:8]
        for key in ('feature_shift_total', 'feature_shift_partial'):
            if key in drift:
                result['dataset_diagnostics'][key] = drift[key]
    features = profile.get("features", {})
    important = sorted(features, key=lambda name: (features[name].get("missing_fraction", 0), features[name].get("unique", 0)), reverse=True)
    result["feature_summary"] = [{"name": name, **{key: features[name][key] for key in ("dtype", "missing_fraction", "unique", "non_finite", "numeric") if key in features[name]}} for name in important[:12]]
    return result

def brief_details(details):
    result = {k:v for k,v in details.items() if k not in {'training','feature_importance','pipeline_steps'}}
    training = details.get('training', {})
    if training:
        result['training'] = {k:v for k,v in training.items() if k not in {'folds','trained_parameters'}}
        if 'folds' in training:
            result['training']['fold_best_iterations'] = [row.get('best_iteration') for row in training['folds']]
    return result


def brief_diagnostics(diagnostics):
    result = dict(diagnostics)
    if 'feature_importance' in result:
        result['feature_importance'] = dict(result['feature_importance'])
        result['feature_importance']['items'] = result['feature_importance'].get('items', [])[:12]
    if 'segments' in result:
        result['segments'] = result['segments'][:10]
    return result


def prompt_tool_result(value):
    if isinstance(value, list):
        return [prompt_tool_result(item) for item in value]
    if isinstance(value, dict):
        return {k:brief_details(v) if k == 'model_details' and isinstance(v, dict)
                else brief_diagnostics(v) if k == 'diagnostics' and isinstance(v, dict)
                else prompt_tool_result(v) for k,v in value.items()}
    return value


def _read_code(result, limit=18000):
    if result is None:
        return ""
    path = Path(result.artifact_dir) / "code.py"
    return path.read_text(encoding="utf-8")[:limit] if path.exists() else ""


def _clip_text(value, limit):
    value = str(value or "")
    if len(value) <= limit:
        return value
    if limit <= 32:
        return value[:limit]
    head = (limit - 24) * 2 // 3
    tail = limit - 24 - head
    return value[:head] + "\n[… контекст сокращён …]\n" + value[-tail:]


def _compact_recipe_diagnostics(diagnostics, level):
    """Keep decision-relevant diagnostics while dropping fold-level noise."""
    diagnostics = diagnostics or {}
    keep = {}
    validation = diagnostics.get("validation")
    if isinstance(validation, dict):
        fields = ("strategy", "coverage", "mean", "std", "fold_count", "warning", "warnings")
        keep["validation"] = {key: validation[key] for key in fields if key in validation}
        folds = validation.get("folds")
        if isinstance(folds, list):
            keep["validation"]["folds"] = [
                {key: row[key] for key in ("fold", "rows", "metric") if key in row}
                for row in folds[: min(5, max(2, 6 - level))] if isinstance(row, dict)
            ]
    if "per_class" in diagnostics:
        rows = diagnostics.get("per_class") or []
        keep["per_class"] = [
            {key: row[key] for key in ("label", "precision", "recall", "f1", "support", "predicted") if key in row}
            for row in rows[: max(10, 20 - level * 3)] if isinstance(row, dict)
        ]
        for key in ("total_classes", "partial"):
            if key in diagnostics:
                keep[key] = diagnostics[key]
    for key in ("total_segments", "segments_partial"):
        if key in diagnostics:
            keep[key] = diagnostics[key]
    confusion = diagnostics.get("confusion_matrix")
    if isinstance(confusion, dict):
        labels = confusion.get("labels")
        matrix = confusion.get("matrix")
        if isinstance(labels, list) and isinstance(matrix, list):
            size = min(10, len(labels), len(matrix))
            rows = matrix[:size]
            if all(isinstance(row, list) and len(row) >= size for row in rows):
                keep["confusion_matrix"] = {
                    "labels": labels[:size],
                    "matrix": [row[:size] for row in rows],
                }
                for key in ("total_classes", "partial", "selection"):
                    if key in confusion:
                        keep["confusion_matrix"][key] = confusion[key]
                if len(labels) > size or len(matrix) > size:
                    keep["confusion_matrix"]["partial"] = True
    if isinstance(diagnostics.get("partial"), bool):
        keep["partial"] = diagnostics["partial"]
    if len(diagnostics.get('per_class') or []) > len(keep.get('per_class', [])):
        keep['partial'] = True
    if "segments" in diagnostics:
        keep["segments"] = (diagnostics.get("segments") or [])[:max(2, 8 - level)]
    importance = diagnostics.get("feature_importance")
    if isinstance(importance, dict):
        keep["feature_importance"] = {
            key: importance[key] for key in
            ("method", "interpretation", "status", "feature_count", "total_items", "partial") if key in importance
        }
        items = importance.get("items") or []
        keep["feature_importance"]["items"] = items[:max(3, 10 - level * 2)]
    return keep


def _recipe_catalog_context(task_type, profile, results, maximize=True):
    from .recipes import recipe_catalog

    catalog = recipe_catalog(task_type)
    packages = profile.get("available_packages", {})
    package_names = {
        "CatBoost": "catboost", "LightGBM": "lightgbm", "XGBoost": "xgboost",
        "ExtraTrees": "scikit-learn", "Linear": "scikit-learn",
    }
    for section in ("families", "boosting_families"):
        families = catalog.get(section, {})
        for family in list(families):
            package = package_names.get(family)
            version = packages.get(package) if package else None
            if version is not None and str(version).lower() in {"unavailable", "none", "not installed", "false"}:
                families.pop(family, None)

    # Field descriptions already state the host contract. Keep the actual supported
    # schema verbatim so ranges, null defaults and enum options remain authoritative.
    fit_options = catalog.get("fit_options", {})
    catalog["fit_options"] = {"options": fit_options.get("options", {})}
    catalog.pop("host_controlled", None)
    catalog.pop("unsupported", None)

    best_by_family = {}
    for result in results:
        recipe = getattr(result, "recipe", None)
        if result.status != "ok" or not isinstance(recipe, dict):
            continue
        family = recipe.get("family")
        previous = best_by_family.get(family)
        better = previous is None or (result.metric is not None and
                                       (previous["metric"] is None or
                                        (result.metric > previous["metric"] if maximize else result.metric < previous["metric"])))
        if better:
            details = result.model_details or {}
            best_by_family[family] = {
                "family": family, "trial_id": result.trial_id, "metric": result.metric,
                "parameters": details.get("parameters", {}),
                "recipe": copy.deepcopy(recipe),
            }
    if best_by_family:
        catalog["measured_bases"] = list(best_by_family.values())
        for section in ('families','boosting_families'):
            catalog[section] = {name:schema for name,schema in catalog.get(section,{}).items()
                                if name in best_by_family}
    return catalog


def _budget_catalog(catalog, keep_families, drop_defaults):
    compact = copy.deepcopy(catalog)
    if keep_families:
        for section in ("families", "boosting_families"):
            families = compact.get(section, {})
            for family in list(families):
                if family not in keep_families:
                    families.pop(family, None)
    for section in ("families", "boosting_families"):
        for family in compact.get(section, {}).values():
            for schema in family.get("parameters", {}).values():
                if drop_defaults:
                    schema.pop("default", None)
            for schema in family.get("preprocessing", {}).values():
                if drop_defaults:
                    schema.pop("default", None)
    fit_options = compact.get("fit_options", {})
    for schema in fit_options.get("options", {}).values():
        if drop_defaults:
            schema.pop("default", None)
    return compact


def _compact_measured_bases(catalog):
    """Keep base IDs actionable without repeating full recipes and fit parameters."""
    bases = catalog.get("measured_bases", [])
    catalog["measured_bases"] = [
        {key: row[key] for key in ("family", "trial_id", "metric") if key in row}
        for row in bases if isinstance(row, dict)
    ]
    return catalog


def _recipe_profile_context(profile, level):
    facts = _profile_facts(profile)
    facts.pop("feature_summary", None)
    summary = _profile_facts(profile).get("feature_summary", [])
    summary_count = max(0, (12, 8, 6, 4, 2, 0)[min(level, 5)])
    if summary_count:
        facts["feature_summary"] = summary[:summary_count]
    if isinstance(facts.get("categorical_columns"), list):
        category_cap = (20, 16, 12, 10, 8, 5)[min(level, 5)]
        facts["categorical_column_count"] = len(facts["categorical_columns"])
        facts["categorical_columns"] = facts["categorical_columns"][:category_cap]

    target = facts.get("target")
    if isinstance(target, dict):
        # Target counts are central for imbalanced classification and stay exact.
        target = {key: target[key] for key in ("name", "dtype", "unique", "missing", "numeric", "class_counts", "counts") if key in target}
        facts["target"] = target

    plan = facts.get("validation_plan")
    if isinstance(plan, dict):
        fields = ("strategy", "fold_count", "coverage", "time_column", "warnings", "groups", "repeated_entity_count")
        facts["validation_plan"] = {key: plan[key] for key in fields if key in plan}
        if isinstance(facts["validation_plan"].get("warnings"), list):
            facts["validation_plan"]["warnings"] = facts["validation_plan"]["warnings"][:3]

    diagnostics = facts.get("dataset_diagnostics")
    if isinstance(diagnostics, dict):
        temporal = diagnostics.get("temporal_candidates", [])
        diagnostics["temporal_candidates"] = temporal[:max(1, 4 - level)] if isinstance(temporal, list) else temporal
        adversarial = diagnostics.get("adversarial_cv")
        if isinstance(adversarial, dict):
            def small_auc(value):
                if not isinstance(value, dict):
                    return None
                fields = ("status", "metric", "reason", "roc_auc_mean", "roc_auc_std", "folds", "seconds", "elapsed_seconds")
                return {key: value[key] for key in fields if key in value}

            compact_auc = small_auc(adversarial)
            without_temporal = small_auc(adversarial.get("without_temporal"))
            if isinstance(compact_auc, dict) and isinstance(without_temporal, dict):
                compact_auc["without_temporal"] = without_temporal
            diagnostics["adversarial_cv"] = compact_auc or {}
        shift = diagnostics.get("feature_shift", {})
        shift_rows = list(shift.items()) if isinstance(shift, dict) else [
            (row.get("feature", f"feature_{index}"), row)
            for index, row in enumerate(shift) if isinstance(row, dict)
        ] if isinstance(shift, list) else []
        compact_shift = []
        for name, value in shift_rows:
            if not isinstance(value, dict):
                continue
            item = {}
            for key in ("kind", "train_missing_rate", "test_missing_rate", "train_mean", "test_mean",
                        "train_std", "test_std", "train_min", "train_max", "test_min", "test_max",
                        "mean_difference", "train_unique_count", "test_unique_count", "train_only_count", "test_only_count"):
                if key in value:
                    item[key] = value[key]
            category_shift = value.get("category_shift")
            if isinstance(category_shift, dict):
                compact_category = {key: category_shift[key] for key in ("train_only_count", "test_only_count") if key in category_shift}
                for key in ("train_only_top", "test_only_top"):
                    rows = category_shift.get(key)
                    if isinstance(rows, list):
                        compact_category[key] = [
                            {k: (_clip_text(row[k], 80) if isinstance(row.get(k), str) else row[k])
                             for k in ("value", "count") if k in row}
                            for row in rows[:2] if isinstance(row, dict)
                        ]
                if compact_category:
                    item["category_shift"] = compact_category
            if item:
                score = max(
                    abs(float(value.get("mean_difference", 0) or 0)),
                    abs(float(value.get("train_missing_rate", 0) or 0) - float(value.get("test_missing_rate", 0) or 0)),
                    float(item.get("train_only_count", 0) or 0), float(item.get("test_only_count", 0) or 0),
                    float((item.get("category_shift") or {}).get("train_only_count", 0) or 0),
                    float((item.get("category_shift") or {}).get("test_only_count", 0) or 0),
                )
                compact_shift.append((score, str(name), item))
        compact_shift.sort(key=lambda row: (-row[0], row[1]))
        if compact_shift:
            shift_limit = max(2, 8 - level)
            diagnostics["feature_shift"] = {name: item for _, name, item in compact_shift[:shift_limit]}
        else:
            diagnostics["feature_shift"] = {}
    return facts


def _recipe_task_context(task, runtime_context, level, compact=False):
    task_data = task.model_context(runtime_context.get("n_jobs", 4))
    feature_columns = task_data.get("feature_columns", [])
    categorical = task_data.get("categorical_columns", [])
    feature_cap = (200, 80, 40, 20, 0, 0)[min(level, 5)]
    category_cap = (40, 24, 16, 10, 0, 0)[min(level, 5)]
    if compact:
        # Keep useful names for feature_ops while bounding large public schemas.
        feature_cap = 100 if len(feature_columns) > 100 else len(feature_columns)
        category_cap = min(24, len(categorical))
    task_data["feature_count"] = len(feature_columns)
    task_data["categorical_count"] = len(categorical)
    if len(feature_columns) > feature_cap:
        task_data["feature_columns"] = feature_columns[:feature_cap]
        task_data["feature_columns_truncated"] = True
    if len(categorical) > category_cap:
        task_data["categorical_columns"] = categorical[:category_cap]
        task_data["categorical_columns_truncated"] = True
    return task_data


def _build_recipe_messages(task, profile, results, best, runtime_context, feedback, context_chars, compact):
    """Construct a bounded, recipe-only prompt without changing the code-mode path."""
    if compact:
        diagnostic_rows = [len(result.diagnostics.get("per_class") or [])
                           for result in results if isinstance(result.diagnostics, dict)]
        compact_target = 12000 if max(diagnostic_rows, default=0) > 20 else 9000
        limit = max(5000, min(compact_target, context_chars // 2))
        hard_limit = max(5000, min(12000, context_chars // 2))
    else:
        limit = max(12000, context_chars)
        hard_limit = limit
    latest = results[-1] if results else None
    ledger_results = list(results[-12:])
    if best is not None and not any(result.trial_id == best.trial_id for result in ledger_results):
        ledger_results.insert(0, best)
    if compact:
        selected_ids = set()
        for result in (best, latest, *results[-1:]):
            if result is not None:
                selected_ids.add(result.trial_id)
        ledger_results = [row for row in results if row.trial_id in selected_ids]
        if best is not None and not any(row.trial_id == best.trial_id for row in ledger_results):
            ledger_results.insert(0, best)
    description = task.description or ""
    catalog = _recipe_catalog_context(task.task_type, profile, results, task.maximize)
    constraints = (
        "REGRESSION: numeric target, optimize " + task.metric +
        ". Feature categories are NOT target classes. Never use class balancing or classification weights."
        if task.task_type == "regression" else
        "CLASSIFICATION: optimize " + task.metric +
        ". Class weighting applies to target labels only; inspect per-class metrics and confusion errors."
    )
    pipeline_contract = {
        "CatBoost": "Numeric NaN is preserved; categories are cleaned to strings and handled natively.",
        "ExtraTrees/LightGBM/XGBoost": "Numeric median imputation; categorical missing sentinel and sparse one-hot. Indicators are optional.",
        "Linear": "Median imputation, sparse categorical one-hot and uncentered numeric scaling.",
    }

    def make_content(level):
        description_limit = ((9000, 6500, 4500, 3000, 1800, 500) if not compact
                             else (1800, 1400, 1000, 500, 350, 250))[min(level, 5)]
        description_value = _clip_text(description, description_limit)
        feedback_limit = (5000, 3500, 2200, 1400, 800, 400)[min(level, 5)] if compact else 5000
        hypothesis_limit = (450, 350, 260, 180, 100, 28)[min(level, 5)]
        experiments = []
        for result in ledger_results:
            row = {"trial_id": result.trial_id, "status": result.status,
                   "hypothesis": _clip_text(result.hypothesis, hypothesis_limit), "metric": result.metric}
            if best is not None and result.trial_id == best.trial_id:
                row["recipe"] = result.recipe
            experiments.append(row)

        def experiment_detail(result, include_error=False):
            if result is None:
                return None
            detail = {
                "trial_id": result.trial_id, "status": result.status,
                "hypothesis": _clip_text(result.hypothesis, hypothesis_limit), "metric": result.metric,
                "recipe": result.recipe,
                "model_details": {
                    key: result.model_details[key] for key in ("class", "parameters", "parameters_source")
                    if key in (result.model_details or {})
                },
                "diagnostics": _compact_recipe_diagnostics(result.diagnostics, level),
            }
            if include_error and result.error:
                error_limit = (5000, 4000, 1200, 1000, 800, 600)[min(level, 5)] if compact else (5000, 4000, 3200, 2600, 1600, 950)[min(level, 5)]
                detail["error"] = _clip_text(result.error, error_limit)
            return detail

        best_summary = experiment_detail(best)
        latest_summary = experiment_detail(latest, include_error=bool(latest and latest.status != "ok"))
        if best is latest and latest_summary is not None:
            latest_summary = {"trial_id": best.trial_id, "status": best.status, "same_as_best": True}
            if best.status != "ok" and best.error:
                latest_summary["error"] = _clip_text(best.error, (5000, 4000, 3200, 2600, 1600, 950)[min(level, 5)])

        comparison = profile.get("family_comparison", {}) or {}
        comparison_rows = []
        for row in comparison.get("trials", []):
            if not isinstance(row, dict):
                continue
            compact_row = {key: row[key] for key in ("family", "trial_id", "status", "metric", "seconds") if key in row}
            details = row.get("model_details", {})
            if isinstance(details, dict):
                compact_row["model_details"] = {key: details[key] for key in ("class", "parameters", "parameters_source") if key in details}
            comparison_rows.append(compact_row)
        strategy = profile.get("research_strategy", {}) or {}
        strategy_facts = strategy.get("facts", {}) if level < 3 else {}
        shortlist = strategy.get("candidate_shortlist", [])
        strategy_compact = {
            "facts": strategy_facts,
            "candidate_shortlist": [{"model_family": item.get("model_family")}
                                    for item in shortlist[:5] if isinstance(item, dict) and item.get("model_family")],
        }
        content = {
            "task_description": description_value,
            "task": _recipe_task_context(task, runtime_context, level, compact),
            "data_profile": _recipe_profile_context(profile, level),
            "task_constraints": constraints,
            "model_catalog": catalog,
            "pipeline_contract": pipeline_contract,
            "research_strategy": strategy_compact,
            "experiments": experiments,
            "best_experiment": best_summary,
            "latest_experiment": latest_summary,
            "family_comparison": {"trials": comparison_rows, "selected_trial": comparison.get("selected_trial")},
            "best_code": "", "last_failed_code": "", "runtime": runtime_context,
            "previous_tool_feedback": _clip_text(feedback, feedback_limit),
        }
        return content

    # Start with informative context, then drop secondary details in fixed stages.
    content = None
    encoded = ""
    for level in range(3 if compact else 0, 6):
        content = make_content(level)
        encoded = json.dumps(content, ensure_ascii=False, allow_nan=False, default=str)
        if len(encoded) <= limit:
            break

    if len(encoded) > limit:
        keep_families = set()
        if best and isinstance(best.recipe, dict) and best.recipe.get("family"):
            keep_families.add(best.recipe["family"])
        keep_families.update(row.get("family") for row in catalog.get("measured_bases", []) if row.get("family"))
        recent = content["experiments"][-8:]
        if best is not None and not any(row.get("trial_id") == best.trial_id for row in recent):
            best_history = next((row for row in content["experiments"] if row.get("trial_id") == best.trial_id), None)
            if best_history:
                recent = [best_history, *recent]
        core = {
            "task": {key: value for key, value in content["task"].items()
            if key in {"task_type", "metric", "target", "classes", "positive_class", "feature_columns",
                       "feature_columns_truncated", "categorical_columns", "seed", "n_jobs", "feature_count",
                       "categorical_count"}},
            "task_constraints": content["task_constraints"],
            "data_profile": content["data_profile"],
            "model_catalog": _compact_measured_bases(
                _budget_catalog(catalog, keep_families, drop_defaults=False)
            ),
            "best_experiment": content["best_experiment"],
            "latest_experiment": content["latest_experiment"],
            "experiments": recent,
            "family_comparison": content["family_comparison"],
            "last_failed_code": "",
            "runtime": {key: runtime_context[key] for key in ("n_jobs", "parallel_trials", "proposal_batch_size", "experiment_mode", "remaining_seconds") if key in runtime_context},
        }

        def encode_core():
            return json.dumps(core, ensure_ascii=False, allow_nan=False, default=str)

        encoded = encode_core()
        if len(encoded) > limit:
            # Trim optional context first. Keep a valid catalog for measured families,
            # the complete validation plan, exact best recipe/parameters and last error.
            core.pop("family_comparison", None)
            core["model_catalog"] = _compact_measured_bases(
                _budget_catalog(catalog, keep_families, drop_defaults=True)
            )
            core["experiments"] = [
                {key: row[key] for key in ("trial_id", "status", "metric") if key in row}
                for row in recent
            ]
            core["latest_experiment"] = {
                key: content["latest_experiment"][key]
                for key in ("trial_id", "status", "metric", "same_as_best", "error", "diagnostics")
                if key in (content["latest_experiment"] or {})
            }
            if "error" in core["latest_experiment"]:
                core["latest_experiment"]["error"] = _clip_text(core["latest_experiment"]["error"], 700)
            encoded = encode_core()
        if len(encoded) > limit:
            # Keep a small, reviewable history plus best and latest while retaining
            # the native catalog schema and validation plan rather than serializing
            # model data as strings or silently exceeding the selected budget.
            core["experiments"] = [
                {key: row[key] for key in ("trial_id", "status", "metric") if key in row}
                for row in recent[-4:]
            ]
            core.pop("family_comparison", None)
            core["model_catalog"] = _compact_measured_bases(
                _budget_catalog(catalog, keep_families, drop_defaults=True)
            )
            core["data_profile"]["dataset_diagnostics"] = {
                key: value for key, value in core["data_profile"].get("dataset_diagnostics", {}).items()
                if key == "adversarial_cv"
            }
            core["task"]["classes"] = core["task"].get("classes", [])[:10]
            core["latest_experiment"] = {
                key: core["latest_experiment"][key]
                for key in ("trial_id", "status", "metric", "same_as_best", "error")
                if key in core["latest_experiment"]
            }
            encoded = encode_core()
    if len(encoded) > hard_limit:
        # Pathological profile/class-count/recipe payloads must fail explicitly;
        # never send a truncated recipe or malformed/empty model catalog.
        raise ValueError(
            f"Recipe prompt essentials require {len(encoded)} characters, above the {hard_limit}-character budget."
        )

    system_prompt = RECIPE_SYSTEM_PROMPT
    active_families={row['family'] for row in catalog.get('measured_bases',[])}
    if task.task_type == 'classification':
        weight_guidance=[]
        if 'CatBoost' in active_families:
            weight_guidance.append('CatBoost: parameters.auto_class_weights="Balanced" or "SqrtBalanced".')
        if active_families & {'ExtraTrees','Linear'}:
            weight_guidance.append('ExtraTrees/Linear: parameters.class_weight="balanced".')
        if active_families & {'LightGBM','XGBoost'}:
            weight_guidance.append('LightGBM/XGBoost: fit_options.sample_weight="sqrt_balanced" or "balanced"; never auto_class_weights or constructor sample_weight.')
        system_prompt='\n'.join(('For imbalanced classification and macro_f1, compare untested weight settings with the measured unweighted base, retaining its learning rate and tree cap. '+ ' '.join(weight_guidance))
            if line.startswith('For imbalanced classification and macro_f1') else line
            for line in system_prompt.splitlines())
    system_prompt += '\nThe model family is fixed by base_trial_id: edit a successful measured base from model_catalog.measured_bases; a different name in hypothesis cannot switch the family. Encode every intended change in parameters/preprocessing/fit_options. A hypothesis alone never changes configuration. Omit unchanged fields instead of copying base defaults.\n'
    if task.task_type == "regression":
        system_prompt = "\n".join(line for line in system_prompt.splitlines()
                                  if not line.startswith("For imbalanced classification and macro_f1"))
        system_prompt = "This task is REGRESSION with a numeric target. Do NOT use any class balancing or classification weights. Categorical feature levels are NOT target classes. Native parameter metadata can include irrelevant weight placeholders; the regression model_catalog is authoritative.\n" + system_prompt
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": encoded}]


def build_messages(task, profile, results, best, runtime_context, feedback="", context_chars=48000, compact=False):
    if runtime_context.get("experiment_mode") == "recipe":
        return _build_recipe_messages(task, profile, results, best, runtime_context, feedback, context_chars, compact)
    ledger = [{"trial_id": r.trial_id, "status": r.status, "hypothesis": r.hypothesis[:1500], "metric": r.metric,
               "seconds": round(r.seconds, 2), "model_details": r.model_details, "recipe": getattr(r, "recipe", None),
               "parent_trial_id": getattr(r, "parent_trial_id", None), "diagnostics": getattr(r, "diagnostics", {}), "error": (r.error or "")[-2500:]} for r in results]
    for item in ledger:
        item['model_details'] = brief_details(item['model_details'])
        item['diagnostics'] = brief_diagnostics(item['diagnostics'])
    ledger = ledger[-8:]
    failed = results[-1] if results and results[-1].status != "ok" else None
    description = task.description
    if len(description) > 9000:
        description = description[:5500] + "\n[description cropped]\n" + description[-3500:]
    content = {"task_description": description, "task": task.model_context(runtime_context["n_jobs"]),
               "data_profile": {k: v for k, v in profile.items() if k not in {"research_strategy", "family_comparison"}},
               "research_strategy": profile.get("research_strategy", {}), "experiments": ledger, "best_code": _read_code(best),
               "last_failed_code": _read_code(failed, 10000), "runtime": runtime_context,
               "previous_tool_feedback": feedback[-5000:], "family_comparison": profile.get("family_comparison", {})}
    content['best_experiment'] = None
    if best and not any(item['trial_id'] == best.trial_id for item in ledger):
        content['best_experiment'] = {'trial_id':best.trial_id,'metric':best.metric,'recipe':best.recipe,
                                      'model_details':brief_details(best.model_details),'diagnostics':brief_diagnostics(best.diagnostics)}
    content['family_comparison'] = {'trials':[{k:row.get(k) for k in ('family','trial_id','status','metric','seconds')}
                                          | {'model_details':brief_details(row.get('model_details',{}))} for row in profile.get('family_comparison',{}).get('trials',[])],
                                     'selected_trial':profile.get('family_comparison',{}).get('selected_trial')}
    limit = max(5000, context_chars // 2) if compact else max(5000, context_chars)
    recipe_mode = runtime_context.get("experiment_mode") == "recipe"
    if recipe_mode:
        from .recipes import recipe_catalog
        content["model_catalog"] = recipe_catalog(task.task_type)
        content['pipeline_contract'] = {'CatBoost':'Numeric NaN values are preserved; categories are cleaned to strings and handled natively.',
            'ExtraTrees/LightGBM/XGBoost':'Numeric median imputation; categorical missing sentinel and sparse one-hot (min_frequency=25, max_categories=64); indicators are optional. The native estimator never sees raw NaN after imputation.',
            'Linear':'Median imputation, sparse categorical one-hot and uncentered numeric scaling.'}
        content['task_constraints'] = (
            'REGRESSION: numeric target, optimize ' + task.metric + '. Feature categories such as building class or facility type are NOT target classes. Never use auto_class_weights, class_weight or balanced sample_weight for regression.'
            if task.task_type == 'regression' else
            'CLASSIFICATION: optimize ' + task.metric + '. Class weighting applies to target classes only; inspect per-class metrics and the confusion matrix.'
        )
        content["best_code"] = ""  # the host preserves the verified pipeline
        content["last_failed_code"] = ""
        content["data_profile"] = _profile_facts(profile)
    if compact:
        # Actually compact auxiliary context even when it is below context_chars.
        # Keep the most recent balanced assistant/tool group in the caller so a
        # failed experiment remains actionable under a tight token budget.
        content["data_profile"] = _profile_facts(profile)
        comparison = profile.get("family_comparison", {})
        content["family_comparison"] = {"trials": [{k: row.get(k) for k in ("family", "trial_id", "status", "metric", "seconds", "recipe")} | {"model_details":brief_details(row.get("model_details",{}))} for row in comparison.get("trials", [])], "selected_trial": comparison.get("selected_trial")}
        strategy = profile.get("research_strategy", {})
        content["research_strategy"] = {
            "facts": strategy.get("facts", {}),
            "candidate_shortlist": [{"model_family": candidate["model_family"]}
                                    for candidate in strategy.get("candidate_shortlist", [])],
        }
    if compact:
        def measured_details(details):
            return {key: details[key] for key in ("class", "parameters", "parameters_source") if key in details}
        for index, entry in enumerate(content["experiments"]):
            entry["model_details"] = measured_details(entry["model_details"])
            entry["error"] = entry["error"][-(2000 if index == len(content["experiments"]) - 1 else 500):]
        for row in content.get("family_comparison", {}).get("trials", []):
            if "model_details" in row:
                row["model_details"] = measured_details(row["model_details"])
        content["previous_tool_feedback"] = feedback[-800:] if recipe_mode else ""
        if not recipe_mode:
            for row in content["family_comparison"]["trials"]:
                row.pop("model_details", None)  # measured details already occur in the experiment ledger
    # Preserve model code and the exact ledger; trim auxiliary profiling first.
    encoded = json.dumps(content, ensure_ascii=False, allow_nan=False, default=str)
    if len(encoded) > limit:
        content["data_profile"] = _profile_facts(profile)
        content["data_profile"]["feature_summary"] = content["data_profile"]["feature_summary"][:4]
        content["task_description"] = description[:3000] + "\n" + description[-1000:]
        content["last_failed_code"] = "" if recipe_mode else _read_code(failed, 5000)
        encoded = json.dumps(content, ensure_ascii=False, allow_nan=False, default=str)
    if len(encoded) > limit:
        content["best_code"] = "" if recipe_mode else _read_code(best, max(1500, limit // 3))
        content["experiments"] = ledger[-6:]
        content["last_failed_code"] = "" if recipe_mode else _read_code(failed, 1500)
        encoded = json.dumps(content, ensure_ascii=False, allow_nan=False, default=str)
    system_prompt = RECIPE_SYSTEM_PROMPT if recipe_mode else (COMPACT_SYSTEM_PROMPT if compact else SYSTEM_PROMPT)
    if recipe_mode and task.task_type == 'regression':
        system_prompt = '\n'.join(line for line in system_prompt.splitlines()
                                  if not line.startswith('For imbalanced classification and macro_f1'))
        system_prompt = 'This task is REGRESSION with a numeric target. Do NOT use any class balancing or classification weights. Categorical feature levels are NOT target classes. Native parameter metadata can include irrelevant weight placeholders; the regression model_catalog is authoritative.\n' + system_prompt
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": encoded}]
