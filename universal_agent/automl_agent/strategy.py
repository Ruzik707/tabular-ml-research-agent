"""Data-aware starting recommendations and independently verified assessments."""

from __future__ import annotations

import math


DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "model_family": {"type": "string", "maxLength": 200},
        "observations": {"type": "array", "maxItems": 10, "items": {"type": "string", "maxLength": 700}},
        "rationale": {"type": "string", "maxLength": 2500},
        "alternatives": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 700}},
        "expected_effect": {"type": "string", "maxLength": 1200},
        "risks": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 700}},
        "previous_result_assessment": {"type": "string", "maxLength": 1500},
    },
    "additionalProperties": False,
}


def plan_research(task, profile, config):
    """Explain a small candidate shortlist, not a predicted benchmark winner.

    Only pre-existing profile facts and installed package versions are used.
    Recommendations cannot change the host split, score or budget enforcement.
    """
    n_rows = profile.get("train_shape", [0])[0]
    features = profile.get("features", {})
    categories = task.categorical_columns
    missing = [name for name, info in features.items() if info.get("missing", 0) > 0]
    non_finite = [name for name, info in features.items() if info.get("non_finite", 0) > 0]
    high_cardinality = [name for name in categories if features.get(name, {}).get("unique", 0) > 100]
    counts = list(profile.get("target", {}).get("class_counts", {}).values())
    imbalance = round(max(counts) / min(counts), 3) if counts and min(counts) > 0 else None
    packages = profile.get("available_packages", {})
    has_catboost = packages.get("catboost", "unavailable") != "unavailable"
    facts = {"train_rows": n_rows, "feature_count": len(task.feature_columns), "categorical_count": len(categories),
             "features_with_missing": len(missing), "missing_features": missing[:12],
             "features_with_infinity": len(non_finite), "high_cardinality_categories": high_cardinality[:12],
             "class_imbalance_ratio": imbalance,
             "class_counts_partial": profile.get("target", {}).get("unique", 0) > len(counts) if counts else False}
    shortlist = []
    if has_catboost:
        shortlist.append({"model_family": "CatBoost", "reason": "Начальный baseline: нативная обработка категорий; числовые пропуски сохраняются, категориальные заполняются строкой."
                          if categories else "Начальный baseline: градиентный бустинг для числовой таблицы; преимущество проверяется на holdout."})
    shortlist.append({"model_family": "ExtraTrees", "reason": "Независимый ансамблевый кандидат и запасной backend baseline; требует корректной обработки пропусков и категорий."})
    if len(task.feature_columns) >= 100:
        shortlist.append({"model_family": "Linear", "reason": "Много признаков: проверить линейную модель с масштабированием и кодированием как дешёвую альтернативу."})
    else:
        shortlist.append({"model_family": "Linear", "reason": "Проверить простую линейную зависимость после масштабирования чисел и разреженного кодирования категорий."})
    priorities = ["Сначала измерить baseline на фиксированном holdout; затем менять одну обоснованную гипотезу за опыт."]
    if missing or non_finite:
        priorities.append("Проверить заполнение пропусков, infinity и индикаторы пропусков без утечки target.")
    if high_cardinality:
        priorities.append("Категории высокой кардинальности: сравнить нативный CatBoost с кодированием, контролировать память one-hot.")
    if imbalance is not None and imbalance >= 3:
        priorities.append("Обнаружен дисбаланс классов: проверить CatBoost auto_class_weights=Balanced/SqrtBalanced или sklearn class_weight=balanced при том же learning_rate и метрики классов; сравнивать по целевой метрике, без обещания улучшения.")
    priorities.append("После каждого опыта сопоставить качество и время; ошибки исправлять по traceback, победителя выбирать только по host-метрике.")
    return {"facts": facts, "candidate_shortlist": shortlist, "priorities": priorities,
            "selection_rule": f"{task.metric}: {'максимум' if task.maximize else 'минимум'} на одном split; при равенстве выбирается более ранний trial_id. Гарантии победы семейства нет.",
            "parallel_trials": getattr(config, "parallel_trials", 2),
            "budget": {"total_seconds": config.time_budget, "finalization_reserve_seconds": config.reserve_seconds,
                       "trial_timeout_seconds": config.trial_timeout, "threads_per_trial": config.n_jobs},
            "validation_caveat": "Проверить требования task.md к времени/группам: по умолчанию host использует CV и выбирает forward periods при подтверждённом временном сдвиге; изучите validation_plan."}


def baseline_decision(strategy):
    """Public explanation of the deterministic baseline policy."""
    candidates = strategy["candidate_shortlist"]
    return {"model_family": candidates[0]["model_family"],
            "observations": [f"Объектов: {strategy['facts']['train_rows']}; признаков: {strategy['facts']['feature_count']}; категориальных: {strategy['facts']['categorical_count']}"],
            "rationale": candidates[0]["reason"],
            "alternatives": [f"{item['model_family']}: {item['reason']}" for item in candidates[1:]],
            "expected_effect": "Получить проверенный стартовый результат; величина метрики заранее неизвестна.",
            "risks": ["Импорт или обучение CatBoost могут завершиться ошибкой; baseline тогда использует ExtraTrees. Фактическая модель фиксируется worker отдельно."],
            "previous_result_assessment": "Первый опыт запуска; результатов для сравнения ещё нет."}


def assess_experiment(result, previous_best, maximize, metric_name):
    """Describe exactly the host's selection rule, without causal claims."""
    previous = previous_best.metric if previous_best else None
    valid = result.status == "ok" and result.metric is not None and math.isfinite(result.metric)
    delta = result.metric - previous if valid and previous is not None else None
    earlier = (previous_best is not None and result.trial_id.startswith("trial_") and previous_best.trial_id.startswith("trial_")
               and result.trial_id[6:].isdigit() and previous_best.trial_id[6:].isdigit()
               and int(result.trial_id[6:]) < int(previous_best.trial_id[6:]))
    selected = valid and (previous is None or (result.metric > previous if maximize else result.metric < previous)
                          or (delta == 0 and earlier))
    if not valid:
        reason = f"Опыт завершён со статусом {result.status}; результат не участвует в выборе. {result.error or ''}".strip()
    elif previous is None:
        reason = f"Первый валидный результат {metric_name}={result.metric}; выбран как текущий лучший."
    elif delta == 0:
        reason = "Метрика равна предыдущей лучшей; выбран более ранний trial_id." if selected else "Метрика равна предыдущей лучшей; сохраняется более ранний кандидат."
    else:
        direction = "больше" if maximize else "меньше"
        reason = f"{metric_name}: {result.metric} против {previous}; лучше — {direction}. " + ("Новый кандидат выбран." if selected else "Предыдущий кандидат сохранён.")
    return {"source": "harness", "trial_id": result.trial_id, "metric_name": metric_name,
            "metric": result.metric if valid else None, "previous_best": previous,
            "maximize": maximize, "selected": selected, "delta": delta, "reason": reason,
            "model_details": getattr(result, "model_details", {})}
