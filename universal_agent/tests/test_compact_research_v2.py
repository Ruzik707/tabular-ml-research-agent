import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from automl_agent.contracts import ExperimentResult, TaskSpec
from automl_agent.prompts import build_messages
from automl_agent.providers import OpenAICompatibleProvider, ProviderError


def _case():
    task = TaskSpec(
        task_path="task.md", train_path="train.csv", test_path="test.csv", target="label",
        task_type="classification", metric="f1", maximize=True,
        feature_columns=["amount", "region", "account_age"], id_columns=["row_id"],
        categorical_columns=["region"], output_columns=["label"],
        data_fingerprint="public-profile", description="Improve detection of rare positive cases.",
        classes=["negative", "positive"], positive_class="positive", seed=19,
    )
    profile = {
        "task_type": "classification", "metric": "f1", "train_shape": [1000, 4],
        "test_shape": [200, 4], "feature_count": 3, "categorical_columns": ["region"],
        "target": {"name": "label", "dtype": "object", "unique": 2,
                   "class_counts": {"negative": 900, "positive": 100}},
        "available_packages": {"catboost": "1.2.8", "scikit-learn": "1.5.2"},
        "validation_plan": {"strategy": "stratified", "fold_count": 3},
        "features": {name: {"dtype": "object" if name == "region" else "float64",
                            "missing_fraction": 0, "unique": 12, "numeric": name != "region"}
                     for name in ("amount", "region", "account_age")},
    }
    best = ExperimentResult(
        trial_id="trial_0002", status="ok", artifact_dir="/missing", code_hash="best",
        hypothesis="Усилить CatBoost по редкому классу", metric=0.61, seconds=12,
        recipe={"family": "CatBoost", "parameters": {"iterations": 900, "learning_rate": 0.07},
                "preprocessing": {"add_missing_indicators": False},
                "fit_options": {"early_stopping_rounds": 80}},
        model_details={"class": "catboost.CatBoostClassifier",
                       "parameters": {"iterations": 612, "learning_rate": 0.07}},
        diagnostics={"per_class": [{"label": "positive", "f1": 0.61, "support": 100}],
                     "feature_importance": {"items": [{"feature": "amount", "importance": 0.8}]}},
    )
    failed = ExperimentResult(
        trial_id="trial_0003", status="invalid_recipe", artifact_dir="/missing", code_hash="bad",
        hypothesis="Проверить несовместимое изменение", metric=None, seconds=0,
        error="ValueError: latest recipe rejected: useful traceback marker",
    )
    return task, profile, [best, failed], best


def test_compact_recipe_prompt_v2_is_small_and_keeps_actionable_facts():
    task, profile, results, best = _case()
    messages = build_messages(
        task, profile, results, best,
        {"n_jobs": 1, "parallel_trials": 1, "proposal_batch_size": 2,
         "experiment_mode": "recipe", "remaining_seconds": 600},
        compact=True,
    )
    content = json.loads(messages[1]["content"])
    catalog = content["model_catalog"]

    # Compact recipes should fit a small local model context; 4 chars/token is a
    # conservative approximation used here to catch runaway payloads.
    assert len(messages[0]["content"] + messages[1]["content"]) <= 12_000
    assert len(json.dumps(catalog, ensure_ascii=False)) <= 12_000
    assert content["task"]["metric"] == "f1"
    assert content["task"]["positive_class"] == "positive"
    assert content["task"]["feature_columns"] == ["amount", "region", "account_age"]
    assert content["data_profile"]["available_packages"] == profile["available_packages"]
    assert content["runtime"]["proposal_batch_size"] == 2
    assert content["runtime"]["parallel_trials"] == 1
    assert content["best_experiment"]["model_details"]["parameters"]["learning_rate"] == 0.07
    assert content["best_experiment"]["metric"] == 0.61
    assert content["latest_experiment"]["error"].endswith("useful traceback marker")
    assert catalog["families"]["CatBoost"]["parameters"]["iterations"]["maximum"] == 5000
    assert catalog["families"]["CatBoost"]["parameters"]["learning_rate"]["exclusive_minimum"] == 0
    assert catalog["fit_options"]["options"]["early_stopping_rounds"]["maximum"] == 300
    assert "proposal_batch_size" in messages[0]["content"]
    assert "parallel_trials" in messages[0]["content"]
    assert "independent" in messages[0]["content"]
    assert "native JSON array" in messages[0]["content"]
    assert "measured" in messages[0]["content"]


def test_cancel_pending_aborts_active_http_request(monkeypatch):
    started = threading.Event()
    cancelled = threading.Event()

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, *_args, **_kwargs):
            started.set()
            try:
                await asyncio.sleep(30)
            finally:
                cancelled.set()

    monkeypatch.setattr("automl_agent.providers.httpx.AsyncClient", FakeAsyncClient)
    provider = OpenAICompatibleProvider("http://fake.local/v1", "", "test", retries=0)
    with ThreadPoolExecutor(max_workers=1) as pool:
        request = pool.submit(provider.complete, [], [], timeout=20, max_tokens=10)
        assert started.wait(2), "fake transport did not start"
        provider.cancel_pending()
        with pytest.raises(ProviderError) as caught:
            request.result(timeout=2)

    assert caught.value.code == "cancelled"
    assert cancelled.wait(1), "cancellation did not reach the active HTTP coroutine"


def test_oversized_prompt_keeps_compact_base_ids_and_proposal_budget():
    task, profile, base, _ = _case()
    task.feature_columns = [f"feature_{index:03}" for index in range(120)]
    task.categorical_columns = task.feature_columns[:12]
    task.description = "Long task description. " * 1200
    profile["feature_count"] = 120
    profile["categorical_columns"] = task.categorical_columns
    profile["features"] = {
        name: {"dtype": "float64", "missing_fraction": index / 500,
               "unique": 100 + index, "numeric": True, "unused": "x" * 500}
        for index, name in enumerate(task.feature_columns)
    }
    families = ["CatBoost", "ExtraTrees", "Linear", "LightGBM"]
    recipes = [
        {"family": "CatBoost", "parameters": {"iterations": 900, "learning_rate": 0.07},
         "preprocessing": {"add_missing_indicators": False}, "fit_options": {"early_stopping_rounds": 80}},
        {"family": "ExtraTrees", "parameters": {"n_estimators": 180, "max_leaf_nodes": 128},
         "preprocessing": {"add_missing_indicators": False}},
        {"family": "Linear", "parameters": {"C": 0.5, "max_iter": 300},
         "preprocessing": {"add_missing_indicators": False}},
        {"family": "LightGBM", "parameters": {"n_estimators": 500, "num_leaves": 31},
         "preprocessing": {"add_missing_indicators": False}, "fit_options": {"early_stopping_rounds": 50}},
    ]
    results = []
    for index, (family, recipe) in enumerate(zip(families, recipes), start=1):
        result = ExperimentResult(
            trial_id=f"trial_{index:04}", status="ok", artifact_dir="/missing", code_hash=family,
            hypothesis=f"Tune {family}", metric=0.7 - index / 100, seconds=10 + index,
            recipe=recipe,
            model_details={"class": f"{family}Classifier", "parameters": {**recipe["parameters"],
                           "measured_noise": "y" * 600}},
        )
        results.append(result)
    results[0].metric = 0.9
    messages = build_messages(
        task, profile, results, results[0],
        {"n_jobs": 1, "parallel_trials": 1, "proposal_batch_size": 2,
         "experiment_mode": "recipe", "remaining_seconds": 300},
        context_chars=16_000, compact=True,
    )
    content = json.loads(messages[1]["content"])
    bases = content["model_catalog"]["measured_bases"]

    assert {row["family"] for row in bases} == set(families)
    assert {row["trial_id"] for row in bases} == {result.trial_id for result in results}
    assert all("parameters" not in row and "recipe" not in row for row in bases)
    assert content["best_experiment"]["recipe"] == results[0].recipe
    assert content["runtime"]["proposal_batch_size"] == 2
