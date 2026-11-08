import json

import pandas as pd

from automl_agent.data import discover_task, prepare_split, profile_task


def make_task(tmp_path):
    task_dir = tmp_path / "nested_task"
    task_dir.mkdir()
    train = pd.DataFrame({
        "record_key": range(30),
        "temperature": [float(i) if i != 4 else float("inf") for i in range(30)],
        "zone": ["north", "south", "east"] * 10,
        "building_code": [1, 2, 3] * 10,
        "outcome": [float(i * 2) for i in range(30)],
    })
    test = pd.DataFrame({
        "record_key": [100, 101, 102],
        "temperature": [1.0, float("nan"), float("-inf")],
        "zone": ["new-zone", None, "north"],
        "building_code": [4, 2, 3],
    })
    train.to_csv(task_dir / "train.csv", index=False)
    test.to_csv(task_dir / "test.csv", index=False)
    pd.DataFrame([
        {"feature": "record_key", "role": "identifier", "dtype": "integer"},
        {"feature": "temperature", "role": "feature", "dtype": "number"},
        {"feature": "zone", "role": "feature", "dtype": "category"},
        {"feature": "building_code", "role": "feature", "dtype": "category"},
        {"feature": "outcome", "role": "target", "dtype": "number"},
    ]).to_csv(task_dir / "data_dictionary.csv", index=False)
    pd.DataFrame({"record_key": test.record_key, "outcome": [0, 0, 0]}).to_csv(
        task_dir / "sample_submission.csv", index=False
    )
    (task_dir / "task.md").write_text(
        """# Universal synthetic task\n\n"
        "Files: `inputs/train.csv`, `inputs/test.csv`, `inputs/data_dictionary.csv`, "
        "`inputs/sample_submission.csv`.\n\n"
        "The task is regression; primary metric is RMSE.\n""",
        encoding="utf-8",
    )
    return task_dir / "task.md"


def test_discovery_uses_roles_and_resolves_sibling_csv_fallback(tmp_path):
    task_path = make_task(tmp_path)
    task = discover_task(task_path, seed=13, validation_fraction=0.25)

    assert task.target == "outcome"
    assert task.task_type == "regression"
    assert task.metric == "rmse"
    assert task.id_columns == ["record_key"]
    assert task.feature_columns == ["temperature", "zone", "building_code"]
    assert task.categorical_columns == ["zone", "building_code"]
    assert task.output_columns == ["record_key", "outcome"]
    assert task.seed == 13
    assert task.dictionary_path.endswith("data_dictionary.csv")


def test_split_is_deterministic_keeps_ids_out_and_profiles_compactly(tmp_path):
    task = discover_task(make_task(tmp_path), seed=7, validation_fraction=0.2)
    split_a = prepare_split(task, tmp_path / "run-a")
    split_b = prepare_split(task, tmp_path / "run-b")

    assert split_a["split_fingerprint"] == split_b["split_fingerprint"]
    for key in ("train_features", "validation_features", "test_features"):
        frame = pd.read_csv(split_a[key])
        assert list(frame.columns) == ["temperature", "zone", "building_code"]
        assert "record_key" not in frame
    assert pd.read_csv(split_a["validation_target"]).columns.tolist() == ["outcome"]
    assert pd.read_csv(split_a["full_train_features"]).shape[0] == 30
    indices = json.loads((tmp_path / "run-a" / "split_indices.json").read_text())
    assert sorted(indices["train"] + indices["validation"]) == list(range(30))
    profile = profile_task(task)
    assert profile["train_shape"] == [30, 5]
    assert "record_key" not in profile["features"]
    assert profile["target"]["name"] == "outcome"


def test_classification_split_is_stratified_when_rare_class_can_be_retained(tmp_path):
    task_dir = tmp_path / "rare_class"
    task_dir.mkdir()
    train = pd.DataFrame({
        "key": range(30),
        "signal": list(range(30)),
        "label": ["common"] * 26 + ["rare"] * 4,
    })
    test = pd.DataFrame({"key": [100, 101], "signal": [31, 32]})
    train.to_csv(task_dir / "train.csv", index=False)
    test.to_csv(task_dir / "test.csv", index=False)
    pd.DataFrame({"key": test.key, "label": ["common", "common"]}).to_csv(
        task_dir / "sample_submission.csv", index=False
    )
    (task_dir / "task.md").write_text("Multiclass classification task. Macro F1 is primary.", encoding="utf-8")

    task = discover_task(task_dir / "task.md", seed=5, validation_fraction=0.2)
    assert task.target == "label"
    assert task.task_type == "classification"
    assert task.classes == ["common", "rare"]
    split = prepare_split(task, tmp_path / "rare-run")
    for key in ("train_target", "validation_target"):
        values = pd.read_csv(split[key])["label"]
        assert "rare" in set(values)


def test_explicit_primary_metric_wins_over_diagnostic_metrics_in_english_and_russian(tmp_path):
    cases = [
        ("Regression. Main metric: MAE; RMSE is reported for diagnostics.", "mae", "regression"),
        ("Regression. Metric: MAE; RMSE is diagnostic.", "mae", "regression"),
        ("Регрессия. Основная метрика MAE, RMSE приводится для справки.", "mae", "regression"),
        ("Primary metric: log_loss. ROC-AUC is diagnostic.", "log_loss", "classification"),
        ("Метрика оценки — log_loss; ROC-AUC справочная.", "log_loss", "classification"),
    ]
    for index, (description, metric, task_type) in enumerate(cases):
        fixture_dir = tmp_path / f"fixture_{index}"
        fixture_dir.mkdir()
        task_path = make_task(fixture_dir)
        # Keep each generated fixture independent while varying only its description.
        task_path.write_text(description, encoding="utf-8")
        if metric == "log_loss":
            train_path = task_path.parent / "train.csv"
            train = pd.read_csv(train_path)
            train["outcome"] = (train["outcome"] > train["outcome"].median()).astype(int)
            train.to_csv(train_path, index=False)

        task = discover_task(task_path)

        assert task.metric == metric
        assert task.task_type == task_type
