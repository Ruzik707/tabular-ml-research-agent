from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from universal_agent.scripts.benchmark_data import prepare_cases, score_submission


def _write_public_sources(root: Path):
    regression = root / "Регрессия"
    regression.mkdir(parents=True)
    pd.DataFrame({
        "id": ["0001", "0002", "0003", "0004", "0005", "0006"],
        "Year_Factor": [1, 1, 2, 2, 3, 3],
        "x": [10, 11, 12, 13, 14, 15],
        "site_eui": [1.0, 2.0, 3.0, 4.0, 100.0, 200.0],
    }).to_csv(regression / "train.csv", index=False)
    pd.DataFrame({
        "feature": ["id", "Year_Factor", "x", "site_eui"],
        "role": ["id", "feature", "feature", "feature"],
        "dtype": ["numeric"] * 4,
        "description_ru": ["id", "year", "x", "wrong role copied from upstream"],
    }).to_csv(regression / "data_dictionary.csv", index=False)

    multiclass = root / "Многоклассовая классификация"
    multiclass.mkdir(parents=True)
    rows = []
    row_id = 0
    for label in ["A", "B", "C"]:
        for group_index in range(10):
            for duplicate in range(2):
                rows.append({
                    "id": f"{row_id:05d}",
                    "signal": f"{label}_{group_index}",
                    "measure": group_index,
                    "attack_cat": label,
                })
                row_id += 1
    pd.DataFrame(rows).to_csv(multiclass / "train.csv", index=False)
    pd.DataFrame({
        "feature": ["id", "signal", "measure", "attack_cat"],
        "role": ["id", "feature", "feature", "target"],
        "dtype": ["numeric", "category", "numeric", "category"],
        "description_ru": ["id", "signal", "measure", "attack"],
    }).to_csv(multiclass / "data_dictionary.csv", index=False)


@pytest.fixture
def public_cases(tmp_path):
    source = tmp_path / "source"
    _write_public_sources(source)
    return source, tmp_path / "work"


def test_prepare_cases_keeps_outer_truth_separate_and_regression_last_period(public_cases):
    source, work = public_cases
    cases = prepare_cases(source, work, seed=17)
    by_name = {case["name"]: case for case in cases}
    regression = by_name["public-regression"]
    task_dir = Path(regression["task_dir"])
    train = pd.read_csv(task_dir / "train.csv", dtype={"id": "string"})
    test = pd.read_csv(task_dir / "test.csv", dtype={"id": "string"})
    truth = pd.read_csv(regression["truth_path"], dtype={"id": "string"})

    assert train["Year_Factor"].max() == 2
    assert test["Year_Factor"].tolist() == [3, 3]
    assert "site_eui" not in test.columns
    assert truth["site_eui"].tolist() == [100.0, 200.0]
    assert train["id"].tolist() == ["0001", "0002", "0003", "0004"]
    assert test["id"].tolist() == truth["id"].tolist() == ["0005", "0006"]
    assert regression["target"] == "site_eui"
    assert regression["metric"] == "rmse" and regression["maximize"] is False
    assert set(regression["feature_columns"]) == {"Year_Factor", "x"}
    assert regression["id_columns"] == ["id"]
    assert "primary metric: rmse" in (task_dir / "task.md").read_text().lower()
    assert not any(key in regression for key in ("labels", "truth", "target_values"))


def test_prepare_cases_public_multiclass_has_disjoint_duplicate_groups_and_all_classes(public_cases):
    source, work = public_cases
    cases = prepare_cases(source, work, seed=0)
    case = next(item for item in cases if item["name"] == "public-multiclass")
    task_dir = Path(case["task_dir"])
    train = pd.read_csv(task_dir / "train.csv", dtype={"id": "string"})
    test = pd.read_csv(task_dir / "test.csv", dtype={"id": "string"})
    truth = pd.read_csv(case["truth_path"], dtype={"id": "string"})
    assert sorted(train["attack_cat"].unique().tolist()) == sorted(test["signal"].str[0].unique().tolist())
    train_groups = set(zip(train["signal"], train["measure"]))
    test_groups = set(zip(test["signal"], test["measure"]))
    assert train_groups.isdisjoint(test_groups)
    assert set(case["classes"]) == set(truth["attack_cat"])
    assert test["id"].tolist() == truth["id"].tolist()
    assert "StratifiedGroupKFold(n_splits=5" in case["split_protocol"]


def test_prepare_cases_creates_fixed_size_synthetic_tasks_and_positive_class(public_cases):
    source, work = public_cases
    cases = prepare_cases(source, work, seed=8)
    by_name = {case["name"]: case for case in cases}
    reg = by_name["synthetic-signed-linear-regression"]
    binary = by_name["synthetic-imbalanced-binary-f1"]
    assert reg["train_rows"] == 600 and reg["test_rows"] == 200
    assert binary["train_rows"] == 800 and binary["test_rows"] == 200
    assert binary["positive_class"] == "alert"
    assert binary["classes"] == ["alert", "normal"]
    assert binary["metric"] == "f1" and binary["maximize"] is True
    for case in (reg, binary):
        task_dir = Path(case["task_dir"])
        assert set(path.name for path in task_dir.iterdir()) == {
            "train.csv", "test.csv", "sample_submission.csv", "data_dictionary.csv", "task.md"
        }
        assert case["input_hashes"].keys() == {
            "train.csv", "test.csv", "sample_submission.csv", "data_dictionary.csv", "task.md"
        }


def test_prepare_cases_is_deterministic_and_hashes_written_agent_inputs(public_cases):
    source, work = public_cases
    first = prepare_cases(source, work / "first", seed=99)
    second = prepare_cases(source, work / "second", seed=99)
    assert [case["split_fingerprint"] for case in first] == [case["split_fingerprint"] for case in second]
    for a, b in zip(first, second):
        assert a["input_hashes"] == b["input_hashes"]
        assert a["train_rows"] == b["train_rows"]
        assert a["test_rows"] == b["test_rows"]


def _case_for_scoring(tmp_path, *, task_type, metric, target, classes, ids, positive_class=None):
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir(exist_ok=True)
    if task_type == "classification":
        target_values = [classes[index % len(classes)] for index in range(len(ids))]
    else:
        target_values = [1.0, 3.0][:len(ids)]
    truth = pd.DataFrame({"row_id": ids, target: target_values})
    truth.to_csv(evaluator / "truth.csv", index=False)
    return {
        "name": "fixture", "target": target, "task_type": task_type, "metric": metric,
        "maximize": True if task_type == "classification" else False,
        "positive_class": positive_class, "classes": classes if task_type == "classification" else [],
        "id_columns": ["row_id"], "truth_path": str(evaluator / "truth.csv"), "test_rows": len(ids),
    }


def test_score_submission_binary_f1_uses_explicit_positive_label_and_preserves_zero_prefix_ids(tmp_path):
    case = _case_for_scoring(
        tmp_path, task_type="classification", metric="f1", target="label",
        classes=["alert", "normal"], ids=["001", "002"], positive_class="alert",
    )
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"row_id": ["001", "002"], "label": ["alert", "alert"]}).to_csv(submission, index=False)
    result = score_submission(case, submission)
    assert result["valid"] is True
    assert result["score"] == pytest.approx(2 / 3)
    assert result["metrics"]["f1"] == pytest.approx(2 / 3)


def test_score_submission_macro_f1_includes_absent_class_from_predictions(tmp_path):
    case = _case_for_scoring(
        tmp_path, task_type="classification", metric="macro_f1", target="label",
        classes=["a", "b", "c"], ids=["x", "y"],
    )
    pd.DataFrame({"row_id": ["x", "y"], "label": ["a", "a"]}).to_csv(tmp_path / "sub.csv", index=False)
    result = score_submission(case, tmp_path / "sub.csv")
    assert result["valid"] is True
    assert result["score"] == pytest.approx(2 / 9)
    assert result["metrics"]["macro_f1"] == pytest.approx(2 / 9)


@pytest.mark.parametrize("frame,expected", [
    (pd.DataFrame({"label": ["normal", "alert"], "row_id": ["001", "002"]}), "columns"),
    (pd.DataFrame({"row_id": ["002", "001"], "label": ["alert", "normal"]}), "id"),
    (pd.DataFrame({"row_id": ["1", "002"], "label": ["alert", "normal"]}), "id"),
    (pd.DataFrame({"row_id": ["001", "002"], "label": ["unexpected", "normal"]}), "class"),
])
def test_score_submission_rejects_wrong_schema_ids_or_classes(tmp_path, frame, expected):
    case = _case_for_scoring(
        tmp_path, task_type="classification", metric="f1", target="label",
        classes=["alert", "normal"], ids=["001", "002"], positive_class="alert",
    )
    submission = tmp_path / "bad.csv"
    frame.to_csv(submission, index=False)
    result = score_submission(case, submission)
    assert result["valid"] is False
    assert result["score"] is None
    assert expected in result["error"].lower()


def test_score_submission_rejects_nonfinite_regression_output(tmp_path):
    case = _case_for_scoring(
        tmp_path, task_type="regression", metric="rmse", target="value",
        classes=[], ids=["001", "002"],
    )
    pd.DataFrame({"row_id": ["001", "002"], "value": [1.0, np.inf]}).to_csv(tmp_path / "bad.csv", index=False)
    result = score_submission(case, tmp_path / "bad.csv")
    assert result["valid"] is False and result["score"] is None
    assert "finite" in result["error"].lower() or "конеч" in result["error"].lower()


def test_score_submission_regression_reports_rmse(tmp_path):
    case = _case_for_scoring(
        tmp_path, task_type="regression", metric="rmse", target="value",
        classes=[], ids=["001", "002"],
    )
    pd.DataFrame({"row_id": ["001", "002"], "value": [2.0, 1.0]}).to_csv(tmp_path / "sub.csv", index=False)
    result = score_submission(case, tmp_path / "sub.csv")
    assert result["valid"] is True
    assert result["score"] == pytest.approx(np.sqrt(2.5))
