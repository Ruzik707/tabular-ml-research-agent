import json

import pytest

from automl_agent.audit import AuditJournal


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_projects_redacted_decisions_and_omits_private_provider_payloads(tmp_path, capsys):
    journal = AuditJournal(tmp_path / "events.jsonl", secrets=["private-key"], echo=False)
    journal.append(
        "llm_request", request_id="request_0001", provider="ollama", model="qwen",
        messages=[{"role": "user", "content": "private prompt private-key"}],
        tools=[{"code": "private generated code"}], max_tokens=512, input_token_bound=42,
    )
    journal.append("trial_started", trial_id="trial_0001", hypothesis="Use CatBoost", code="model code")
    journal.append(
        "decision_recorded", trial_id="trial_0001", source="llm",
        hypothesis={"model_family": "CatBoost", "rationale": "Strong categorical baseline"},
        decision={
            "alternatives": ["LightGBM"], "risks": ["Small sample"],
            "expected_effect": "Improve ROC AUC", "chain_of_thought": "private hidden reasoning",
        },
        explanation_provided=True, raw_prompt="do not project", source_code="do not project",
    )
    journal.append(
        "experiment_assessed", trial_id="trial_0001", metric_name="roc_auc", metric=0.81,
        previous_best=0.77, maximize=True, selected=True, delta=0.04,
        reason="Improved validation AUC", model_details={
            "class": "catboost.CatBoostClassifier",
            "estimator_chain": ["sklearn.pipeline.Pipeline", "catboost.CatBoostClassifier"],
        },
    )

    decisions = _records(tmp_path / "decisions.jsonl")
    decision = next(record for record in decisions if record["event"] == "decision_recorded")
    request = next(record for record in decisions if record["event"] == "llm_request")
    assert set(decision) == {"seq", "timestamp", "event", "source", "details"}
    assert decision["source"] == "LLM"
    assert decision["details"]["trial_id"] == "trial_0001"
    assert decision["details"]["explanation_provided"] is True
    assert request["details"] == {
        "request_id": "request_0001", "provider": "ollama", "model": "qwen",
        "max_tokens": 512, "input_token_bound": 42,
    }
    contents = (tmp_path / "agent.log").read_text(encoding="utf-8") + (tmp_path / "decisions.jsonl").read_text(encoding="utf-8")
    assert "private-key" not in contents
    assert "private prompt" not in contents
    assert "private generated code" not in contents
    assert "do not project" not in contents
    assert "private hidden reasoning" not in contents
    assert "Use CatBoost" in contents
    assert "roc_auc" in contents
    human = (tmp_path / "agent.log").read_text(encoding="utf-8")
    assert "Ожидаемый эффект: Improve ROC AUC" in human
    assert "Фактическая валидация" in human
    assert "sklearn.pipeline.Pipeline → catboost.CatBoostClassifier" in human
    assert capsys.readouterr().err == ""


def test_append_echoes_russian_audit_line_once_and_init_does_not_replay(tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    journal = AuditJournal(path, echo=True)
    first = journal.append("research_finished", source="llm", reason="Candidate validated")
    live = capsys.readouterr().err
    assert "исследован" in live.lower() or "заверш" in live.lower()
    assert "[LLM]" in live
    assert first["seq"] == 1

    rebuilt = AuditJournal(path, echo=True)
    assert capsys.readouterr().err == ""
    rebuilt.append("research_stopped", reason="max_trials", total_tokens=11, elapsed_seconds=2.5)
    lines = (tmp_path / "agent.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "research_finished" not in lines[0]
    assert "max_trials" in lines[1]


def test_init_rebuilds_missing_and_corrupt_projections_without_duplicates(tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    journal = AuditJournal(path, echo=False)
    journal.append("run_started", config={"provider": "ollama"})
    journal.append("trial_started", trial_id="trial_0001", hypothesis="First candidate", is_baseline=True)
    journal.append("trial_completed", result={"trial_id": "trial_0001", "status": "ok", "metric": 0.7})
    (tmp_path / "agent.log").write_text("partial\n", encoding="utf-8")
    (tmp_path / "decisions.jsonl").write_text("broken json\n", encoding="utf-8")

    AuditJournal(path, echo=True)
    assert capsys.readouterr().err == ""
    records = _records(tmp_path / "decisions.jsonl")
    assert [record["seq"] for record in records] == [1, 2, 3]
    assert (tmp_path / "agent.log").read_text(encoding="utf-8").count("First candidate") == 1
    AuditJournal(path, echo=False)
    assert len(_records(tmp_path / "decisions.jsonl")) == 3


def test_legacy_events_and_missing_rationale_are_kept_without_invention(tmp_path):
    path = tmp_path / "events.jsonl"
    legacy = AuditJournal(path, echo=False)
    legacy.append("trial_started", trial_id="trial_0001", hypothesis="Old run", is_baseline=False)
    legacy.append("trial_completed", result={"trial_id": "trial_0001", "status": "error", "metric": None})
    legacy.append("llm_error", request_id="request_0001", code="timeout", error="long raw provider body")
    legacy.append("final_fit_completed", result={
        "status": "ok", "seconds": 1.2,
        "model_details": {"estimator_chain": ["sklearn.pipeline.Pipeline", "sklearn.linear_model.LogisticRegression"]},
    })
    legacy.append("run_completed", report={"status": "degraded", "stop_reason": "provider_error", "total_tokens": 20})
    legacy.append("decision_recorded", trial_id="trial_0002", source="LLM", hypothesis="Older idea")

    records = _records(tmp_path / "decisions.jsonl")
    decision = records[-1]
    assert decision["details"]["hypothesis"] == "Older idea"
    assert "rationale" not in json.dumps(decision)
    projected = (tmp_path / "agent.log").read_text(encoding="utf-8")
    assert "trial_0001" in projected
    assert "timeout" in projected
    assert "events.jsonl #2" in projected
    assert "trial_0001/process.json" in projected
    assert "финал" in projected.lower()
    assert "sklearn.linear_model.LogisticRegression" in projected
    assert "degraded" in projected
    assert "long raw provider body" not in projected


def test_projection_failure_keeps_canonical_event_and_warns(monkeypatch, tmp_path, capsys):
    journal = AuditJournal(tmp_path / "events.jsonl", echo=False)

    def fail(*args, **kwargs):
        raise OSError("read-only projection")

    monkeypatch.setattr("automl_agent.audit._write_projection", fail)
    record = journal.append("research_finished", reason="done")
    assert record["seq"] == 1
    assert len(journal.read_events()) == 1
    warning = capsys.readouterr().err.lower()
    assert "events.jsonl" in warning
    assert "восстанов" in warning or "rebuild" in warning
