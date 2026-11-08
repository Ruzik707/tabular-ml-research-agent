from pathlib import Path

import pytest

from automl_agent.config import parse_config


def test_baseline_cli_and_secret_exclusion(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:7777/v1")
    config = parse_config([str(tmp_path / "task.md"), "--time-budget", "120"])
    assert config.task_path == (tmp_path / "task.md").resolve()
    assert config.provider == "openai"
    assert config.model == "test-model"
    assert config.api_key == "private-test-key"
    assert "private-test-key" not in str(config.public_dict())
    assert 0 < config.reserve_seconds < config.time_budget


def test_offline_needs_no_model_or_api(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = parse_config(["--offline", "--time-budget", "10"])
    assert config.offline is True
    assert config.task_path == Path("task.md").resolve()


def test_default_research_is_limited_by_time_not_six_trials():
    config = parse_config(['--offline'])
    assert config.max_trials is None and config.max_steps is None
    assert config.trial_limit == float('inf')
    assert config.step_limit == float('inf')
    assert config.cv_folds == 5 and config.family_trials == 4
    assert config.trial_timeout == 1800
    assert config.validation_strategy == 'auto'


def test_explicit_limits_still_apply():
    config = parse_config(['--offline', '--max-trials', '9', '--max-steps', '12', '--cv-folds', '3'])
    assert config.trial_limit == 9 and config.step_limit == 12 and config.cv_folds == 3


@pytest.mark.parametrize("args", [
    ["--time-budget", "0"], ["--time-budget", "10801"],
    ["--n-jobs", "9"], ["--max-trials", "0"],
    ["--validation-fraction", "0.99"], ["--resume"],
])
def test_invalid_configuration_is_rejected(args):
    with pytest.raises((ValueError, SystemExit)):
        parse_config(["--offline", *args])


def test_finalize_only_requires_resume_and_needs_no_llm(monkeypatch, tmp_path):
    monkeypatch.delenv('OPENAI_MODEL', raising=False)
    with pytest.raises(SystemExit):
        parse_config(['--finalize-only'])
    config = parse_config(['--resume', '--finalize-only', '--run-dir', str(tmp_path)])
    assert config.finalize_only and config.resume
