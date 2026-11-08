import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl_agent.agent import run, select_best, ResearchRun
from automl_agent.config import parse_config
from automl_agent.contracts import ExperimentResult, ModelResponse
from automl_agent.providers import OpenAICompatibleProvider


def task_at(path):
    path.mkdir()
    x = np.linspace(0, 5, 60)
    pd.DataFrame({'x': x, 'category': ['a', 'b'] * 30, 'target': x * 2}).to_csv(path / 'train.csv', index=False)
    pd.DataFrame({'x': [1., 2., 3.], 'category': ['a', 'b', 'unknown']}).to_csv(path / 'test.csv', index=False)
    (path / 'task.md').write_text('Regression. Predict target. Metric RMSE. Files train.csv test.csv. Save submission.csv.')
    return path / 'task.md'


def config(task, directory, **updates):
    cfg = parse_config(["--cv-folds", "1", "--family-trials", "3", *(['--offline'] if updates.get('offline') else []), str(task), '--run-dir', str(directory), '--time-budget', '90', '--trial-timeout', '20',
                        '--max-trials', '4', '--max-steps', '3', '--n-jobs', '4'])
    cfg.family_trials = 1
    cfg.experiment_mode = "code"
    cfg.parallel_trials = 2
    for key, value in updates.items():
        setattr(cfg, key, value)
    return cfg


def ledger(directory):
    return [json.loads(s) for s in (directory / 'events.jsonl').read_text().splitlines()]


def install_batch_provider(monkeypatch, broken=False):
    seen = []
    def complete(self, messages, tools, **kwargs):
        seen.append((messages, tools))
        if len(seen) > 1:
            name, arguments = 'finish_research', {'reason': 'Пара измерена; новых обоснованных гипотез нет.'}
        else:
            experiments = []
            for i in (1, 2):
                code = f'from sklearn.dummy import DummyRegressor\ndef build_model(context):\n    return DummyRegressor(strategy="constant", constant={i}.0)\n'
                if broken and i == 1:
                    code = 'def build_model(context):\n    return (\n'
                experiments.append({'hypothesis': f'Проверка константы {i}', 'code': code,
                                    'decision': {'model_family': 'DummyRegressor', 'rationale': 'Измерить независимый контрольный кандидат'}})
            name, arguments = 'run_experiments', {'experiments': experiments}
        return ModelResponse({'role': 'assistant', 'tool_calls': [{'id': 'batch', 'type': 'function', 'function': {
            'name': name, 'arguments': json.dumps(arguments)}}]}, total_tokens=30)
    monkeypatch.setattr(OpenAICompatibleProvider, 'complete', complete)
    monkeypatch.setenv('OPENAI_MODEL', 'fake')
    monkeypatch.setenv('AUTOML_PROVIDER', 'openai')
    return seen


def test_batch_has_two_real_workers_four_threads_and_single_writer(tmp_path, monkeypatch):
    seen = install_batch_provider(monkeypatch)
    directory = tmp_path / 'run'
    report = run(config(task_at(tmp_path / 'data'), directory))
    assert report['llm_trials'] == 2, 'both candidates from one model response must execute'
    trials = [t for t in report['trials'] if not t['is_baseline']]
    processes = [json.loads((Path(t['artifact_dir']) / 'process.json').read_text()) for t in trials]
    assert all(p['n_jobs'] == 4 and p['pid'] is not None for p in processes)
    assert max(p['started_at'] for p in processes) < min(p['finished_at'] for p in processes), 'real process intervals must overlap'
    assert len({t['trial_id'] for t in report['trials']}) == len(report['trials'])
    log = ledger(directory)
    assert [e['seq'] for e in log] == list(range(1, len(log) + 1))
    assert len([e for e in log if e['event'] == 'trial_completed']) == 3
    assert any(e['event'] == 'batch_completed' for e in log)
    assert report['llm_calls'] == len(seen) == 2
    assert report['reserve_plan']['selected_trial'] == report['selected_trial']
    assert report['reserve_plan']['reserve_seconds'] > 0
    assert any(e['event'] == 'reserve_updated' for e in log)


def test_one_bad_candidate_does_not_prevent_companion_completion(tmp_path, monkeypatch):
    install_batch_provider(monkeypatch, broken=True)
    report = run(config(task_at(tmp_path / 'data'), tmp_path / 'run'))
    generated = [t for t in report['trials'] if not t['is_baseline']]
    assert [t['status'] for t in generated] == ['invalid_code', 'ok']
    assert report['llm_trials'] == 1 and report['validation']['valid']


def test_full_family_comparison_is_host_measured_and_not_counted_as_llm(tmp_path):
    directory = tmp_path / 'run'
    report = run(config(task_at(tmp_path / 'data'), directory, offline=True, family_trials=3, max_trials=3))
    assert len(report['trials']) == 3, 'measure three distinct model families'
    assert report['llm_trials'] == 0
    comparison = json.loads((directory / 'family_comparison.json').read_text())
    assert len(comparison['trials']) == 3
    assert len({r['family'] for r in comparison['trials']}) == 3
    assert all(t['is_baseline'] for t in report['trials'])
    assert all(t['status'] == 'ok' for t in report['trials'])
    assert all(row['metric'] is not None for row in comparison['trials'])
    assert comparison['split_fingerprint'] == json.loads((directory / 'split_manifest.json').read_text())['split_fingerprint']
    assert comparison['selected_trial'] == report['selected_trial']


def test_selection_ties_do_not_depend_on_worker_completion_order():
    def row(name):
        return ExperimentResult(name, 'ok', '.', name, 'tie', metric=.5)
    assert select_best([row('trial_0002'), row('trial_0001')], True).trial_id == 'trial_0001'
    assert select_best([row('trial_0002'), row('trial_0001')], False).trial_id == 'trial_0001'


@pytest.mark.parametrize('args', [ ['--parallel-trials', '2', '--n-jobs', '5'],
                                  ['--parallel-trials', '3'],
                                  ['--parallel-trials', '2', '--memory-limit-mb', '16000'] ])
def test_resource_overcommit_is_rejected(args):
    with pytest.raises(SystemExit):
        parse_config(["--cv-folds", "1", "--family-trials", "3", '--offline', *args])


def test_two_by_four_is_default_and_final_reserve_has_no_old_ten_minute_cap():
    cfg = parse_config(["--cv-folds", "1", "--family-trials", "3", '--offline'])
    assert cfg.parallel_trials == 2 and cfg.n_jobs == 4
    assert cfg.reserve_seconds == pytest.approx(1620)
    assert parse_config(["--cv-folds", "1", "--family-trials", "3", '--offline', '--parallel-trials', '1', '--n-jobs', '8']).n_jobs == 8


def test_parallel_unknown_outcomes_are_settled_once_without_replay(tmp_path, monkeypatch):
    from automl_agent.journal import EventJournal, atomic_write_json
    task = task_at(tmp_path / 'data')
    directory = tmp_path / 'run'
    run(config(task, directory, offline=True))
    state = json.loads((directory / 'state.json').read_text())
    state.update(active=True, checkpoint_epoch=time.time())
    atomic_write_json(directory / 'state.json', state)
    journal = EventJournal(directory / 'events.jsonl')
    journal.append('tool_call', call_id='unknown-batch', name='run_experiments', arguments={'experiments': []})
    for i in (2, 3):
        journal.append('trial_started', trial_id=f'trial_{i:04d}', code_hash=f'unknown-{i}', hypothesis=f'Unknown{i}', is_baseline=False, origin='llm')
    def forbidden(*args, **kwargs):
        raise AssertionError('unknown batch must not replay')
    monkeypatch.setattr(ResearchRun, 'experiment', forbidden)
    report = run(config(task, directory, offline=True, resume=True))
    assert len([t for t in report['trials'] if t['status'] == 'interrupted']) == 2
    first = len(ledger(directory))
    run(config(task, directory, offline=True, resume=True))
    log = ledger(directory)
    assert len(log) == first
    settled = [e for e in log if e['event'] == 'tool_result' and e['payload']['call_id'] == 'unknown-batch']
    assert len(settled) == 1 and settled[0]['payload']['result']['error']['code'] == 'INTERRUPTED'


def test_json_only_batch_response_is_recognized():
    from automl_agent.tools import parse_response_calls
    calls = parse_response_calls({'content': json.dumps({'experiments': [{'hypothesis': 'h', 'code': 'c'}]})})
    assert calls[0]['function']['name'] == 'run_experiments'


def test_finalization_uses_remaining_total_budget_and_preserves_csv_on_failure(tmp_path, monkeypatch):
    from automl_agent.contracts import ProcessResult
    from automl_agent import runtime
    task = task_at(tmp_path / 'data')
    directory = tmp_path / 'run'
    run(config(task, directory, offline=True))
    research = ResearchRun(config(task, directory, offline=True, resume=True))
    best = select_best(research.results, research.task.maximize)
    before = (task.parent / 'submission.csv').read_bytes()
    research.split = json.loads((directory / 'split_manifest.json').read_text())
    research.config.trial_timeout = .01
    seen = []
    def fail(*args, **kwargs):
        seen.append(kwargs['timeout'])
        return ProcessResult('error', 1, '', 'synthetic refit failure', .01)
    monkeypatch.setattr(runtime, 'run_process', fail)
    result = research.finalize(best)
    assert seen[0] > 10, 'refit must not inherit the .01 second experiment timeout'
    assert result['status'] == 'error'
    assert (task.parent / 'submission.csv').read_bytes() == before
