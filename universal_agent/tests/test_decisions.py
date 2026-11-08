import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from automl_agent.agent import run
from automl_agent.config import parse_config
from automl_agent.contracts import ExperimentResult, ModelResponse
from automl_agent.providers import OpenAICompatibleProvider
from automl_agent.strategy import assess_experiment, plan_research


def make_task(path):
    path.mkdir()
    x = np.linspace(0, 5, 50)
    pd.DataFrame({'x': x, 'group': ['a', 'b'] * 25, 'target': x * 3}).to_csv(path / 'train.csv', index=False)
    pd.DataFrame({'x': [1., 2., 3.], 'group': ['a', 'b', 'c']}).to_csv(path / 'test.csv', index=False)
    (path / 'task.md').write_text('Regression. Predict target. Metric RMSE. Files train.csv test.csv. Save submission.csv.')
    return path / 'task.md'


def config_for(task, directory, *extra):
    return parse_config(["--cv-folds", "1", "--family-trials", "3", str(task), '--run-dir', str(directory), '--time-budget', '90', '--trial-timeout', '20',
                         '--max-trials', '2', '--n-jobs', '1', '--family-trials', '1', '--experiment-mode', 'code', *extra])


def events(directory):
    return [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]


def test_offline_run_explains_data_validation_strategy_and_actual_model(tmp_path):
    task = make_task(tmp_path / 'data')
    directory = tmp_path / 'run'
    report = run(config_for(task, directory, '--offline'))
    assert (directory / 'agent.log').is_file(), 'a readable live decision log is missing'
    assert (directory / 'decisions.jsonl').is_file()
    log = events(directory)
    for name in ('data_profiled', 'validation_planned', 'research_strategy', 'decision_recorded',
                 'experiment_assessed', 'submission_validated'):
        assert any(e['event'] == name for e in log), name
    assessed = next(e['payload'] for e in log if e['event'] == 'experiment_assessed')
    assert assessed['source'] == 'harness'
    assert assessed['metric'] == report['best_metric']
    assert assessed['model_details']['estimator_chain'][-1].endswith('CatBoostRegressor')
    assert report['trials'][0]['decision']['model_family'] == 'CatBoost'
    assert report['decision_log'] == str(directory / 'agent.log')
    assert (directory / 'strategy.json').is_file()
    # Derivative corruption is repaired even for a completed no-op resume.
    before = (directory / 'decisions.jsonl').read_bytes()
    (directory / 'decisions.jsonl').write_text('torn')
    resumed = run(config_for(task, directory, '--offline', '--resume'))
    assert resumed['selected_trial'] == report['selected_trial']
    assert (directory / 'decisions.jsonl').read_bytes() == before


@pytest.mark.parametrize('with_decision', [True, False])
def test_llm_decision_is_saved_separately_from_measured_outcome(tmp_path, monkeypatch, with_decision):
    task = make_task(tmp_path / 'data')
    directory = tmp_path / 'run'
    decision = {'model_family': 'DummyRegressor', 'observations': ['Numeric x and categorical group are available'],
                'rationale': 'Check a constant reference to detect baseline value', 'alternatives': ['CatBoost'],
                'expected_effect': 'No improvement is claimed before evaluation', 'risks': ['Underfitting'],
                'previous_result_assessment': 'A successful host baseline is present'}
    captured = []
    def complete(_self, messages, tools, **kwargs):
        captured.append((messages, tools))
        args = {'hypothesis': 'Проверить константную модель', 'code': 'from sklearn.dummy import DummyRegressor\ndef build_model(context):\n    return DummyRegressor()\n'}
        if with_decision:
            args['decision'] = decision
        return ModelResponse({'role': 'assistant', 'tool_calls': [{'id': 'decision', 'type': 'function', 'function': {
            'name': 'run_experiment', 'arguments': json.dumps(args)}}]}, total_tokens=50)
    monkeypatch.setattr(OpenAICompatibleProvider, 'complete', complete)
    monkeypatch.setenv('AUTOML_PROVIDER', 'openai')
    monkeypatch.setenv('OPENAI_MODEL', 'fake')
    report = run(config_for(task, directory))
    assert report['llm_trials'] == 1, 'structured decision arguments must be accepted'
    trial = report['trials'][-1]
    assert trial['decision'] == (decision if with_decision else {})
    assert trial['model_details']['estimator_chain'][-1].endswith('DummyRegressor')
    log = events(directory)
    explanation = next(e for e in log if e['event'] == 'decision_recorded' and e['payload']['trial_id'] == trial['trial_id'])
    assert explanation['payload']['explanation_provided'] is with_decision
    assert explanation['payload']['source'] == 'llm'
    started = next(e for e in log if e['event'] == 'trial_started' and e['payload']['trial_id'] == trial['trial_id'])
    assert explanation['seq'] < started['seq'], 'record the choice before running the generated code'
    assessment = next(e['payload'] for e in log if e['event'] == 'experiment_assessed' and e['payload']['trial_id'] == trial['trial_id'])
    assert assessment['selected'] is False and assessment['delta'] > 0  # RMSE worse than baseline
    assert assessment['previous_best'] == report['trials'][0]['metric']
    assert assessment['source'] == 'harness'
    assert json.loads((directory / trial['trial_id'] / 'decision.json').read_text())['decision'] == trial['decision']
    prompt = json.loads(captured[0][0][1]['content'])
    assert prompt['research_strategy']['candidate_shortlist'][0]['model_family'] == 'CatBoost'
    assert 'decision' in captured[0][1][0]['function']['parameters']['properties']


def test_strategy_uses_data_facts_and_available_packages():
    task = SimpleNamespace(task_type='classification', metric='macro_f1', maximize=True, categorical_columns=['city'], feature_columns=['x', 'city'])
    profile = {'train_shape': [1000, 3], 'categorical_columns': ['city'], 'features': {'city': {'missing': 5, 'unique': 300}},
               'target': {'class_counts': {'a': 900, 'b': 100}, 'unique': 2},
               'available_packages': {'catboost': '1.2.8', 'scikit-learn': '1.7.0'}}
    config = SimpleNamespace(time_budget=10800, reserve_seconds=600, trial_timeout=900, n_jobs=4)
    strategy = plan_research(task, profile, config)
    assert strategy['candidate_shortlist'][0]['model_family'] == 'CatBoost'
    assert strategy['facts']['class_imbalance_ratio'] == 9
    assert strategy['facts']['features_with_missing'] == 1
    assert strategy['budget']['finalization_reserve_seconds'] == 600
    assert strategy['parallel_trials'] == 2
    assert any('class' in item.lower() or 'дисбаланс' in item.lower() for item in strategy['priorities'])
    profile['available_packages']['catboost'] = 'unavailable'
    other = plan_research(task, profile, config)
    assert all(c['model_family'] != 'CatBoost' for c in other['candidate_shortlist'])
    task.categorical_columns = []
    task.feature_columns = [f'x{i}' for i in range(400)]
    profile['features'] = {}
    numeric = plan_research(task, profile, config)
    assert any(c['model_family'] == 'Linear' for c in numeric['candidate_shortlist'])


@pytest.mark.parametrize('maximize,previous,current,selected', [(True, .5, .7, True), (True, .5, .4, False),
                                                              (False, 3., 2., True), (False, 3., 4., False),
                                                              (False, 3., 3., False)])
def test_host_assessment_reports_direction_and_ties(maximize, previous, current, selected):
    old = ExperimentResult('old', 'ok', '.', 'a', 'old', metric=previous)
    new = ExperimentResult('new', 'ok', '.', 'b', 'new', metric=current)
    assessment = assess_experiment(new, old, maximize, 'metric')
    assert assessment['selected'] is selected
    assert assessment['delta'] == pytest.approx(current - previous)
    assert assessment['previous_best'] == previous
    assert assessment['source'] == 'harness'
    assert assessment['reason']
    new.status = 'timeout'
    failed = assess_experiment(new, old, maximize, 'metric')
    assert failed['selected'] is False and failed['delta'] is None


def test_first_success_and_legacy_result_have_explicit_defaults():
    legacy = {'trial_id': 'old', 'status': 'ok', 'artifact_dir': '.', 'code_hash': 'a', 'hypothesis': 'legacy', 'metric': 2.}
    restored = ExperimentResult.from_dict(legacy)
    assert restored.decision == {} and restored.model_details == {}
    first = assess_experiment(restored, None, False, 'rmse')
    assert first['selected'] is True and first['previous_best'] is None and first['delta'] is None
    restored.metric = float('nan')
    assert assess_experiment(restored, None, False, 'rmse')['selected'] is False


def test_baseline_fallback_records_failure_and_estimator_backbone(monkeypatch):
    import catboost
    from automl_agent.modeling import baseline_code
    from automl_agent.worker import _estimator_details

    def unavailable_backend(**kwargs):
        raise TypeError('synthetic CatBoost parameter failure')
    monkeypatch.setattr(catboost, 'CatBoostRegressor', unavailable_backend)
    namespace = {}
    exec(baseline_code(None, n_jobs=1, iterations=3), namespace)
    model = namespace['build_model']({'task_type': 'regression', 'n_jobs': 1})
    model.fit(pd.DataFrame({'x': range(12)}), np.arange(12))
    details = _estimator_details(model)
    assert details.get('fallback_reason') == 'CatBoost: synthetic CatBoost parameter failure'
    assert details['estimator_chain'][-1].endswith('ExtraTreesRegressor'), 'report the ensemble, not its individual trees'
    assert len(model.predict(pd.DataFrame({'x': [1, 3]}))) == 2


def test_key_echo_in_hypothesis_is_redacted_in_snapshots_and_reports(tmp_path, monkeypatch):
    from automl_agent.journal import EventJournal
    key = 'private-review-key-924'
    task = make_task(tmp_path / 'data')
    directory = tmp_path / 'run'
    def complete(*args, **kwargs):
        args = {'hypothesis': 'Reference ' + key, 'code': 'from sklearn.dummy import DummyRegressor\ndef build_model(context):\n    return DummyRegressor()\n'}
        return ModelResponse({'role': 'assistant', 'tool_calls': [{'id': 'echo', 'type': 'function', 'function': {
            'name': 'run_experiment', 'arguments': json.dumps(args)}}]}, total_tokens=30)
    monkeypatch.setenv('OPENAI_MODEL', 'fake')
    monkeypatch.setenv('AUTOML_PROVIDER', 'openai')
    monkeypatch.setenv('OPENAI_API_KEY', key)
    monkeypatch.setattr(OpenAICompatibleProvider, 'complete', complete)
    report = run(config_for(task, directory))
    assert key not in json.dumps(report)
    for name in ('events.jsonl', 'decisions.jsonl', 'agent.log', 'state.json', 'experiments.json', 'report.json', 'report.md'):
        assert key not in (directory / name).read_text(), name
    assert '[REDACTED]' in report['trials'][-1]['hypothesis']


def test_strategy_direction_follows_task_contract():
    task = SimpleNamespace(task_type='classification', metric='weighted_f1', maximize=True, categorical_columns=[], feature_columns=['x'])
    profile = {'train_shape': [50, 2], 'target': {}, 'features': {}, 'available_packages': {}}
    config = SimpleNamespace(time_budget=90, reserve_seconds=10, trial_timeout=20, n_jobs=1)
    assert 'максимум' in plan_research(task, profile, config)['selection_rule']


def test_resume_recovers_assessment_after_completed_trial_without_reexecution(tmp_path, monkeypatch):
    from automl_agent.agent import ResearchRun
    from automl_agent.journal import EventJournal

    task = make_task(tmp_path / 'data')
    directory = tmp_path / 'run'
    run(config_for(task, directory, '--offline'))
    journal_path = directory / 'events.jsonl'
    original = [json.loads(line) for line in journal_path.read_text().splitlines()]
    kept = [e for e in original if e['event'] != 'experiment_assessed']
    journal_path.write_text(''.join(json.dumps({**e, 'seq': i}) + '\n' for i, e in enumerate(kept, 1)))
    before = len([e for e in kept if e['event'] == 'trial_started'])
    def forbidden(*args, **kwargs):
        raise AssertionError('resume must not rerun a completed trial')
    monkeypatch.setattr(ResearchRun, 'experiment', forbidden)
    run(config_for(task, directory, '--offline', '--resume'))
    log = events(directory)
    recovered = [e['payload'] for e in log if e['event'] == 'experiment_assessed']
    assert len(recovered) == 1 and recovered[0]['recovered'] is True
    assert recovered[0]['selected'] is True
    assert before == len([e for e in log if e['event'] == 'trial_started'])
    run(config_for(task, directory, '--offline', '--resume'))
    assert len([e for e in events(directory) if e['event'] == 'experiment_assessed']) == 1


def test_completed_legacy_resume_cleans_old_reports_without_new_work(tmp_path, monkeypatch):
    from automl_agent.agent import ResearchRun
    key = 'legacy-review-key-729'
    monkeypatch.setenv('OPENAI_API_KEY', key)
    task = make_task(tmp_path / 'data')
    directory = tmp_path / 'run'
    run(config_for(task, directory, '--offline'))
    # Simulate reports/snapshots from the old implementation; canonical events
    # remain authoritative, and no new work is needed for a completed run.
    for name in ('state.json', 'experiments.json', 'report.json'):
        path = directory / name
        content = json.loads(path.read_text())
        content['legacy_note'] = key
        path.write_text(json.dumps(content))
    (directory / 'report.md').write_text('Legacy hypothesis ' + key)
    def forbidden(*args, **kwargs):
        raise AssertionError('completed resume must not train')
    monkeypatch.setattr(ResearchRun, 'experiment', forbidden)
    report = run(config_for(task, directory, '--offline', '--resume'))
    assert key not in json.dumps(report)
    for name in ('state.json', 'experiments.json', 'report.json', 'report.md'):
        assert key not in (directory / name).read_text(), name
