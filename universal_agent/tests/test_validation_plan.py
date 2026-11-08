import numpy as np
import pandas as pd

from automl_agent.contracts import TaskSpec
from automl_agent.validation import make_validation_plan, prepare_cv_split


def task(tmp_path, classification=False):
    frame = pd.DataFrame({'id': range(100), 'period': np.repeat(range(1, 6), 20),
                          'x': np.arange(100) % 7, 'target': np.arange(100) % 2})
    frame.to_csv(tmp_path / 'train.csv', index=False)
    frame.drop(columns='target').iloc[:10].assign(period=6).to_csv(tmp_path / 'test.csv', index=False)
    return TaskSpec(str(tmp_path/'task.md'), str(tmp_path/'train.csv'), str(tmp_path/'test.csv'),
                    'target', 'classification' if classification else 'regression',
                    'macro_f1' if classification else 'rmse', classification,
                    ['period', 'x'], ['id'], [], ['id', 'target'], 'fingerprint', classes=[0, 1])


def test_forward_cv_never_trains_on_future(tmp_path):
    spec = task(tmp_path)
    train, test = pd.read_csv(spec.train_path), pd.read_csv(spec.test_path)
    plan = make_validation_plan(spec, train, test, folds=5)
    assert plan['strategy'] == 'forward_group'
    assert len(plan['folds']) == 4
    assert plan['coverage'] == .8
    for fold in plan['folds']:
        assert train.iloc[fold['train']]['period'].max() < train.iloc[fold['validation']]['period'].min()
        assert set(fold['train']).isdisjoint(fold['validation'])
    assert make_validation_plan(spec, train, test, folds=5) == plan


def test_iid_oof_is_disjoint_complete_stratified(tmp_path):
    spec = task(tmp_path, True)
    train, test = pd.read_csv(spec.train_path), pd.read_csv(spec.test_path)
    plan = make_validation_plan(spec, train, test, folds=5, strategy='iid')
    valid = [i for fold in plan['folds'] for i in fold['validation']]
    assert sorted(valid) == list(range(100))
    assert plan['coverage'] == 1
    assert len(plan['folds']) == 5
    for fold in plan['folds']:
        assert len(train.iloc[fold['validation']]['target'].unique()) == 2


def test_host_oof_rows_have_same_order_as_manifest(tmp_path):
    spec = task(tmp_path)
    split = prepare_cv_split(spec, tmp_path/'run', folds=5)
    y = pd.read_csv(split['validation_target'])['target'].to_numpy()
    expected = pd.read_csv(spec.train_path).iloc[split['validation_indices']]['target'].to_numpy()
    np.testing.assert_array_equal(y, expected)
    assert split['mode'] == 'evaluate_cv'
    assert len(pd.read_csv(split['full_train_features'])) == 100


def test_future_rule_requires_semantics_and_three_groups(tmp_path):
    spec = task(tmp_path)
    train, test = pd.read_csv(spec.train_path), pd.read_csv(spec.test_path)
    train = train.rename(columns={'period': 'arbitrary'})
    test = test.rename(columns={'period': 'arbitrary'})
    spec.feature_columns = ['arbitrary', 'x']
    # Renaming removes temporal semantics; duplicate-safe group CV now applies.
    assert make_validation_plan(spec, train, test)['strategy'] == 'group'


def test_changed_validation_policy_changes_fingerprint(tmp_path):
    spec = task(tmp_path)
    train, test = pd.read_csv(spec.train_path), pd.read_csv(spec.test_path)
    assert make_validation_plan(spec, train, test)['fingerprint'] != make_validation_plan(spec, train, test, strategy='iid')['fingerprint']
