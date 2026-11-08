"""Deterministic OOF plans; future-only periods require forward validation."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold


def _period_values(series):
    numeric = pd.to_numeric(series, errors='coerce')
    if numeric.notna().sum() == series.notna().sum():
        return numeric
    dates = pd.to_datetime(series, errors='coerce', utc=True)
    return dates if dates.notna().sum() == series.notna().sum() else None


def future_time_column(task, train, test):
    """Require time semantics AND observed future-only test distribution."""
    from .data_diagnostics import _temporal_candidates
    candidates = _temporal_candidates(train, test, task.feature_columns)
    return candidates[0]['feature'] if candidates else None


def make_validation_plan(task, train, test, folds=5, strategy='auto', time_column=None, diagnostics=None):
    if not 2 <= folds <= 10:
        raise ValueError('CV folds must be between 2 and 10')
    if strategy not in {'auto', 'iid', 'temporal'}:
        raise ValueError('Unknown validation strategy')
    column = time_column
    if column is not None and column not in task.feature_columns:
        raise ValueError('Time column must be a model feature')
    if column is None and strategy != 'iid':
        column = future_time_column(task, train, test)
    if strategy == 'temporal' and column is None:
        raise ValueError('Temporal validation requires a time column or a confirmed future-only feature')
    rows = np.arange(len(train))
    warnings = []
    partitions = []
    if column and strategy != 'iid':
        values = _period_values(train[column])
        if values is None:
            raise ValueError('Time column must contain ordered numeric or datetime values')
        groups = sorted(values.dropna().unique())
        if len(groups) < 3:
            raise ValueError('Temporal CV requires at least three distinct training periods')
        for group in groups[-min(folds, len(groups)-1):]:
            fit, valid = rows[(values < group).to_numpy()], rows[(values == group).to_numpy()]
            if len(fit) and len(valid):
                partitions.append((fit, valid))
        policy = 'forward_group'
        warnings.append('The earliest training period has no OOF prediction; future periods never enter training.')
        if values.isna().any():
            warnings.append('Rows with missing time are excluded from CV; final fitting still uses all rows.')
    else:
        column = None
        count = min(folds, len(train))
        if count < 2:
            raise ValueError('CV needs at least two rows')
        y = train[task.target]
        counts = y.value_counts()
        feature_groups = pd.util.hash_pandas_object(train[task.feature_columns], index=False).to_numpy()
        unique_groups = len(np.unique(feature_groups))
        if strategy == 'auto' and unique_groups < len(train) and unique_groups >= 2:
            count = min(count, unique_groups)
            if task.task_type == 'classification' and len(counts) > 1:
                class_groups = pd.DataFrame({'y':y.to_numpy(), 'g':feature_groups}).groupby('y')['g'].nunique()
                if class_groups.min() >= 2:
                    count = min(count, int(class_groups.min()))
                    splitter = StratifiedGroupKFold(count, shuffle=True, random_state=task.seed)
                    policy = 'stratified_group'
                else:
                    splitter = GroupKFold(count)
                    policy = 'group'
                    warnings.append('Some classes occur in only one duplicate group; a fold can miss a class.')
            else:
                splitter = GroupKFold(count)
                policy = 'group'
            partitions = list(splitter.split(rows, y, feature_groups))
            warnings.append('Identical feature rows share a validation fold; IDs and target are excluded from grouping.')
        elif strategy == 'auto' and unique_groups == 1:
            raise ValueError('Independent CV is impossible: every training feature row is identical; use explicit iid only if appropriate.')
        elif task.task_type == 'classification' and len(counts) > 1 and counts.min() >= 2:
            count = min(count, int(counts.min()))
            splitter = StratifiedKFold(count, shuffle=True, random_state=task.seed)
            policy = 'stratified'
            partitions = list(splitter.split(rows, y))
        else:
            splitter = KFold(count, shuffle=True, random_state=task.seed)
            policy = 'kfold'
            partitions = list(splitter.split(rows))
            if task.task_type == 'classification':
                warnings.append('Rare classes prevent stratification; a fold can miss a class.')
    fold_list = [{'fold': i, 'train': a.tolist(), 'validation': b.tolist()} for i, (a, b) in enumerate(partitions)]
    eligible = sorted(i for _, valid in partitions for i in valid.tolist())
    if len(eligible) != len(set(eligible)):
        raise ValueError('Validation folds overlap')
    payload = {'version': 1, 'data': task.data_fingerprint, 'seed': task.seed, 'strategy': policy,
               'time_column': column, 'requested_folds': folds, 'folds': fold_list,
               'validation_indices': eligible, 'coverage': len(eligible)/len(train), 'warnings': warnings}
    payload['fingerprint'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def prepare_cv_split(task, run_dir, *, folds=5, strategy='auto', time_column=None, diagnostics=None):
    from .journal import atomic_write_json
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    train, test = pd.read_csv(task.train_path, low_memory=False), pd.read_csv(task.test_path, low_memory=False)
    plan = make_validation_plan(task, train, test, folds, strategy, time_column, diagnostics)
    eligible = plan['validation_indices']
    cache = run_dir/'data'
    cache.mkdir(exist_ok=True)
    frames = {'train_features': train[task.feature_columns], 'train_target': train[[task.target]],
              'validation_features': train.iloc[eligible][task.feature_columns],
              'validation_target': train.iloc[eligible][[task.target]], 'test_features': test[task.feature_columns]}
    paths = {}
    for name, frame in frames.items():
        path = cache/f'{name}.csv'
        frame.to_csv(path, index=False)
        paths[name] = str(path)
    paths['full_train_features'], paths['full_train_target'] = paths['train_features'], paths['train_target']
    plan_path = run_dir/'cv_plan.json'
    atomic_write_json(plan_path, plan)
    return {**paths, 'mode': 'evaluate_cv', 'cv_plan': str(plan_path),
            'split_fingerprint': plan['fingerprint'], 'validation_indices': eligible,
            'strategy': plan['strategy'], 'time_column': plan['time_column'],
            'fold_count': len(plan['folds']), 'coverage': plan['coverage'], 'warnings': plan['warnings']}


def oof_metric_summary(task, split, trial_dir):
    """Host-computed fold metrics; candidate-reported scores are never trusted."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
    path = Path(trial_dir)/'oof_fold_ids.npy'
    if not path.exists():
        return {'strategy':'legacy_holdout'}
    ids = np.load(path, allow_pickle=False)
    y = pd.read_csv(split['validation_target'])[task.target].to_numpy()
    predicted = pd.read_csv(Path(trial_dir)/'valid_predictions.csv')[task.target].to_numpy()
    rows = []
    for fold in np.unique(ids):
        mask = ids == fold
        if task.metric == 'rmse':
            value = np.sqrt(mean_squared_error(y[mask], predicted[mask]))
        elif task.metric == 'mae':
            value = mean_absolute_error(y[mask], predicted[mask])
        elif task.metric == 'mse':
            value = mean_squared_error(y[mask], predicted[mask])
        elif task.metric == 'mape':
            value = mean_absolute_percentage_error(y[mask], predicted[mask])
        elif task.metric == 'rmsle':
            value = np.sqrt(mean_squared_error(np.log1p(y[mask]), np.log1p(np.maximum(predicted[mask], 0.)))) if (y[mask] >= 0).all() else None
        elif task.metric == 'r2':
            value = r2_score(y[mask], predicted[mask]) if mask.sum() > 1 else None
        elif task.metric == 'balanced_accuracy':
            value = balanced_accuracy_score(y[mask], predicted[mask])
        elif task.metric in {'macro_f1','weighted_f1','accuracy','f1'}:
            if task.metric == 'accuracy':
                value = accuracy_score(y[mask], predicted[mask])
            else:
                average = {'macro_f1':'macro','weighted_f1':'weighted','f1':'binary'}[task.metric]
                from .evaluation import _positive_class
                kwargs = {'labels':task.classes} if average != 'binary' else {'pos_label':_positive_class(task.classes, getattr(task,'positive_class',None))}
                value = f1_score(y[mask], predicted[mask], average=average, zero_division=0, **kwargs)
        else:
            value = None
        rows.append({'fold':int(fold), 'rows':int(mask.sum()), 'metric':float(value) if value is not None and np.isfinite(value) else None})
    values = [r['metric'] for r in rows if r['metric'] is not None]
    return {'strategy':split['strategy'], 'coverage':split['coverage'], 'folds':rows,
            'mean':float(np.mean(values)) if values else None, 'std':float(np.std(values)) if values else None,
            'warning':'Model selection is adaptive; OOF scores are not an independent final test.'}
