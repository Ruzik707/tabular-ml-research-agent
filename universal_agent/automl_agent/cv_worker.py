"""Worker-owned fold training; only the host selects models and scores OOF."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

from .prediction import PredictionEnsemble, aligned_probabilities


def _inner_time_split(X, column):
    if column is None:
        return None
    from .validation import _period_values
    values = _period_values(X[column])
    if values is None or values.dropna().nunique() < 2:
        return None
    last = values.max()
    fit, valid = np.flatnonzero(values < last), np.flatnonzero(values == last)
    return (fit, valid) if len(fit) >= 20 and len(valid) >= 10 else None


def _preflight(builder, X, y, context, directory):
    from .training import fit_model
    from .worker import _atomic_json, _validate_predictions
    started = time.monotonic()
    # Include rare classes in the small train-only sample; no OOF labels are used.
    indices = list(range(min(200, len(X))))
    if context['task_type'] == 'classification':
        indices = sorted(set(indices + [int(np.flatnonzero(y.to_numpy() == c)[0]) for c in pd.unique(y)]))
    try:
        fitted, _ = fit_model(builder(context), X.iloc[indices], y.iloc[indices].to_numpy(), context=context,
                              options={}, iteration_override=10)
        probe = X.iloc[indices[:min(20, len(indices))]]
        before = np.asarray(fitted.predict(probe)).reshape(-1)
        _validate_predictions(before, context['task_type'], 'preflight')
        restored = cloudpickle.loads(cloudpickle.dumps(fitted))
        np.testing.assert_array_equal(before, np.asarray(restored.predict(probe)).reshape(-1))
        result = {'status':'ok', 'rows':len(indices), 'seconds':time.monotonic()-started}
    except Exception as exc:
        result = {'status':'error', 'rows':len(indices), 'seconds':time.monotonic()-started,
                  'error':f'{type(exc).__name__}: {exc}'[:2000]}
        _atomic_json(result, directory/'preflight.json')
        raise ValueError('Preflight failed before full CV: ' + result['error']) from exc
    _atomic_json(result, directory/'preflight.json')
    return result


def run_cv_manifest(manifest):
    from .training import fit_model
    from .worker import (_load_features, _load_model, _atomic_csv, _atomic_json, _atomic_npy,
                         _validate_predictions, _capture_native_parameter_types, _estimator_details)
    inputs, context = manifest['inputs'], manifest['context']
    if 'validation_target' in inputs or 'validation_target' in context:
        raise ValueError('validation_target belongs to host')
    directory = Path(manifest['artifact_dir']).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    columns, categorical = context['feature_columns'], set(context.get('categorical_columns', []))
    X = _load_features(inputs['train_features'], columns, categorical)
    target = context['target']
    yframe = pd.read_csv(inputs['train_target'], low_memory=False)
    if list(yframe.columns) != [target] or len(yframe) != len(X) or yframe[target].isna().any():
        raise ValueError('Invalid train target schema or rows')
    y = yframe[target]
    test = _load_features(inputs['test_features'], columns, categorical)
    plan = json.loads(Path(manifest['cv_plan']).read_text())
    eligible = np.asarray(plan['validation_indices'], dtype=int)
    _capture_native_parameter_types()
    _, builder = _load_model(Path(manifest['code_path']))
    preflight = _preflight(builder, X.iloc[plan['folds'][0]['train']].reset_index(drop=True),
                           y.iloc[plan['folds'][0]['train']].reset_index(drop=True), context, directory) if manifest.get('preflight') else {'status':'not_requested'}
    classification = context['task_type'] == 'classification'
    classes = list(context.get('classes') or sorted(pd.unique(y).tolist())) if classification else []
    oof = np.empty(len(X), dtype=object if classification else float)
    oof_proba = np.zeros((len(X), len(classes))) if classification else None
    models, summaries, importance_rows = [], [], []
    all_proba = True
    for fold in plan['folds']:
        fit, valid = np.asarray(fold['train']), np.asarray(fold['validation'])
        if np.intersect1d(fit, valid).size:
            raise ValueError('CV train and validation overlap')
        train_x, train_y = X.iloc[fit].reset_index(drop=True), y.iloc[fit].to_numpy()
        inner = _inner_time_split(train_x, plan.get('time_column'))
        model, summary = fit_model(builder(context), train_x, train_y, context=context,
                                   options=manifest.get('fit_options'), inner_split=inner)
        predicted = np.asarray(model.predict(X.iloc[valid])).reshape(-1)
        _validate_predictions(predicted, context['task_type'], 'OOF')
        if len(predicted) != len(valid):
            raise ValueError('OOF length mismatch')
        oof[valid] = predicted
        if classification and callable(getattr(model, 'predict_proba', None)):
            oof_proba[valid] = aligned_probabilities(model, X.iloc[valid], classes)
        elif classification:
            all_proba = False
        models.append(model)
        summary = {**summary, 'fold':fold['fold'], 'validation_rows':len(valid)}
        summaries.append(summary)
        _atomic_json(summary, directory/f"fold_{fold['fold']}.json")
        try:
            from .data_diagnostics import feature_importance
            importance_rows.append(feature_importance(model, context))
        except (ImportError, AttributeError):
            pass
    ensemble = PredictionEnsemble(models, context['task_type'], classes)
    test_predictions = ensemble.predict(test)
    _validate_predictions(test_predictions, context['task_type'], 'test')
    _atomic_csv(pd.DataFrame({target:oof[eligible]}), directory/'valid_predictions.csv')
    _atomic_csv(pd.DataFrame({target:test_predictions}), directory/'test_predictions.csv')
    np.save(directory/'oof_indices.npy', eligible, allow_pickle=False)
    fold_ids = np.full(len(X), -1)
    for f in plan['folds']:
        fold_ids[f['validation']] = f['fold']
    np.save(directory/'oof_fold_ids.npy', fold_ids[eligible], allow_pickle=False)
    probability_required = context.get("output_kind") in {"binary_probability", "class_probabilities"} or context.get("metric") in {"roc_auc", "log_loss"}
    if probability_required and not all_proba:
        raise ValueError("Probability output/metric requires predict_proba from every CV fold")
    probability_written = classification and all_proba
    if probability_written:
        _atomic_npy(oof_proba[eligible], directory/'valid_probabilities.npy')
        test_probabilities = ensemble.predict_proba(test)
        if (not np.isfinite(test_probabilities).all() or (test_probabilities < 0).any()
                or not np.allclose(test_probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6)):
            raise ValueError("CV ensemble returned invalid test probabilities")
        _atomic_npy(test_probabilities, directory/'test_probabilities.npy')
        _atomic_json({'classes':classes}, directory/'probability_classes.json')
    details = _estimator_details(models[-1])
    best_trees = [s['best_iteration'] for s in summaries if s.get('best_iteration')]
    # Temporal validation's most recent period best reflects final train size.
    final_trees = (best_trees[-1] if plan['strategy']=='forward_group' else int(np.median(best_trees))) if best_trees else None
    last = summaries[-1]
    estimate = last['refit_seconds'] * len(X)/last['train_rows'] if last.get('refit_seconds') else None
    from .worker import _merge_training_parameters
    _merge_training_parameters(details, last)
    details['training'] = {'folds':summaries, 'final_iterations':final_trees,
                           'estimated_refit_seconds':estimate, 'oof_coverage':plan['coverage'],
                           'validation_strategy':plan['strategy'], 'split_fingerprint':plan['fingerprint']}
    usable = [row for row in importance_rows if row.get('status') == 'ok']
    if usable:
        combined = {}
        for row in usable:
            for item in row['items']:
                combined[item['feature']] = combined.get(item['feature'], 0.) + item['importance']/len(usable)
        details['feature_importance'] = {**usable[0], 'method':'mean_fold_native_importance',
            'folds':len(usable), 'items':sorted([{'feature':k,'importance':v} for k,v in combined.items()], key=lambda row:-row['importance'])}
    else:
        details['feature_importance'] = importance_rows[-1] if importance_rows else {'status':'unavailable'}
    if manifest.get('save_model'):
        with (directory/'model.joblib').open('wb') as stream:
            cloudpickle.dump(ensemble, stream)
    result = {'status':'ok', 'mode':'evaluate_cv', 'train_rows':len(X), 'validation_rows':len(eligible),
              'test_rows':len(test), 'target':target, 'estimator_details':details,
              'preflight':preflight,'split_fingerprint':plan['fingerprint'],
              'probabilities':'valid_probabilities.npy' if probability_written else None,
              'test_probabilities':'test_probabilities.npy' if probability_written else None,
              'model_path':str(directory/'model.joblib') if manifest.get('save_model') else None}
    _atomic_json(result, directory/'worker_result.json')
    return result
