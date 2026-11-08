"""Small reproducible fallback search inspired by rakovskiy.researcher.

These are host proposals, never counted as experiments proposed by an LLM.
The normal recipe validator, same CV, deduplication and deadlines still apply.
"""
from .recipes import normalize_recipe, recipe_fingerprint


_SPACE = {
    'CatBoost': {'depth':[4,5,7,8], 'l2_leaf_reg':[1.,3.,8.,15.], 'learning_rate':[.025,.04,.08]},
    'LightGBM': {'num_leaves':[15,47,63,95], 'min_child_samples':[10,30,60,100], 'reg_lambda':[.1,1.,5.,15.]},
    'XGBoost': {'max_depth':[3,4,5,8], 'min_child_weight':[1.,3.,8.], 'reg_lambda':[.1,1.,5.,15.]},
    'ExtraTrees': {'min_samples_leaf':[1,2,5,10], 'max_features':[.5,.7,1.], 'max_leaf_nodes':[64,256,512]},
    'Linear': {'C':[.1,1.,3.,10.]},
}


def next_search_proposal(results, task_type, maximize, seed, step):
    measured = [r for r in results if r.status == 'ok' and r.recipe and r.metric is not None]
    if not measured:
        return None
    measured.sort(key=lambda r: r.metric, reverse=maximize)
    seen = {recipe_fingerprint(r.recipe, task_type) for r in results if r.recipe}
    # Keep refinement near the leaders: include at most the three families
    # with the best measured incumbent, and enumerate their bounded grids in
    # stable score/key/value order instead of drawing blind random mutations.
    family_best = {}
    for result in measured:
        family_best.setdefault(result.recipe['family'], result)
    best_families = {
        result.recipe['family']
        for result in list(family_best.values())[:3]
    }
    measured = [r for r in measured if r.recipe['family'] in best_families]
    for base in measured:
        space = dict(_SPACE[base.recipe['family']])
        if task_type == 'regression' and base.recipe['family'] == 'Linear':
            space = {'alpha': [.1, 1., 10.]}
        if task_type == 'classification' and base.recipe['family'] == 'CatBoost':
            space['auto_class_weights'] = [None, 'SqrtBalanced', 'Balanced']
        for key in sorted(space):
            for value in space[key]:
                recipe = normalize_recipe(base.recipe, task_type)
                recipe['parameters'][key] = value
                try:
                    recipe = normalize_recipe(recipe, task_type)
                except ValueError:
                    continue
                if recipe_fingerprint(recipe, task_type) not in seen:
                    return {'base_trial_id':base.trial_id, 'parameters':{key:value},
                            'hypothesis':f'Воспроизводимый локальный поиск: проверить {key}={value} у {recipe["family"]} на той же валидации.'}
    return None


def next_feature_or_weight_proposal(
    results,
    task_type,
    metric,
    feature_columns,
    categorical_columns,
    seed,
    step,
):
    """Return the next deduplicated, measured-base feature/weight candidate.

    Class weighting is restricted to explicit F1 objectives and a measured
    LightGBM baseline. Feature operations use only declared columns; their
    fitted statistics are learned by the existing fold-local transformer.
    """
    from .recipes import normalize_recipe, recipe_fingerprint

    task_type = str(task_type).strip().lower()
    metric = str(metric).strip().lower()
    seen = {
        recipe_fingerprint(result.recipe, task_type)
        for result in results
        if getattr(result, 'recipe', None)
    }
    measured = [
        result for result in results
        if getattr(result, 'status', None) == 'ok'
        and getattr(result, 'recipe', None)
        and getattr(result, 'metric', None) is not None
    ]
    if not measured:
        return None
    maximize = metric in {'r2', 'macro_f1', 'weighted_f1', 'f1', 'accuracy', 'balanced_accuracy', 'roc_auc'}
    measured.sort(key=lambda result: result.metric, reverse=maximize)

    # Probe the donor's soft class balance only where F1 is the task objective.
    if task_type == 'classification' and metric in {'macro_f1', 'f1'}:
        lightgbm = [r for r in measured if r.recipe.get('family') == 'LightGBM']
        if lightgbm:
            base = lightgbm[0]
            candidate = normalize_recipe(base.recipe, task_type)
            options = dict(candidate.get('fit_options', {}))
            if options.get('sample_weight') is None:
                options['sample_weight'] = 'sqrt_balanced'
                candidate['fit_options'] = options
                fingerprint = recipe_fingerprint(candidate, task_type)
                if fingerprint not in seen:
                    return {
                        'base_trial_id': base.trial_id,
                        'parameters': {},
                        'fit_options': {'sample_weight': 'sqrt_balanced'},
                        'hypothesis': (
                            f'Проверить sqrt_balanced на измеренном LightGBM-кандидате '
                            f'для метрики {metric} (seed={int(seed)}, шаг={int(step)}).'
                        ),
                    }

    # Once this feature-chain family has been tried anywhere in the run,
    # don't repeat it against another baseline. Failed trials count too:
    # their recipes remain in the durable result list. LLM-proposed frequency
    # encodings or missing flags count as the same exploration.
    for result in results:
        recipe = getattr(result, 'recipe', None)
        preprocessing = recipe.get('preprocessing', {}) if isinstance(recipe, dict) else {}
        feature_ops = preprocessing.get('feature_ops', []) if isinstance(preprocessing, dict) else []
        if any(
            isinstance(operation, dict)
            and operation.get('op') in {'freq_encode', 'missing_flags'}
            for operation in feature_ops
        ):
            return None

    columns = [c for c in feature_columns if isinstance(c, str) and c]
    categories = [c for c in categorical_columns if c in columns]
    numeric = [c for c in columns if c not in set(categories)]
    operations = []
    if categories:
        operations.append({'op': 'freq_encode', 'cols': categories[:32]})
    if numeric:
        operations.append({'op': 'missing_flags', 'cols': numeric[:32]})
    if not operations:
        return None

    for base in measured:
        candidate = normalize_recipe(base.recipe, task_type)
        preprocessing = dict(candidate.get('preprocessing', {}))
        existing = preprocessing.get('feature_ops', [])
        if existing:
            continue
        preprocessing['feature_ops'] = operations
        candidate['preprocessing'] = preprocessing
        fingerprint = recipe_fingerprint(candidate, task_type)
        if fingerprint in seen:
            continue
        return {
            'base_trial_id': base.trial_id,
            'parameters': {},
            'preprocessing': {'feature_ops': operations},
            'hypothesis': (
                f'Проверить fit-only частотные признаки категорий и индикаторы пропусков '
                f'на измеренной базовой модели (seed={int(seed)}, шаг={int(step)}).'
            ),
        }
    return None
