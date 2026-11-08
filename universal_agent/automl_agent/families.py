"""Deterministic, dependency-aware tabular model family candidates."""

from __future__ import annotations

import importlib.util
import textwrap


def _value(source, name, default=None):
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _catboost_available(profile):
    packages = _value(profile, "available_packages", {}) or {}
    if "catboost" in packages:
        value = packages["catboost"]
        if value is None or str(value).strip().lower() in {"", "unavailable", "none", "not installed"}:
            return False
        return True
    try:
        return importlib.util.find_spec("catboost") is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _source(default_n_jobs, model_family):
    if model_family == "CatBoost":
        return textwrap.dedent(f'''\
            import numpy as np
            import pandas as pd
            from catboost import CatBoostClassifier, CatBoostRegressor
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import FunctionTransformer

            DEFAULT_N_JOBS = {default_n_jobs}

            def _threads(context):
                return max(1, min(int(context.get("n_jobs", DEFAULT_N_JOBS)), 8))

            def _prepare_frame(X, categorical_columns):
                frame = pd.DataFrame(X).copy()
                for column in frame.columns:
                    if column in categorical_columns:
                        values = frame[column]
                        frame[column] = values.where(values.notna(), "__missing__").astype(str)
                    else:
                        values = pd.to_numeric(frame[column], errors="coerce")
                        frame[column] = values.replace([np.inf, -np.inf], np.nan)
                return frame

            def build_model(context):
                categorical = list(context.get("categorical_columns", []))
                seed = int(context.get("seed", 42))
                common = {{
                    "iterations": 150,
                    "random_seed": seed,
                    "thread_count": _threads(context),
                    "verbose": False,
                    "allow_writing_files": False,
                }}
                if context.get("task_type") == "classification":
                    classification = {{}}
                    if len(context.get("classes", [])) > 2:
                        classification["loss_function"] = "MultiClass"
                    estimator = CatBoostClassifier(**common, **classification, cat_features=categorical)
                else:
                    estimator = CatBoostRegressor(**common, cat_features=categorical)
                prepare = FunctionTransformer(
                    _prepare_frame,
                    kw_args={{"categorical_columns": categorical}},
                    validate=False,
                )
                return Pipeline([("prepare", prepare), ("model", estimator)])
        ''')

    if model_family == "ExtraTrees":
        return textwrap.dedent(f'''\
            import numpy as np
            import pandas as pd
            from sklearn.compose import ColumnTransformer
            from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

            DEFAULT_N_JOBS = {default_n_jobs}

            def _threads(context):
                return max(1, min(int(context.get("n_jobs", DEFAULT_N_JOBS)), 8))

            def _seed(context):
                return int(context.get("seed", 42))

            def _clean_numeric(X):
                frame = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
                values = frame.to_numpy(dtype=np.float64)
                return np.where(np.isfinite(values), values, np.nan)

            def _clean_categorical(X):
                frame = pd.DataFrame(X).copy()
                for column in frame.columns:
                    values = frame[column].astype("object")
                    frame[column] = values.where(pd.notna(values), "__missing__").astype(str)
                return frame

            def build_model(context):
                columns = list(context.get("feature_columns", []))
                categorical = [name for name in context.get("categorical_columns", []) if name in columns]
                numeric = [name for name in columns if name not in categorical]
                branches = []
                if numeric:
                    numeric_pipeline = Pipeline([
                        ("clean", FunctionTransformer(_clean_numeric, validate=False)),
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ])
                    branches.append(("numeric", numeric_pipeline, numeric))
                if categorical:
                    categorical_pipeline = Pipeline([
                        ("clean", FunctionTransformer(_clean_categorical, validate=False)),
                        ("encode", OneHotEncoder(
                            handle_unknown="ignore",
                            min_frequency=25,
                            max_categories=64,
                            sparse_output=True,
                            dtype=np.float32,
                        )),
                    ])
                    branches.append(("categorical", categorical_pipeline, categorical))
                preprocess = ColumnTransformer(branches, remainder="drop", sparse_threshold=1.0)
                common = {{
                    "n_estimators": 180,
                    "max_leaf_nodes": 128,
                    "min_samples_leaf": 3,
                    "max_features": 0.8,
                    "n_jobs": _threads(context),
                    "random_state": _seed(context),
                    "class_weight": None,
                }}
                if context.get("task_type") == "classification":
                    estimator = ExtraTreesClassifier(**common)
                else:
                    common.pop("class_weight")
                    estimator = ExtraTreesRegressor(**common)
                return Pipeline([("preprocess", preprocess), ("model", estimator)])
        ''')

    return textwrap.dedent(f'''\
        import numpy as np
        import pandas as pd
        from scipy.sparse import csr_matrix
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

        DEFAULT_N_JOBS = {default_n_jobs}

        def _threads(context):
            return max(1, min(int(context.get("n_jobs", DEFAULT_N_JOBS)), 8))

        def _seed(context):
            return int(context.get("seed", 42))

        def _clean_numeric(X):
            frame = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
            values = frame.to_numpy(dtype=np.float64)
            return np.where(np.isfinite(values), values, np.nan)

        def _clean_categorical(X):
            frame = pd.DataFrame(X).copy()
            for column in frame.columns:
                values = frame[column].astype("object")
                frame[column] = values.where(pd.notna(values), "__missing__").astype(str)
            return frame

        def build_model(context):
            columns = list(context.get("feature_columns", []))
            categorical = [name for name in context.get("categorical_columns", []) if name in columns]
            numeric = [name for name in columns if name not in categorical]
            branches = []
            if numeric:
                numeric_pipeline = Pipeline([
                    ("clean", FunctionTransformer(_clean_numeric, validate=False)),
                    ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scale", StandardScaler(with_mean=False)),
                    ("sparse", FunctionTransformer(csr_matrix, validate=False, accept_sparse=True)),
                ])
                branches.append(("numeric", numeric_pipeline, numeric))
            if categorical:
                categorical_pipeline = Pipeline([
                    ("clean", FunctionTransformer(_clean_categorical, validate=False)),
                    ("encode", OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=25,
                        max_categories=64,
                        sparse_output=True,
                        dtype=np.float32,
                    )),
                ])
                branches.append(("categorical", categorical_pipeline, categorical))
            preprocess = ColumnTransformer(branches, remainder="drop", sparse_threshold=1.0)
            if context.get("task_type") == "classification":
                estimator = LogisticRegression(
                    C=0.5,
                    max_iter=300,
                    solver="saga",
                    random_state=_seed(context),
                    n_jobs=_threads(context),
                    class_weight=None,
                )
            else:
                estimator = Ridge(alpha=1.0)
            return Pipeline([("preprocess", preprocess), ("model", estimator)])
    ''')


def family_candidates(task, profile, n_jobs=4, limit=3):
    """Return stable, complete candidate dictionaries for usable CPU families."""
    try:
        default_n_jobs = max(1, min(int(n_jobs), 8))
    except (TypeError, ValueError):
        default_n_jobs = 4
    try:
        maximum = max(0, int(limit))
    except (TypeError, ValueError):
        maximum = 3
    if maximum == 0:
        return []

    metric = str(_value(task, "metric", _value(profile, "metric", "")))
    task_type = str(_value(task, "task_type", _value(profile, "task_type", "classification")))
    families = []
    if _catboost_available(profile):
        families.append((
            "CatBoost",
            "Compare a bounded, native-categorical boosted-tree model on the shared split.",
            {
                "metric": metric,
                "task_type": task_type,
                "n_jobs": default_n_jobs,
                "iterations": 150,
                "depth": "CatBoost default",
                "learning_rate": "CatBoost automatic default",
                "multiclass_loss": "MultiClass when context has more than two classes",
                "class_weights": None,
            },
        ))
    families.extend([
        (
            "ExtraTrees",
            "Compare randomized bounded-depth trees after frequency-limited categorical encoding.",
            {
                "metric": metric,
                "task_type": task_type,
                "n_jobs": default_n_jobs,
                "n_estimators": 180,
                "max_leaf_nodes": 128,
                "class_weights": None,
            },
        ),
        (
            "Linear",
            "Compare a sparse one-hot linear model with imputation and uncentered numeric scaling.",
            {
                "metric": metric,
                "task_type": task_type,
                "n_jobs": default_n_jobs,
                "categorical_encoding": "sparse one-hot, capped at 64 categories per feature",
                "class_weights": None,
            },
        ),
    ])
    return [
        {
            "family": family,
            "hypothesis": {
                "CatBoost": "Сравнить CatBoost с нативными категориями и 150 итерациями на общем holdout.",
                "ExtraTrees": "Сравнить ExtraTrees с ограниченной сложностью деревьев и кодированием категорий.",
                "Linear": "Проверить линейную модель после масштабирования чисел и разреженного one-hot.",
            }[family],
            "code": _source(default_n_jobs, family),
            "decision": {**decision, "model_family": family,
                         "observations": [f"Тип задачи: {task_type}; метрика: {metric}; категориальных признаков: {len(_value(task, 'categorical_columns', []) or [])}"],
                         "rationale": {
                             "CatBoost": "Нативная обработка категорий; сохраняем настройки ранее проверенного baseline.",
                             "ExtraTrees": "Проверить другой тип ансамбля с контролируемыми памятью и сложностью деревьев.",
                             "Linear": "Проверить более простую зависимость и стоимость обучения на разреженном представлении.",
                         }[family],
                         "alternatives": [other[0] for other in families if other[0] != family],
                         "expected_effect": "Измерить качество семейства; улучшение заранее неизвестно.",
                         "risks": ["Один holdout не доказывает статистически значимое преимущество; контролируем время и память."],
                         "previous_result_assessment": "Начальное сравнение: результаты независимых кандидатов ещё не известны."},
        }
        for family, hypothesis, decision in families[:maximum]
    ]
