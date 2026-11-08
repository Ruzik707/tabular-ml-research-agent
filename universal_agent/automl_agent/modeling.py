"""Generate a self-contained, data-agnostic starting estimator for the LLM."""

from __future__ import annotations

from .contracts import TaskSpec


def baseline_code(task: TaskSpec, n_jobs: int = 4, iterations: int = 150) -> str:
    """Return candidate source with a CatBoost model and sklearn fallback.

    The emitted file deliberately defines ``build_model(context)`` rather than
    fitting here. The worker owns fitting and prediction, while the LLM may
    replace this complete implementation with another sklearn-compatible
    estimator or pipeline.
    """
    jobs = max(1, int(n_jobs))
    rounds = max(1, int(iterations))
    return f'''# Generic tabular baseline. Replace build_model(context) to try a new model.
import numpy as np
import pandas as pd


def _clean_frame(X, categorical_columns):
    frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    categories = [name for name in categorical_columns if name in frame.columns]
    numeric = [name for name in frame.columns if name not in categories]
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    for name in categories:
        values = frame[name].astype("object")
        frame[name] = values.where(pd.notna(values), "__MISSING__").astype(str)
    return frame, categories, numeric


class _AdaptiveModel:
    def __init__(self, context):
        self.context = dict(context)
        self.categories = list(self.context.get("categorical_columns", []))
        self.kind = self.context.get("task_type", "regression")
        self.model = None
        self._catboost = False
        self.fallback_reason = None

    def fit(self, X, y):
        self.fallback_reason = None
        self._catboost = False
        frame, categories, numeric = _clean_frame(X, self.categories)
        self.categories = categories
        self.feature_names_in_ = list(frame.columns)
        y = np.asarray(y).reshape(-1)
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
            klass = CatBoostClassifier if self.kind == "classification" else CatBoostRegressor
            params = dict(
                iterations={rounds},
                random_seed=int(self.context.get("seed", 42)),
                thread_count=max(1, int(self.context.get("n_jobs", {jobs}))),
                verbose=False,
                allow_writing_files=False,
            )
            if self.kind == "classification" and len(np.unique(y)) > 2:
                params["loss_function"] = "MultiClass"
            try:
                self.model = klass(**params)
                self.model.fit(frame, y, cat_features=categories)
                self._catboost = True
            except Exception as exc:
                self.fallback_reason = ("CatBoost: " + str(exc))[:1000]
                # Keep the baseline usable when a particular CatBoost build or
                # fold cannot represent the observed target/category shape.
                self.model = self._make_sklearn(frame, categories, numeric)
                self.model.fit(frame, y)
        except ImportError as exc:
            self.fallback_reason = ("CatBoost: " + str(exc))[:1000]
            self.model = self._make_sklearn(frame, categories, numeric)
            self.model.fit(frame, y)
        return self

    def _make_sklearn(self, frame, categories, numeric):
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        transformers = []
        if numeric:
            transformers.append(("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric))
        if categories:
            categorical = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ])
            transformers.append(("categorical", categorical, categories))
        if not transformers:
            raise ValueError("No usable feature columns were provided")
        transform = ColumnTransformer(transformers, remainder="drop")
        estimator_type = ExtraTreesClassifier if self.kind == "classification" else ExtraTreesRegressor
        estimator = estimator_type(
            n_estimators={rounds}, random_state=int(self.context.get("seed", 42)),
            n_jobs=max(1, int(self.context.get("n_jobs", {jobs}))),
        )
        return Pipeline([("preprocess", transform), ("model", estimator)])

    def _prepare(self, X):
        frame, _, _ = _clean_frame(X, self.categories)
        # Recreate fit-time column order and give missing fields a neutral value.
        if hasattr(self, "feature_names_in_"):
            frame = frame.reindex(columns=self.feature_names_in_)
        return frame

    def predict(self, X):
        return self.model.predict(self._prepare(X))

    def predict_proba(self, X):
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("model does not provide predict_proba")
        return self.model.predict_proba(self._prepare(X))

    @property
    def classes_(self):
        return self.model.classes_


def build_model(context: dict):
    return _AdaptiveModel(context)
'''
