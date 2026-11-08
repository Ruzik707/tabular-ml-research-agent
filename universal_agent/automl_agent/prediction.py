"""Serializable prediction wrappers for aligned fold and final ensembles."""
from __future__ import annotations

import numpy as np


def aligned_probabilities(model, X, classes):
    p = np.asarray(model.predict_proba(X), dtype=float)
    axis = list(model.classes_)
    if p.ndim != 2 or p.shape != (len(X), len(axis)) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1., atol=1e-6, rtol=1e-6):
        raise ValueError('Invalid probability matrix')
    if len(axis) != len(set(axis)) or any(label not in classes for label in axis):
        raise ValueError('Invalid probability class axis')
    aligned = np.zeros((len(X), len(classes)))
    for i, label in enumerate(classes):
        if label in axis:
            aligned[:, i] = p[:, axis.index(label)]
    total = aligned.sum(axis=1)
    if (total <= 0).any():
        raise ValueError('Empty probability row after class alignment')
    return aligned / total[:, None]


class PredictionEnsemble:
    def __init__(self, models, task_type, classes=None, weights=None, class_multipliers=None,
                 binary_threshold=None, positive_class=None):
        self.models = models
        self.model = models[-1]  # bounded, known metadata traversal link
        self.task_type = task_type
        self.classes_ = np.asarray(list(classes) if classes is not None else [], dtype=object)
        self.weights = np.asarray(weights if weights is not None else [1/len(models)]*len(models), dtype=float)
        self.class_multipliers = np.asarray(class_multipliers if class_multipliers is not None else np.ones(len(self.classes_)))
        self.binary_threshold = binary_threshold
        self.positive_class = positive_class
        if binary_threshold is not None:
            if task_type != 'classification' or class_multipliers is not None:
                raise ValueError('binary_threshold requires binary classification without class_multipliers')
            if not all(callable(getattr(model, 'predict_proba', None)) for model in models):
                raise ValueError('binary_threshold requires every member to provide predict_proba')
            from .postprocessing import binary_threshold_labels
            binary_threshold_labels(np.empty((0, len(self.classes_))), self.classes_, binary_threshold, positive_class)

    def predict_proba(self, X):
        p = sum(w * aligned_probabilities(m, X, list(self.classes_)) for w, m in zip(self.weights, self.models))
        p = p * self.class_multipliers
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        if self.task_type == 'regression':
            return sum(w*np.asarray(m.predict(X)).reshape(-1) for w,m in zip(self.weights,self.models))
        if all(callable(getattr(m, 'predict_proba', None)) for m in self.models):
            probabilities = self.predict_proba(X)
            # getattr keeps previously serialized ensembles usable after update.
            if getattr(self, 'binary_threshold', None) is not None:
                from .postprocessing import binary_threshold_labels
                return binary_threshold_labels(probabilities, self.classes_, self.binary_threshold, self.positive_class)
            return self.classes_[probabilities.argmax(axis=1)]
        votes = np.array([np.asarray(m.predict(X)).reshape(-1) for m in self.models])
        counts = np.stack([np.sum(self.weights[:, None]*(votes == c), axis=0) for c in self.classes_], axis=1)
        return self.classes_[counts.argmax(axis=1)]
