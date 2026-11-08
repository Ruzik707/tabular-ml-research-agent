"""Shared on-disk contracts; providers and workers never own model selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TaskSpec:
    task_path: str
    train_path: str
    test_path: str
    target: str
    task_type: str
    metric: str
    maximize: bool
    feature_columns: list[str]
    id_columns: list[str]
    categorical_columns: list[str]
    output_columns: list[str]
    data_fingerprint: str
    description: str = ""
    dictionary_path: str | None = None
    sample_path: str | None = None
    classes: list[Any] = field(default_factory=list)
    seed: int = 42
    validation_fraction: float = 0.2
    output_kind: str = "labels"
    probability_column_classes: dict[str, Any] = field(default_factory=dict)
    positive_class: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSpec:
        return cls(**data)

    def model_context(self, n_jobs: int = 4) -> dict[str, Any]:
        """Only modelling facts: no labels or original dataset paths."""
        return {
            "task_type": self.task_type,
            "metric": self.metric,
            "target": self.target,
            "feature_columns": self.feature_columns,
            "categorical_columns": self.categorical_columns,
            "classes": self.classes,
            "output_kind": self.output_kind,
            "probability_column_classes": self.probability_column_classes,
            "positive_class": self.positive_class,
            "seed": self.seed,
            "n_jobs": n_jobs,
        }


@dataclass
class ExperimentResult:
    trial_id: str
    status: str
    artifact_dir: str
    code_hash: str
    hypothesis: str
    metric: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    error: str | None = None
    is_baseline: bool = False
    prediction_path: str | None = None
    model_path: str | None = None
    decision: dict[str, Any] = field(default_factory=dict)
    model_details: dict[str, Any] = field(default_factory=dict)
    origin: str | None = None
    recipe: dict[str, Any] | None = None
    recipe_hash: str | None = None
    parent_trial_id: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentResult:
        return cls(**data)


@dataclass
class ProcessResult:
    status: str
    exit_code: int | None
    stdout: str
    stderr: str
    seconds: float
    peak_memory_mb: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    pid: int | None = None
    n_jobs: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelResponse:
    message: dict[str, Any]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
