"""CLI configuration; kept independent from expensive ML imports."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit


@dataclass
class Config:
    task_path: Path
    provider: str
    base_url: str
    model: str
    api_key: str
    time_budget: float = 10800
    trial_timeout: float = 1800
    llm_timeout: float = 180
    max_trials: int | None = None
    max_steps: int | None = None
    max_tokens: int = 2000000
    response_tokens: int = 6144
    n_jobs: int = 4
    parallel_trials: int = 2
    family_trials: int = 4
    search_trials: int = 0
    experiment_mode: str = "recipe"
    memory_limit_mb: int = 12000
    context_chars: int = 48000
    seed: int = 42
    validation_fraction: float = 0.2
    cv_folds: int = 5
    validation_strategy: str = "auto"
    time_column: str | None = None
    run_dir: Path | None = None
    resume: bool = False
    finalize_only: bool = False
    offline: bool = False
    target: str | None = None
    task_type: str | None = None
    metric: str | None = None
    extra_body: dict | None = None
    process_started_at: float | None = None

    @property
    def trial_limit(self):
        return self.max_trials if self.max_trials is not None else math.inf

    @property
    def step_limit(self):
        return self.max_steps if self.max_steps is not None else math.inf

    @property
    def reserve_seconds(self) -> float:
        from .budgeting import initial_reserve_seconds
        return initial_reserve_seconds(self.time_budget)

    def public_dict(self) -> dict:
        data = asdict(self)
        data.pop("api_key")
        data.pop("extra_body")
        data.pop('process_started_at')
        return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}


def parse_config(argv=None) -> Config:
    parser = argparse.ArgumentParser(description="Autonomous tabular ML research agent; writes submission.csv beside task.md")
    parser.add_argument("task_path", nargs="?", default="task.md", type=Path)
    parser.add_argument("--time-budget", type=float, default=10800, help="Total wall time in seconds, at most 10800")
    parser.add_argument("--trial-timeout", type=float, default=1800)
    parser.add_argument("--llm-timeout", type=float, default=180)
    parser.add_argument("--max-trials", type=int, default=None, help="Optional cap including baseline and failed trials; default uses the time budget")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=2000000, help="Total input + output token budget")
    parser.add_argument("--response-tokens", type=int, default=6144)
    parser.add_argument("--n-jobs", type=int, default=4, help="Threads per trial; total parallel allocation must be <=8")
    parser.add_argument("--parallel-trials", type=int, choices=[1, 2], default=2)
    parser.add_argument("--family-trials", type=int, choices=[1, 2, 3, 4, 5], default=4, help="Maximum warm-up families; 1 keeps only the legacy baseline")
    parser.add_argument('--search-trials', type=int, default=0, help='Bounded reproducible recipe search; on provider failure defaults to four trials')
    parser.add_argument("--experiment-mode", choices=["recipe", "code"], default="recipe", help="Validated parameter recipes by default; code enables free-form Python proposals")
    parser.add_argument("--memory-limit-mb", type=int, default=12000)
    parser.add_argument("--context-chars", type=int, default=48000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], default=5, help="OOF folds; 1 enables legacy holdout")
    parser.add_argument("--validation-strategy", choices=["auto", "iid", "temporal"], default="auto")
    parser.add_argument("--time-column")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize-only", action="store_true", help="With --resume: reuse saved trials and train final model without research or LLM calls")
    parser.add_argument("--offline", action="store_true", help="Explicit deterministic baseline smoke, without LLM")
    parser.add_argument("--target")
    parser.add_argument("--task-type", choices=["regression", "classification"])
    parser.add_argument("--metric", choices=["rmse", "mse", "rmsle", "mape", "mae", "r2", "macro_f1", "weighted_f1", "f1", "accuracy", "balanced_accuracy", "roc_auc", "log_loss"])
    args = parser.parse_args(argv)
    endpoint = os.getenv('OPENAI_BASE_URL', '')
    # baseline.py sets OPENAI_* on the platform; a keyless local invocation
    # should work with the installed Ollama model without extra setup.
    default_provider = 'openai' if (os.getenv('OPENAI_API_KEY') or (endpoint and not any(host in endpoint for host in ('localhost:11434', '127.0.0.1:11434')))) else 'ollama'
    provider = os.getenv("AUTOML_PROVIDER", default_provider).lower()
    if provider not in {"openai", "ollama"}:
        parser.error("AUTOML_PROVIDER must be openai or ollama")
    base_url = os.getenv("OPENAI_BASE_URL") or (os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434") if provider == "ollama" else "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL") or (os.getenv("OLLAMA_MODEL", "qwen3.5:9b") if provider == "ollama" else "")
    api_key = os.getenv("OPENAI_API_KEY", "")
    url = urlsplit(base_url)
    if url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password or url.query or url.fragment:
        parser.error("OPENAI_BASE_URL must be an HTTP(S) URL without embedded credentials or query")
    if not args.offline and not args.finalize_only and not model:
        parser.error("Set OPENAI_MODEL, or use --offline for a deterministic smoke run")
    for name in ("time_budget", "trial_timeout", "llm_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.time_budget > 10800:
        parser.error("--time-budget must be at most 10800 seconds")
    if not 0 <= args.search_trials <= 32:
        parser.error('--search-trials must be between 0 and 32')
    for name in ("max_trials", "max_steps", "max_tokens", "response_tokens", "memory_limit_mb", "context_chars"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 1 <= args.n_jobs <= 8:
        parser.error("--n-jobs must be between 1 and 8")
    if args.n_jobs * args.parallel_trials > 8:
        parser.error("--parallel-trials times --n-jobs must be <=8")
    if args.memory_limit_mb * args.parallel_trials > 28000:
        parser.error("Parallel worker RSS caps must sum to <=28000 MB")
    if not 0.05 <= args.validation_fraction <= 0.5:
        parser.error("--validation-fraction must be between 0.05 and 0.5")
    if args.resume and args.run_dir is None:
        parser.error("--resume requires --run-dir")
    if args.finalize_only and not args.resume:
        parser.error("--finalize-only requires --resume")
    try:
        extra_body = json.loads(os.getenv("AUTOML_EXTRA_BODY", "{}"))
        if not isinstance(extra_body, dict):
            raise ValueError
    except (ValueError, TypeError):
        parser.error("AUTOML_EXTRA_BODY must be a JSON object")
    args.task_path = args.task_path.expanduser().resolve()
    if args.run_dir is not None:
        args.run_dir = args.run_dir.expanduser().resolve()
    if args.resume:
        saved_path = args.run_dir / "config.json"
        if saved_path.exists():
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            supplied = list(argv) if argv is not None else __import__('sys').argv[1:]
            for name, legacy in (("cv_folds", 1), ("validation_strategy", "iid"), ("time_column", None)):
                flag = "--" + name.replace("_", "-")
                if not any(v == flag or v.startswith(flag + "=") for v in supplied):
                    setattr(args, name, saved.get(name, legacy))
    return Config(**vars(args), provider=provider, base_url=base_url, model=model, api_key=api_key, extra_body=extra_body)
