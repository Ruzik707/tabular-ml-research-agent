"""Pure policies for reserving time for final model fitting."""

from __future__ import annotations

import math
from numbers import Integral, Real

from automl_agent.contracts import ExperimentResult

_INITIAL_FRACTION = 0.15
_INITIAL_RESERVE_CAP_SECONDS = 150.0
_SAFETY_FACTOR = 1.5
_STARTUP_GUARD_SECONDS = 8.0
_OVERHEAD_CAP_SECONDS = 60.0


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _total_budget(value: object) -> float:
    seconds = _finite_number(value, "total_seconds")
    if seconds <= 0:
        raise ValueError("total_seconds must be positive")
    return seconds


def _measurement_seconds(best: ExperimentResult | None) -> float | None:
    if best is None or getattr(best, "status", None) != "ok":
        return None
    raw_seconds = getattr(best, "seconds", None)
    try:
        seconds = _finite_number(raw_seconds, "best.seconds")
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def _initial_plan(total_seconds: float) -> dict:
    floor = minimum_finalization_reserve_seconds(total_seconds)
    return {
        "reserve_seconds": min(total_seconds, max(floor, min(initial_reserve_seconds(total_seconds), _INITIAL_RESERVE_CAP_SECONDS))),
        "estimated_fit_seconds": None,
        "source": "initial",
        "selected_trial": None,
        "row_scale": None,
        "safety_factor": None,
        "startup_guard_seconds": 0.0,
        "overhead_seconds": 0.0,
        "required_seconds": None,
        "feasible": True,
    }


def minimum_finalization_reserve_seconds(total_seconds: float) -> float:
    """A short-run finalization floor that stays bounded for long jobs."""
    total = _total_budget(total_seconds)
    short_run_floor = min(30.0, 6.0 + total * 0.10)
    proportional_floor = min(_INITIAL_RESERVE_CAP_SECONDS, total * 0.05)
    return min(total, max(12.0, short_run_floor, proportional_floor))


def initial_reserve_seconds(total_seconds: float) -> float:
    """Return the original 15% startup reserve before measurements exist."""
    return _total_budget(total_seconds) * _INITIAL_FRACTION


def plan_finalization(
    *,
    total_seconds: float,
    validation_fraction: float,
    best: ExperimentResult | None = None,
    train_rows: int | None = None,
) -> dict:
    """Estimate a full-data fit deadline from a successful validation trial.

    This is a policy estimate, not a runtime guarantee. Recreating it from the
    same durable winning result yields the same plan because it reads no clock
    and keeps no history.
    """
    total = _total_budget(total_seconds)
    fraction = _finite_number(validation_fraction, "validation_fraction")
    if fraction < 0 or fraction >= 1:
        raise ValueError("validation_fraction must be in [0, 1)")

    if train_rows is None:
        row_scale = 1.0 / (1.0 - fraction)
    else:
        if isinstance(train_rows, bool) or not isinstance(train_rows, Integral) or train_rows <= 0:
            raise ValueError("train_rows must be a positive integer")
        validation_rows = math.ceil(int(train_rows) * fraction)
        fit_rows = int(train_rows) - validation_rows
        if fit_rows <= 0:
            raise ValueError("train_rows and validation_fraction leave no fit rows")
        row_scale = int(train_rows) / fit_rows

    measured_seconds = _measurement_seconds(best)
    if measured_seconds is None:
        return _initial_plan(total)

    training = getattr(best, 'model_details', {}).get('training', {})
    cv_estimate = training.get('estimated_refit_seconds')
    if isinstance(cv_estimate, (int, float)) and not isinstance(cv_estimate, bool) and math.isfinite(cv_estimate) and cv_estimate > 0:
        measured_seconds, row_scale = float(cv_estimate), 1.0
    estimated_fit = measured_seconds * row_scale * _SAFETY_FACTOR
    # Reserve time for writing artifacts and the final result. Scale with the
    # run while retaining a small absolute minimum for short runs.
    overhead = min(total, _OVERHEAD_CAP_SECONDS, max(5.0, total * 0.01))
    startup_guard = min(total, _STARTUP_GUARD_SECONDS)
    projected_cost = estimated_fit + startup_guard + overhead
    reserve = min(total, max(minimum_finalization_reserve_seconds(total), projected_cost))
    return {
        "reserve_seconds": reserve,
        "estimated_fit_seconds": estimated_fit,
        "source": "best",
        "selected_trial": getattr(best, "trial_id", None),
        "row_scale": row_scale,
        "safety_factor": _SAFETY_FACTOR,
        "startup_guard_seconds": startup_guard,
        "overhead_seconds": overhead,
        "required_seconds": projected_cost,
        "feasible": projected_cost <= total,
    }


def can_start_final_fit(remaining_seconds: object, plan: dict) -> bool:
    """Return whether the entire measured fit and finalization window remains."""
    if not isinstance(plan, dict) or not plan.get("feasible"):
        return False
    estimated = plan.get("estimated_fit_seconds")
    if estimated is None:
        return False
    try:
        remaining = _finite_number(remaining_seconds, "remaining_seconds")
        required = _finite_number(plan.get("required_seconds"), "required_seconds")
        guard = _finite_number(plan.get("startup_guard_seconds"), "startup_guard_seconds")
    except ValueError:
        return False
    return remaining >= max(required, float(estimated) + guard)


def trial_time_cap(
    *, total_seconds: object, research_seconds: object, requested_cap: object | None = None
) -> float:
    """Keep a stable total-budget cap until the remaining research window binds."""
    total = _total_budget(total_seconds)
    research = _finite_number(research_seconds, "research_seconds")
    if research < 0:
        raise ValueError("research_seconds must be nonnegative")
    cap = min(total / 6.0, max(0.0, research - 2.0))
    if requested_cap is not None:
        requested = _finite_number(requested_cap, "requested_cap")
        if requested <= 0:
            raise ValueError("requested_cap must be positive")
        cap = min(cap, requested)
    return max(0.0, cap)
