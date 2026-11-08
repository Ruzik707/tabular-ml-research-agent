from __future__ import annotations

import pytest

from automl_agent.budgeting import initial_reserve_seconds, plan_finalization
from automl_agent.contracts import ExperimentResult


def measured_result(seconds: float) -> ExperimentResult:
    return ExperimentResult(
        trial_id="trial-measured",
        status="ok",
        artifact_dir="/tmp/trial",
        code_hash="hash",
        hypothesis="measured fit",
        seconds=seconds,
        model_details={"training": {"estimated_refit_seconds": seconds}},
    )


@pytest.mark.parametrize(
    ("total", "expected"),
    [(60, 9), (180, 27), (10_800, 1_620)],
)
def test_initial_startup_reserve_keeps_existing_fifteen_percent_contract(total, expected):
    assert initial_reserve_seconds(total) == expected


@pytest.mark.parametrize(("total", "minimum"), [(60, 12), (180, 20)])
def test_budget_window_preserves_short_run_finalization_reserve(total, minimum):
    plan = plan_finalization(
        total_seconds=total,
        validation_fraction=0.2,
        best=measured_result(2.96),
        train_rows=1_000,
    )

    assert plan["reserve_seconds"] >= minimum


def test_no_measurement_finalization_plan_does_not_park_a_large_share_of_three_hours():
    plan = plan_finalization(total_seconds=10_800, validation_fraction=0.2)

    assert plan["reserve_seconds"] <= 150


def test_measured_refit_includes_worker_startup_and_serialization_guard():
    plan = plan_finalization(
        total_seconds=60,
        validation_fraction=0.2,
        best=measured_result(2.96),
        train_rows=1_000,
    )

    assert plan["estimated_fit_seconds"] == pytest.approx(2.96 * 1.5)
    assert plan["startup_guard_seconds"] >= 7
    assert plan["overhead_seconds"] == pytest.approx(5)
    assert plan["required_seconds"] == pytest.approx(
        plan["estimated_fit_seconds"]
        + plan["startup_guard_seconds"]
        + plan["overhead_seconds"]
    )
    assert plan["reserve_seconds"] == pytest.approx(plan["required_seconds"])


def test_cv_refit_estimate_does_not_get_row_scaled_twice():
    plan = plan_finalization(
        total_seconds=180,
        validation_fraction=0.2,
        best=measured_result(2.96),
        train_rows=100_000,
    )

    assert plan["estimated_fit_seconds"] == pytest.approx(2.96 * 1.5)
    assert plan["reserve_seconds"] == pytest.approx(24, abs=1)


def test_full_fit_admission_is_strict_about_the_whole_required_window():
    from automl_agent.budgeting import can_start_final_fit

    plan = plan_finalization(
        total_seconds=180,
        validation_fraction=0.2,
        best=measured_result(20),
    )

    assert plan["required_seconds"] > plan["estimated_fit_seconds"]
    assert plan["required_seconds"] == pytest.approx(
        plan["estimated_fit_seconds"]
        + plan["startup_guard_seconds"]
        + plan["overhead_seconds"]
    )
    assert can_start_final_fit(plan["required_seconds"], plan)
    assert not can_start_final_fit(plan["required_seconds"] - 0.001, plan)
    assert not can_start_final_fit(180, {**plan, "feasible": False})


def test_no_measurement_is_safe_and_does_not_claim_fit_admission():
    from automl_agent.budgeting import can_start_final_fit

    plan = plan_finalization(total_seconds=60, validation_fraction=0.2)

    assert plan["reserve_seconds"] >= 12
    assert plan["source"] == "initial"
    assert plan["estimated_fit_seconds"] is None
    assert not can_start_final_fit(60, plan)


def test_trial_cap_scales_with_research_window_and_stays_bounded_on_long_runs():
    from automl_agent.budgeting import trial_time_cap

    short = trial_time_cap(total_seconds=60, research_seconds=48)
    medium = trial_time_cap(total_seconds=180, research_seconds=156)
    long = trial_time_cap(total_seconds=10_800, research_seconds=10_740)

    assert 0 < short <= 12
    assert 0 < medium <= 30
    assert short < medium
    assert long <= 1_800


def test_requested_trial_cap_can_only_lower_adaptive_cap():
    from automl_agent.budgeting import trial_time_cap

    cap = trial_time_cap(total_seconds=180, research_seconds=156, requested_cap=8)

    assert cap <= 8
    assert cap > 0


def test_stable_trial_cap_uses_total_budget_until_research_window_is_tight():
    from automl_agent.budgeting import trial_time_cap

    at_start = trial_time_cap(total_seconds=180, research_seconds=156, requested_cap=120)
    later = trial_time_cap(total_seconds=180, research_seconds=90, requested_cap=120)
    near_cutoff = trial_time_cap(total_seconds=180, research_seconds=20, requested_cap=120)
    at_cutoff = trial_time_cap(total_seconds=180, research_seconds=1.99, requested_cap=120)
    long_run = trial_time_cap(total_seconds=10_800, research_seconds=10_740, requested_cap=2_000)

    assert at_start == later == 30
    assert near_cutoff == 18
    assert at_cutoff == 0
    assert long_run == 1_800


@pytest.mark.parametrize("remaining", [0, -1, float("nan"), float("inf")])
def test_final_fit_admission_rejects_invalid_remaining_time(remaining):
    from automl_agent.budgeting import can_start_final_fit

    plan = plan_finalization(total_seconds=60, validation_fraction=0.2, best=measured_result(1))

    assert not can_start_final_fit(remaining, plan)
