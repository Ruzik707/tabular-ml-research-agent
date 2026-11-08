from __future__ import annotations

import math

import pytest

from automl_agent.budgeting import initial_reserve_seconds, plan_finalization
from automl_agent.contracts import ExperimentResult


def result(trial_id: str, status: str, seconds: float) -> ExperimentResult:
    return ExperimentResult(
        trial_id=trial_id,
        status=status,
        artifact_dir="/tmp/trial",
        code_hash=trial_id,
        hypothesis="test trial",
        seconds=seconds,
    )


def test_initial_reserve_is_fifteen_percent_without_old_cap() -> None:
    assert initial_reserve_seconds(10_800) == 1_620


def test_no_successful_measurement_uses_initial_reserve() -> None:
    plan = plan_finalization(total_seconds=1_000, validation_fraction=0.2)

    assert plan["reserve_seconds"] == 150
    assert plan["estimated_fit_seconds"] is None
    assert plan["source"] == "initial"
    assert plan["selected_trial"] is None
    assert plan["row_scale"] is None
    assert plan["safety_factor"] is None
    assert plan["overhead_seconds"] == 0
    assert plan["feasible"] is True


def test_measured_best_uses_exact_ceiling_row_count_and_costs() -> None:
    # ceil(101 * 0.2) = 21 validation rows, leaving 80 fit rows.
    plan = plan_finalization(
        total_seconds=1_000,
        validation_fraction=0.2,
        train_rows=101,
        best=result("trial-7", "ok", 100),
    )

    assert plan["row_scale"] == pytest.approx(101 / 80)
    assert plan["safety_factor"] >= 1.5
    assert plan["estimated_fit_seconds"] == pytest.approx(100 * (101 / 80) * plan["safety_factor"])
    assert plan["overhead_seconds"] > 0
    assert plan["reserve_seconds"] == pytest.approx(
        plan["estimated_fit_seconds"]
        + plan["startup_guard_seconds"]
        + plan["overhead_seconds"]
    )
    assert plan["source"] == "best"
    assert plan["selected_trial"] == "trial-7"
    assert plan["feasible"] is True


def test_row_scale_matches_splitter_when_float_product_is_integral() -> None:
    plan = plan_finalization(
        total_seconds=1_000,
        validation_fraction=0.2,
        train_rows=60,
        best=result("trial-60", "ok", 100),
    )

    # data.py and sklearn use ceil(n * fraction) in float arithmetic: 60 * 0.2 is 12.
    assert plan["row_scale"] == pytest.approx(60 / 48)


def test_row_scale_falls_back_to_fraction_ratio() -> None:
    plan = plan_finalization(
        total_seconds=1_000,
        validation_fraction=0.25,
        best=result("trial-8", "ok", 200),
    )

    assert plan["row_scale"] == pytest.approx(1 / 0.75)
    assert plan["estimated_fit_seconds"] == pytest.approx(200 * (1 / 0.75) * 1.5)


def test_winner_switch_recomputes_reserve_in_both_directions() -> None:
    cheaper = plan_finalization(
        total_seconds=1_000,
        validation_fraction=0.2,
        best=result("slow", "ok", 200),
    )
    switched_to_cheaper = plan_finalization(
        total_seconds=1_000,
        validation_fraction=0.2,
        best=result("fast", "ok", 50),
    )
    switched_to_slower = plan_finalization(
        total_seconds=1_000,
        validation_fraction=0.2,
        best=result("slower", "ok", 400),
    )

    assert switched_to_cheaper["reserve_seconds"] < cheaper["reserve_seconds"]
    assert switched_to_cheaper["selected_trial"] == "fast"
    assert switched_to_slower["reserve_seconds"] > cheaper["reserve_seconds"]
    assert switched_to_slower["selected_trial"] == "slower"


def test_reserve_has_five_percent_floor_and_reports_infeasible_full_fit() -> None:
    floor_plan = plan_finalization(
        total_seconds=1_000,
        validation_fraction=0.2,
        best=result("tiny", "ok", 1),
    )
    over_budget = plan_finalization(
        total_seconds=1_000,
        validation_fraction=0.2,
        best=result("huge", "ok", 800),
    )

    assert floor_plan["reserve_seconds"] == 50
    assert floor_plan["feasible"] is True
    assert over_budget["estimated_fit_seconds"] + over_budget["overhead_seconds"] > 1_000
    assert over_budget["reserve_seconds"] == 1_000
    assert over_budget["feasible"] is False


@pytest.mark.parametrize("status", ["error", "timeout", "interrupted", "cancelled"])
def test_failed_best_is_not_used_as_a_time_measurement(status: str) -> None:
    plan = plan_finalization(
        total_seconds=1_000,
        validation_fraction=0.2,
        best=result("failed", status, 900),
    )

    assert plan["source"] == "initial"
    assert plan["reserve_seconds"] == 150
    assert plan["selected_trial"] is None


@pytest.mark.parametrize("seconds", [0, -1, math.inf, math.nan, 10**400])
def test_nonpositive_or_nonfinite_measurement_is_ignored(seconds: float) -> None:
    plan = plan_finalization(
        total_seconds=1_000,
        validation_fraction=0.2,
        best=result("invalid-time", "ok", seconds),
    )

    assert plan["source"] == "initial"
    assert plan["reserve_seconds"] == 150


@pytest.mark.parametrize("total", [0, -1, math.inf, math.nan, True, 10**400])
def test_total_budget_must_be_finite_and_positive(total: float) -> None:
    with pytest.raises(ValueError):
        initial_reserve_seconds(total)


@pytest.mark.parametrize("fraction", [-0.01, 1, math.inf, math.nan, True])
def test_validation_fraction_must_be_finite_and_in_range(fraction: float) -> None:
    with pytest.raises(ValueError):
        plan_finalization(total_seconds=1_000, validation_fraction=fraction)


@pytest.mark.parametrize("rows", [0, -1, 1.5, math.inf, True])
def test_train_rows_must_be_a_positive_integer(rows: int) -> None:
    with pytest.raises(ValueError):
        plan_finalization(total_seconds=1_000, validation_fraction=0.2, train_rows=rows)


def test_too_few_rows_for_validation_fit_is_rejected() -> None:
    with pytest.raises(ValueError):
        plan_finalization(total_seconds=1_000, validation_fraction=0.99, train_rows=1)


def test_same_durable_best_reconstruction_produces_same_plan() -> None:
    best = result("durable-winner", "ok", 123.5)
    restored = ExperimentResult.from_dict(best.to_dict())

    first = plan_finalization(total_seconds=3_600, validation_fraction=0.2, train_rows=1_001, best=best)
    resumed = plan_finalization(total_seconds=3_600, validation_fraction=0.2, train_rows=1_001, best=restored)

    assert resumed == first


def test_cv_reserve_uses_one_measured_refit_not_all_five_folds():
    from automl_agent.contracts import ExperimentResult
    from automl_agent.budgeting import plan_finalization
    best=ExperimentResult('trial_0001','ok','.','hash','CV',metric=.5,seconds=900,
                          model_details={'training':{'estimated_refit_seconds':100.}})
    plan=plan_finalization(total_seconds=3600,validation_fraction=.2,best=best,train_rows=1000)
    assert plan['estimated_fit_seconds']==150.
    assert plan['reserve_seconds']==194.
