"""Tests for the intervention-impact sensitivity analysis: pure arithmetic,
no model or data dependency, so these run regardless of whether a model has
been trained."""
import pytest

from student_journey.analysis.intervention_simulation import simulate_intervention_impact

COUNTS = {"low": 1000, "medium": 500, "high": 200}
RATES = {"low": 0.06, "medium": 0.23, "high": 0.485}


def test_zero_effect_leaves_baseline_unchanged():
    result = simulate_intervention_impact(
        COUNTS, RATES, target_categories=["high"], participation_rate=0.8, relative_risk_reduction=0.0
    )
    assert result["additional_students_retained"] == 0.0
    assert result["baseline_expected_non_persisters"] == result["scenario_expected_non_persisters"]


def test_zero_participation_leaves_baseline_unchanged():
    result = simulate_intervention_impact(
        COUNTS, RATES, target_categories=["high"], participation_rate=0.0, relative_risk_reduction=0.5
    )
    assert result["additional_students_retained"] == 0.0


def test_full_participation_and_full_effect_eliminates_targeted_attrition():
    result = simulate_intervention_impact(
        COUNTS, RATES, target_categories=["high"], participation_rate=1.0, relative_risk_reduction=1.0
    )
    assert result["scenario_expected_non_persisters"] == 0.0
    assert result["additional_students_retained"] == pytest.approx(200 * 0.485, abs=1e-6)


def test_known_arithmetic_matches_hand_computation():
    # 200 high-risk students, 48.5% baseline attrition -> 97 expected non-persisters.
    # 50% participate, effect halves their risk (0.5 relative reduction):
    #   participants: 100 * 0.485 * 0.5 = 24.25
    #   non-participants: 100 * 0.485 = 48.5
    #   scenario total = 72.75 -> retained = 97 - 72.75 = 24.25
    result = simulate_intervention_impact(
        COUNTS, RATES, target_categories=["high"], participation_rate=0.5, relative_risk_reduction=0.5
    )
    assert result["baseline_expected_non_persisters"] == pytest.approx(97.0, abs=1e-6)
    assert result["scenario_expected_non_persisters"] == pytest.approx(72.75, abs=1e-6)
    assert result["additional_students_retained"] == pytest.approx(24.25, abs=1e-6)


def test_multiple_target_categories_sum_correctly():
    single_medium = simulate_intervention_impact(
        COUNTS, RATES, target_categories=["medium"], participation_rate=0.6, relative_risk_reduction=0.2
    )
    single_high = simulate_intervention_impact(
        COUNTS, RATES, target_categories=["high"], participation_rate=0.6, relative_risk_reduction=0.2
    )
    combined = simulate_intervention_impact(
        COUNTS, RATES, target_categories=["medium", "high"], participation_rate=0.6, relative_risk_reduction=0.2
    )
    assert combined["additional_students_retained"] == pytest.approx(
        single_medium["additional_students_retained"] + single_high["additional_students_retained"], abs=1e-6
    )
    assert set(combined["by_category"].keys()) == {"medium", "high"}


@pytest.mark.parametrize("bad_rate", [-0.1, 1.1])
def test_rejects_out_of_range_participation_rate(bad_rate):
    with pytest.raises(ValueError):
        simulate_intervention_impact(COUNTS, RATES, ["high"], bad_rate, 0.2)


@pytest.mark.parametrize("bad_effect", [-0.1, 1.1])
def test_rejects_out_of_range_effect_size(bad_effect):
    with pytest.raises(ValueError):
        simulate_intervention_impact(COUNTS, RATES, ["high"], 0.5, bad_effect)


def test_rejects_unknown_target_category():
    with pytest.raises(ValueError):
        simulate_intervention_impact(COUNTS, RATES, ["not_a_real_category"], 0.5, 0.2)
