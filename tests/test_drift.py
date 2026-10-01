"""Tests for PSI-based feature drift detection (monitoring/drift.py). Pure
arithmetic over synthetic pandas DataFrames — no trained model or database
dependency, so these run in any environment."""
import numpy as np
import pandas as pd
import pytest

from student_journey.monitoring import drift

NUMERIC_FEATURES = ["gpa"]
CATEGORICAL_FEATURES = ["program"]


def _reference_df(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "gpa": rng.normal(loc=2.8, scale=0.5, size=n),
        "program": rng.choice(["Business", "Engineering", "Nursing"], size=n, p=[0.5, 0.3, 0.2]),
    })


def test_identical_distribution_has_near_zero_psi_and_is_stable():
    ref_df = _reference_df(seed=1)
    reference = drift.build_reference_distribution(ref_df, NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    current_df = _reference_df(seed=2)  # same generating distribution, different draw
    report = drift.compute_drift_report(current_df, reference)

    assert report["overall_status"] == "stable"
    for feature in report["features"]:
        assert feature["psi"] < drift.PSI_WARNING_THRESHOLD
        assert feature["status"] == "stable"


def test_shifted_numeric_distribution_is_flagged_as_significant_shift():
    ref_df = _reference_df(seed=1)
    reference = drift.build_reference_distribution(ref_df, NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    rng = np.random.default_rng(3)
    shifted_df = pd.DataFrame({
        "gpa": rng.normal(loc=1.0, scale=0.3, size=1000),  # far from the 2.8-centered reference
        "program": rng.choice(["Business", "Engineering", "Nursing"], size=1000, p=[0.5, 0.3, 0.2]),
    })
    report = drift.compute_drift_report(shifted_df, reference)

    gpa_result = next(f for f in report["features"] if f["feature"] == "gpa")
    assert gpa_result["status"] == "significant_shift"
    assert gpa_result["psi"] >= drift.PSI_ALERT_THRESHOLD
    assert report["overall_status"] == "significant_shift"


def test_shifted_categorical_distribution_is_flagged():
    ref_df = _reference_df(seed=1)
    reference = drift.build_reference_distribution(ref_df, NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    rng = np.random.default_rng(4)
    shifted_df = pd.DataFrame({
        "gpa": rng.normal(loc=2.8, scale=0.5, size=1000),
        "program": rng.choice(["Business", "Engineering", "Nursing"], size=1000, p=[0.05, 0.05, 0.9]),
    })
    report = drift.compute_drift_report(shifted_df, reference)

    program_result = next(f for f in report["features"] if f["feature"] == "program")
    assert program_result["status"] in ("moderate_shift", "significant_shift")


def test_insufficient_samples_reports_null_psi():
    ref_df = _reference_df(seed=1)
    reference = drift.build_reference_distribution(ref_df, NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    tiny_current = _reference_df(n=5, seed=5)
    report = drift.compute_drift_report(tiny_current, reference, min_samples=30)

    assert report["overall_status"] == "insufficient_data"
    for feature in report["features"]:
        assert feature["psi"] is None
        assert feature["status"] == "insufficient_data"


def test_missing_feature_in_current_data_is_insufficient_data_not_a_crash():
    ref_df = _reference_df(seed=1)
    reference = drift.build_reference_distribution(ref_df, NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    current_missing_column = pd.DataFrame({"program": ["Business"] * 50})
    report = drift.compute_drift_report(current_missing_column, reference, min_samples=30)

    gpa_result = next(f for f in report["features"] if f["feature"] == "gpa")
    assert gpa_result["status"] == "insufficient_data"


def test_overall_status_is_the_worst_of_any_individual_feature():
    ref_df = _reference_df(seed=1)
    reference = drift.build_reference_distribution(ref_df, ["gpa"], ["program"])

    rng = np.random.default_rng(6)
    # gpa unchanged (stable), program drastically shifted (significant).
    mixed_df = pd.DataFrame({
        "gpa": rng.normal(loc=2.8, scale=0.5, size=1000),
        "program": ["Nursing"] * 1000,
    })
    report = drift.compute_drift_report(mixed_df, reference)
    assert report["overall_status"] == "significant_shift"


def test_load_reference_distribution_returns_none_when_missing(tmp_path):
    assert drift.load_reference_distribution(tmp_path) is None


def test_build_reference_distribution_handles_near_constant_feature():
    """A feature with (almost) no variance has degenerate quantiles — this
    should fall back to a single bin instead of crashing on duplicate edges."""
    df = pd.DataFrame({"constant_feature": [1.0] * 200, "program": ["Business"] * 200})
    reference = drift.build_reference_distribution(df, ["constant_feature"], ["program"])
    assert len(reference["numeric"]["constant_feature"]["proportions"]) >= 1

    current = pd.DataFrame({"constant_feature": [1.0] * 50, "program": ["Business"] * 50})
    report = drift.compute_drift_report(current, reference, min_samples=30)
    constant_result = next(f for f in report["features"] if f["feature"] == "constant_feature")
    assert constant_result["status"] == "stable"


@pytest.mark.parametrize("psi,expected", [
    (None, "insufficient_data"),
    (0.0, "stable"),
    (0.05, "stable"),
    (0.1, "moderate_shift"),
    (0.2, "moderate_shift"),
    (0.25, "significant_shift"),
    (1.0, "significant_shift"),
])
def test_classify_psi_thresholds(psi, expected):
    assert drift.classify_psi(psi) == expected
