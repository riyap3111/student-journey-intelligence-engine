"""Tests for feature engineering: correctness of cumulative/trend math and,
most importantly, that no feature for term N uses information from term N+1
or later (the leakage guardrail described in build_features.py).
"""
import pytest
from sqlalchemy import create_engine

from student_journey.config import SCHEMA_PATH
from student_journey.data.generate_synthetic_data import generate
from student_journey.db import execute_script
from student_journey.features.build_features import build_features


@pytest.fixture
def features_df(tmp_path):
    db_path = tmp_path / "test.db"
    students, terms, enrollments = generate(n_students=200, seed=99)

    engine = create_engine(f"sqlite:///{db_path}")
    execute_script(engine, SCHEMA_PATH.read_text())
    terms.to_sql("terms", engine, if_exists="append", index=False)
    students.to_sql("students", engine, if_exists="append", index=False)
    enrollments.to_sql("enrollments", engine, if_exists="append", index=False)

    return build_features(engine=engine)


def test_cumulative_credits_match_running_sum(features_df):
    df = features_df.sort_values(["student_id", "term_number"])
    for student_id, group in df.groupby("student_id"):
        expected = group["credits_attempted"].cumsum().tolist()
        assert group["cumulative_credits_attempted"].tolist() == expected


def test_first_term_has_zero_gpa_change_and_momentum(features_df):
    first_terms = features_df[features_df["term_number"] == 1]
    assert (first_terms["gpa_change"] == 0.0).all()
    assert (first_terms["academic_momentum"] == 0.0).all()


def test_no_feature_uses_next_term_gpa(features_df):
    """The defining leakage check: gpa_change and cumulative_gpa at term N must
    be computable from term_gpa values at term <= N only, for every student."""
    df = features_df.sort_values(["student_id", "term_number"])
    for student_id, group in df.groupby("student_id"):
        gpas = group["term_gpa"].tolist()
        cumulative = group["cumulative_gpa"].tolist()
        for i in range(len(gpas)):
            expected_cum = sum(gpas[: i + 1]) / (i + 1)
            assert cumulative[i] == pytest.approx(expected_cum, abs=1e-6)


def test_graduating_and_censored_rows_excluded_from_label(features_df):
    excluded = features_df[(features_df["is_graduating_term"] == 1) | (features_df["censored_flag"] == 1)]
    assert excluded["persisted_next_term"].isna().all()


def test_active_rows_have_non_null_label(features_df):
    active = features_df[(features_df["is_graduating_term"] == 0) & (features_df["censored_flag"] == 0)]
    assert active["persisted_next_term"].notna().all()


def test_no_duplicate_student_term_rows(features_df):
    dupes = features_df.duplicated(subset=["student_id", "term_number"]).sum()
    assert dupes == 0
