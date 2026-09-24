"""Tests for the synthetic data generator and SQL validation checks.

Uses a temporary SQLite DB (not the project's db/student_journey.db) so tests
never depend on or mutate whatever data happens to be generated locally.
"""
import sqlite3

import pandas as pd
import pytest

from student_journey.config import SCHEMA_PATH
from student_journey.data.generate_synthetic_data import generate
from student_journey.data.validate import run_validation


@pytest.fixture
def small_db(tmp_path):
    db_path = tmp_path / "test.db"
    students, terms, enrollments = generate(n_students=300, seed=123)

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text())
    terms.to_sql("terms", conn, if_exists="append", index=False)
    students.to_sql("students", conn, if_exists="append", index=False)
    enrollments.to_sql("enrollments", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

    return db_path


def test_generate_produces_rows():
    students, terms, enrollments = generate(n_students=50, seed=1)
    assert len(students) == 50
    assert len(terms) > 0
    assert len(enrollments) > 0


def test_generate_is_reproducible_with_same_seed():
    students_a, _, enrollments_a = generate(n_students=50, seed=7)
    students_b, _, enrollments_b = generate(n_students=50, seed=7)
    pd.testing.assert_frame_equal(students_a, students_b)
    pd.testing.assert_frame_equal(enrollments_a, enrollments_b)


def test_all_validation_checks_pass_on_generated_data(small_db):
    results = run_validation(db_path=small_db)
    failures = [r for r in results if not r["passed"]]
    assert not failures, f"Validation checks failed: {[f['name'] for f in failures]}"


def test_persisted_one_always_has_matching_next_row(small_db):
    conn = sqlite3.connect(small_db)
    try:
        rows = conn.execute(
            """
            SELECT e1.student_id
            FROM enrollments e1
            LEFT JOIN enrollments e2
              ON e1.student_id = e2.student_id AND e2.term_number = e1.term_number + 1
            WHERE e1.persisted_next_term = 1 AND e2.student_id IS NULL
            """
        ).fetchall()
    finally:
        conn.close()
    assert rows == []


def test_graduating_and_censored_rows_have_null_target(small_db):
    conn = sqlite3.connect(small_db)
    try:
        rows = conn.execute(
            """
            SELECT COUNT(*) FROM enrollments
            WHERE (is_graduating_term = 1 OR censored_flag = 1) AND persisted_next_term IS NOT NULL
            """
        ).fetchone()
    finally:
        conn.close()
    assert rows[0] == 0


def test_stopout_then_return_students_have_correct_labels():
    """Regression test for a real bug caught during development: students who
    stop out for a term or two and return later must have
    persisted_next_term=0 on the row right before the gap (they did not
    persist to the immediate next calendar term) even though they go on to
    have later rows. An early, incorrect version of the generator's loop
    silently discarded the gap advancement each iteration, so no gaps ever
    actually appeared in the output despite the stop-out logic "running" —
    this test would have caught that by asserting gaps actually occur."""
    _students, terms, enrollments = generate(n_students=1500, seed=42)
    term_order_map = dict(zip(terms.term_id, terms.term_order))
    df = enrollments.copy()
    df["term_order"] = df["term_id"].map(term_order_map)
    df = df.sort_values(["student_id", "term_number"]).reset_index(drop=True)
    df["prev_term_order"] = df.groupby("student_id")["term_order"].shift(1)
    df["gap"] = df["term_order"] - df["prev_term_order"]

    gap_rows = df[df["gap"] > 1]
    assert len(gap_rows) > 0, "expected at least one stop-out-and-return case at this sample size"

    pre_gap_rows = df.loc[gap_rows.index - 1]
    assert (pre_gap_rows["persisted_next_term"] == 0).all()

    # These students have later rows despite the 0 label — exactly the case a
    # naive "persisted=0 implies no later rows" check would wrongly flag.
    for student_id in pre_gap_rows["student_id"]:
        assert len(df[df.student_id == student_id]) > 1


def test_label_matches_term_order_adjacency_bidirectionally(small_db):
    """The real invariant persisted_next_term must satisfy: 1 iff a row
    exists at term_order+1, 0 iff it does not — checked in both directions,
    via terms.term_order rather than term_number (which a stop-out gap
    would make an unreliable proxy for calendar adjacency)."""
    conn = sqlite3.connect(small_db)
    try:
        rows = conn.execute(
            """
            WITH e AS (
                SELECT en.student_id, en.term_id, en.persisted_next_term, t.term_order
                FROM enrollments en JOIN terms t ON en.term_id = t.term_id
            )
            SELECT e1.student_id, e1.term_order, e1.persisted_next_term
            FROM e e1
            LEFT JOIN e e2 ON e1.student_id = e2.student_id AND e2.term_order = e1.term_order + 1
            WHERE e1.persisted_next_term IS NOT NULL
              AND (
                    (e1.persisted_next_term = 1 AND e2.student_id IS NULL)
                 OR (e1.persisted_next_term = 0 AND e2.student_id IS NOT NULL)
              )
            """
        ).fetchall()
    finally:
        conn.close()
    assert rows == []
