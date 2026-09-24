"""Feature engineering: SQL window functions (db/queries/transform.sql) for
cumulative/lag features, finished in pandas for the rolling-trend feature
that needs more than a single window function, then written to a `features`
table (and a parquet copy) for the training phase to consume.

Leakage guardrail: every feature here is computed using only data available
through term N (the row's own term_number or earlier). Nothing from term N+1
or later is ever read. This is checked explicitly in tests/test_features.py.

Run:
    python -m student_journey.features.build_features
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from student_journey.config import DB_PATH, FEATURES_PARQUET, FEATURES_TABLE, QUERIES_DIR

TRANSFORM_SQL_PATH = QUERIES_DIR / "transform.sql"

MOMENTUM_WINDOW = 3  # number of trailing terms used for the academic_momentum trend


def _load_base(db_path=DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(TRANSFORM_SQL_PATH.read_text())
        df = pd.read_sql("SELECT * FROM enrollment_base ORDER BY student_id, term_number", conn)
    finally:
        conn.close()
    return df


def _academic_momentum(df: pd.DataFrame) -> pd.Series:
    """Trend of the last MOMENTUM_WINDOW terms' GPA vs. this term's GPA.

    Positive = improving, negative = declining. Computed only from the
    current row and earlier rows for the same student (shift(1) before the
    rolling window excludes the current term's own GPA from its own trend
    baseline).
    """
    prior_gpa = df.groupby("student_id")["term_gpa"].shift(1)
    prior_avg = prior_gpa.groupby(df["student_id"]).transform(
        lambda s: s.rolling(window=MOMENTUM_WINDOW, min_periods=1).mean()
    )
    momentum = df["term_gpa"] - prior_avg
    return momentum.fillna(0.0)


def build_features(db_path=DB_PATH) -> pd.DataFrame:
    df = _load_base(db_path)
    df["academic_momentum"] = _academic_momentum(df)

    # First-term rows have no prior GPA to diff against; 0 = "no observed change yet"
    # rather than NaN, since a NaN would silently break most sklearn estimators.
    df["gpa_change"] = df["gpa_change"].fillna(0.0)
    df["credit_completion_rate"] = df["credit_completion_rate"].fillna(0.0)
    df["cumulative_credit_completion_rate"] = df["cumulative_credit_completion_rate"].fillna(0.0)

    column_order = [
        "student_id", "term_id", "term_order", "term_number",
        "program", "entry_type", "age_band", "gender", "first_gen_flag", "distance_from_campus_band",
        "enrollment_intensity", "enrollment_intensity_numeric",
        "credits_attempted", "credits_completed", "credit_completion_rate",
        "cumulative_credits_attempted", "cumulative_credits_completed", "cumulative_credit_completion_rate",
        "term_gpa", "cumulative_gpa", "gpa_change", "academic_momentum",
        "courses_withdrawn", "cumulative_withdrawals",
        "courses_repeated", "cumulative_repeats",
        "advising_contact_flag", "financial_aid_flag", "prior_term_enrolled_flag",
        "is_graduating_term", "censored_flag", "persisted_next_term",
    ]
    return df[column_order]


def main() -> None:
    features = build_features()

    conn = sqlite3.connect(DB_PATH)
    try:
        features.to_sql(FEATURES_TABLE, conn, if_exists="replace", index=False)
    finally:
        conn.close()

    FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PARQUET, index=False)

    labeled = features["persisted_next_term"].notna().sum()
    print(f"Built {len(features)} feature rows ({labeled} labeled) with {features.shape[1]} columns.")
    print(f"  -> SQLite table '{FEATURES_TABLE}' in {DB_PATH}")
    print(f"  -> {FEATURES_PARQUET}")


if __name__ == "__main__":
    main()
