"""Tests for the prediction_log table (monitoring/prediction_log.py): logging
a scored request and reading recent ones back, against a real (temporary)
SQLite database — no mocking, since SQLAlchemy Core + SQLite is fast and
this is exactly the code path the local-dev API actually runs."""
from sqlalchemy import create_engine

from student_journey.monitoring import prediction_log

FEATURES = {"term_gpa": 2.5, "program": "Business", "credits_attempted": 15}


def _engine(tmp_path):
    return create_engine(f"sqlite:///{tmp_path / 'test.db'}")


def test_log_prediction_creates_table_and_stores_a_row(tmp_path):
    engine = _engine(tmp_path)
    prediction_log.log_prediction(engine, FEATURES, "v1", persistence_probability=0.8, risk_category="low")
    assert prediction_log.count_logged(engine) == 1


def test_load_recent_features_round_trips_feature_values(tmp_path):
    engine = _engine(tmp_path)
    prediction_log.log_prediction(engine, FEATURES, "v1", persistence_probability=0.8, risk_category="low")

    df = prediction_log.load_recent_features(engine)
    assert len(df) == 1
    assert df.iloc[0]["term_gpa"] == 2.5
    assert df.iloc[0]["program"] == "Business"


def test_load_recent_features_returns_empty_dataframe_when_nothing_logged(tmp_path):
    engine = _engine(tmp_path)
    df = prediction_log.load_recent_features(engine)
    assert df.empty


def test_load_recent_features_respects_limit_and_orders_newest_first(tmp_path):
    engine = _engine(tmp_path)
    for i in range(5):
        prediction_log.log_prediction(
            engine, {"term_gpa": float(i)}, "v1", persistence_probability=0.5, risk_category="medium"
        )

    df = prediction_log.load_recent_features(engine, limit=2)
    assert len(df) == 2
    # Newest first: the last two logged (term_gpa=4, then 3).
    assert list(df["term_gpa"]) == [4.0, 3.0]


def test_count_logged_reflects_multiple_inserts(tmp_path):
    engine = _engine(tmp_path)
    for _ in range(3):
        prediction_log.log_prediction(engine, FEATURES, "v1", persistence_probability=0.5, risk_category="medium")
    assert prediction_log.count_logged(engine) == 3
