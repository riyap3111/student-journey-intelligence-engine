"""Durable log of scored requests, feeding drift.py's "current distribution"
side. Works against either backend the project supports (SQLite locally,
PostgreSQL/Cloud SQL in the cloud deployment) through the same SQLAlchemy
engine everything else in this project uses (see db.py) — no separate
monitoring datastore to stand up.

Feature values are stored as a single JSON column rather than one column per
feature: FEATURE_COLUMNS can change as the model evolves, and a JSON blob
means this table's schema doesn't need a migration every time it does.
SQLAlchemy's JSON type serializes to TEXT under SQLite and a native JSON
column under PostgreSQL — application code here doesn't need to know which.

Logging is called from the API's request path (main.py) but is deliberately
best-effort there: a monitoring write failing must never fail or slow down
an actual prediction response.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.types import JSON

TABLE_NAME = "prediction_log"

_metadata = MetaData()

prediction_log_table = Table(
    TABLE_NAME,
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("logged_at_utc", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("model_version", String, nullable=False),
    Column("features", JSON, nullable=False),
    Column("persistence_probability", Float, nullable=False),
    Column("risk_category", String, nullable=False),
)


def ensure_table(engine: Engine) -> None:
    _metadata.create_all(engine, tables=[prediction_log_table])


def log_prediction(
    engine: Engine, features: dict[str, Any], model_version: str, persistence_probability: float, risk_category: str
) -> None:
    ensure_table(engine)
    with engine.begin() as conn:
        conn.execute(
            prediction_log_table.insert().values(
                model_version=model_version,
                features=features,
                persistence_probability=persistence_probability,
                risk_category=risk_category,
            )
        )


def load_recent_features(engine: Engine, limit: int = 1000) -> pd.DataFrame:
    """Returns the most recent `limit` logged requests' feature values as a
    flat DataFrame (one column per feature), newest first. Empty DataFrame
    if nothing has been logged yet."""
    ensure_table(engine)
    query = select(prediction_log_table).order_by(prediction_log_table.c.id.desc()).limit(limit)
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(row["features"]) for row in rows])


def count_logged(engine: Engine) -> int:
    ensure_table(engine)
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(prediction_log_table)).scalar())
