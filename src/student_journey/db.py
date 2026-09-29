"""Database connection abstraction: SQLite by default (this project's
original design, still the default for local dev and Docker Compose), or
PostgreSQL / Google Cloud SQL when DATABASE_URL is set.

Every script that touches the database (ingest.py, validate.py,
build_features.py, train.py, predict.py, evaluate.py, the dashboard) goes
through get_engine() rather than calling sqlite3.connect() directly, so
switching databases is a connection-string change — exactly what
db/schema.sql's own comment already promised when it was written as
portable SQL, now actually true rather than aspirational.

Cloud SQL connection strings look like:
    postgresql+psycopg2://USER:PASSWORD@/DBNAME?host=/cloudsql/PROJECT:REGION:INSTANCE
(the Unix-socket form Cloud Run uses when the Cloud SQL instance is attached
to the service — see docs/deployment/gcp.md for the full setup). A normal
TCP Postgres instance uses the standard
    postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DBNAME
form instead.

This module was exercised against a real local SQLite database extensively
(every phase of this project) but NOT against a real Cloud SQL/PostgreSQL
instance in the environment this project was developed in (no GCP
project/credentials available there) — the schema and queries are written
in portable SQL and reviewed for PostgreSQL compatibility, but you should
run the pipeline against a real Postgres instance you control before
relying on this path in production.
"""
from __future__ import annotations

import os
import re

from sqlalchemy import Engine, create_engine, text

from student_journey.config import DB_PATH

DATABASE_URL = os.environ.get("DATABASE_URL")


def get_engine() -> Engine:
    if DATABASE_URL:
        return create_engine(DATABASE_URL)
    return create_engine(f"sqlite:///{DB_PATH}")


def is_postgres(engine: Engine) -> bool:
    return engine.dialect.name == "postgresql"


def execute_script(engine: Engine, sql_text: str) -> None:
    """Runs a multi-statement SQL script (schema.sql, transform.sql) as one
    transaction. '-- ...' line comments are stripped before splitting on
    ';' — this project's SQL files are heavily commented in plain English,
    and prose can contain semicolons (found the hard way: a comment reading
    "...returning from a gap;" broke a naive split). Not a general-purpose
    SQL script parser — assumes no semicolons inside string literals or
    function bodies, which holds for this project's own schema/transform
    scripts.
    """
    without_comments = re.sub(r"--[^\n]*", "", sql_text)
    statements = [s.strip() for s in without_comments.split(";") if s.strip()]
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
