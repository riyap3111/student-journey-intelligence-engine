"""Load the generated synthetic CSVs into the database using db/schema.sql.

SQLite by default; set DATABASE_URL to load into PostgreSQL/Cloud SQL
instead — see db.py.

Run:
    python -m student_journey.data.ingest
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from student_journey.config import ENROLLMENTS_CSV, SCHEMA_PATH, STUDENTS_CSV, TERMS_CSV
from student_journey.db import execute_script, get_engine


def ingest(engine=None) -> None:
    if not STUDENTS_CSV.exists():
        raise FileNotFoundError(
            f"{STUDENTS_CSV} not found. Run `python -m student_journey.data.generate_synthetic_data` first."
        )

    engine = engine or get_engine()
    execute_script(engine, SCHEMA_PATH.read_text())

    terms = pd.read_csv(TERMS_CSV)
    students = pd.read_csv(STUDENTS_CSV)
    enrollments = pd.read_csv(ENROLLMENTS_CSV)

    terms.to_sql("terms", engine, if_exists="append", index=False)
    students.to_sql("students", engine, if_exists="append", index=False)
    enrollments.to_sql("enrollments", engine, if_exists="append", index=False)

    with engine.connect() as conn:
        counts = {
            table: conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            for table in ("terms", "students", "enrollments")
        }

    print(f"Ingested into {engine.url}: {counts}")


if __name__ == "__main__":
    ingest()
