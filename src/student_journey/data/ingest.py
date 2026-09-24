"""Load the generated synthetic CSVs into SQLite using db/schema.sql.

Run:
    python -m student_journey.data.ingest
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from student_journey.config import DB_PATH, ENROLLMENTS_CSV, SCHEMA_PATH, STUDENTS_CSV, TERMS_CSV


def ingest(db_path=DB_PATH) -> None:
    if not STUDENTS_CSV.exists():
        raise FileNotFoundError(
            f"{STUDENTS_CSV} not found. Run `python -m student_journey.data.generate_synthetic_data` first."
        )

    schema_sql = SCHEMA_PATH.read_text()

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(schema_sql)

        terms = pd.read_csv(TERMS_CSV)
        students = pd.read_csv(STUDENTS_CSV)
        enrollments = pd.read_csv(ENROLLMENTS_CSV)

        terms.to_sql("terms", conn, if_exists="append", index=False)
        students.to_sql("students", conn, if_exists="append", index=False)
        enrollments.to_sql("enrollments", conn, if_exists="append", index=False)
        conn.commit()

        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("terms", "students", "enrollments")
        }
    finally:
        conn.close()

    print(f"Ingested into {db_path}: {counts}")


if __name__ == "__main__":
    ingest()
