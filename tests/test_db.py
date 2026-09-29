"""Tests for the database connection abstraction: defaults to SQLite,
switches to PostgreSQL when DATABASE_URL is set, and the comment-aware SQL
script splitter (the exact thing that broke on a semicolon inside a SQL
comment during development — see db.py's execute_script docstring).
"""
import importlib

from sqlalchemy import create_engine, text

import student_journey.db as db_module


def test_defaults_to_sqlite_when_database_url_unset(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    importlib.reload(db_module)
    try:
        engine = db_module.get_engine()
        assert engine.dialect.name == "sqlite"
        assert db_module.is_postgres(engine) is False
    finally:
        importlib.reload(db_module)


def test_uses_postgres_when_database_url_set(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg2://user:pass@/dbname?host=/cloudsql/proj:region:instance",
    )
    importlib.reload(db_module)
    try:
        engine = db_module.get_engine()
        assert engine.dialect.name == "postgresql"
        assert db_module.is_postgres(engine) is True
    finally:
        monkeypatch.delenv("DATABASE_URL", raising=False)
        importlib.reload(db_module)


def test_execute_script_runs_multiple_statements(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    db_module.execute_script(
        engine,
        """
        CREATE TABLE t1 (id INTEGER);
        CREATE TABLE t2 (id INTEGER);
        INSERT INTO t1 VALUES (1);
        """,
    )
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM t1")).scalar() == 1
        assert conn.execute(text("SELECT COUNT(*) FROM t2")).scalar() == 0


def test_execute_script_ignores_semicolons_inside_comments(tmp_path):
    """Regression test for a real bug found during development: a comment
    reading "...returning from a gap;" broke a naive semicolon-based
    statement splitter, truncating a CREATE VIEW statement mid-way through."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    db_module.execute_script(
        engine,
        """
        -- This comment has a semicolon in it; right here.
        CREATE TABLE t1 (id INTEGER);
        -- Another one; and another;;; for good measure.
        CREATE TABLE t2 (id INTEGER);
        """,
    )
    with engine.connect() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        }
    assert {"t1", "t2"}.issubset(tables)
