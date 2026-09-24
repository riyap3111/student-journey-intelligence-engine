"""Shared paths and constants for the Student Journey Intelligence Engine.

Centralized here so scripts (generation, ingestion, features, training, API)
all agree on where things live instead of hardcoding paths independently.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

DB_DIR = PROJECT_ROOT / "db"
DB_PATH = DB_DIR / "student_journey.db"
SCHEMA_PATH = DB_DIR / "schema.sql"
QUERIES_DIR = DB_DIR / "queries"

MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
DOCS_SCREENSHOTS_DIR = PROJECT_ROOT / "docs" / "screenshots"

RANDOM_SEED = 42

# Raw synthetic data filenames
STUDENTS_CSV = RAW_DATA_DIR / "students.csv"
TERMS_CSV = RAW_DATA_DIR / "terms.csv"
ENROLLMENTS_CSV = RAW_DATA_DIR / "enrollments.csv"

# Feature table output
FEATURES_TABLE = "features"
FEATURES_PARQUET = PROCESSED_DATA_DIR / "features.parquet"

for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, DB_DIR, MODELS_DIR, MLRUNS_DIR, DOCS_SCREENSHOTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
