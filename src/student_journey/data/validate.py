"""Run the SQL data-quality checks in db/queries/validation.sql against the
ingested SQLite database. Each check is a SELECT that must return zero rows.

Exits non-zero (and raises) if any check fails, so this can gate the pipeline
(e.g. in CI or before feature engineering) rather than silently passing bad data.

Run:
    python -m student_journey.data.validate
"""
from __future__ import annotations

import re
import sqlite3
import sys

from student_journey.config import DB_PATH, QUERIES_DIR

VALIDATION_SQL_PATH = QUERIES_DIR / "validation.sql"
CHECK_HEADER_RE = re.compile(r"--\s*CHECK:\s*(?P<name>\S+)\s*\|\s*(?P<description>.+)")


def parse_checks(sql_text: str) -> list[tuple[str, str, str]]:
    """Split validation.sql into (name, description, query) tuples.

    Only lines starting with "-- CHECK:" (start of line, not just anywhere in
    the text) begin a new check, so explanatory prose mentioning the marker
    can't be mistaken for a check header.
    """
    checks = []
    blocks = re.split(r"^-- CHECK:", sql_text, flags=re.MULTILINE)
    for block in blocks[1:]:
        header_line, _, rest = block.partition("\n")
        match = CHECK_HEADER_RE.match("-- CHECK:" + header_line)
        if not match:
            continue
        name = match.group("name")
        description = match.group("description")
        query = rest.strip().rstrip(";")
        checks.append((name, description, query))
    return checks


def run_validation(db_path=DB_PATH) -> list[dict]:
    sql_text = VALIDATION_SQL_PATH.read_text()
    checks = parse_checks(sql_text)
    if not checks:
        raise RuntimeError(f"No checks parsed from {VALIDATION_SQL_PATH} — check the file format.")

    conn = sqlite3.connect(db_path)
    results = []
    try:
        for name, description, query in checks:
            cursor = conn.execute(query)
            rows = cursor.fetchall()
            results.append({
                "name": name,
                "description": description,
                "failing_rows": len(rows),
                "passed": len(rows) == 0,
                "sample": rows[:5],
            })
    finally:
        conn.close()
    return results


def main() -> None:
    results = run_validation()

    failures = [r for r in results if not r["passed"]]
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['name']}: {r['description']} ({r['failing_rows']} failing rows)")
        if not r["passed"]:
            print(f"         sample failing rows: {r['sample']}")

    print(f"\n{len(results) - len(failures)}/{len(results)} checks passed.")

    if failures:
        print(f"\nVALIDATION FAILED: {[f['name'] for f in failures]}", file=sys.stderr)
        sys.exit(1)

    print("All data validation checks passed.")


if __name__ == "__main__":
    main()
