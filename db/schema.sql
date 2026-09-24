-- Student Journey Intelligence Engine — raw schema
-- Grain and field meanings are documented in data/DATA_DICTIONARY.md.
-- Written in portable SQL (no SQLite-only extensions) so moving to
-- PostgreSQL later is a connection-string change, not a rewrite.

DROP TABLE IF EXISTS enrollments;
DROP TABLE IF EXISTS students;
DROP TABLE IF EXISTS terms;

CREATE TABLE terms (
    term_id     TEXT PRIMARY KEY,
    term_order  INTEGER NOT NULL UNIQUE,
    term_name   TEXT NOT NULL
);

CREATE TABLE students (
    student_id                 TEXT PRIMARY KEY,
    entry_type                 TEXT NOT NULL CHECK (entry_type IN ('first_time', 'transfer')),
    program                    TEXT NOT NULL,
    entry_term_id               TEXT NOT NULL REFERENCES terms(term_id),
    expected_program_terms     INTEGER NOT NULL CHECK (expected_program_terms > 0),
    age_band                   TEXT NOT NULL,
    gender                     TEXT NOT NULL,
    first_gen_flag             INTEGER NOT NULL CHECK (first_gen_flag IN (0, 1)),
    distance_from_campus_band  TEXT NOT NULL
);

CREATE TABLE enrollments (
    student_id              TEXT NOT NULL REFERENCES students(student_id),
    term_id                  TEXT NOT NULL REFERENCES terms(term_id),
    term_number              INTEGER NOT NULL CHECK (term_number > 0),
    enrollment_intensity     TEXT NOT NULL CHECK (enrollment_intensity IN ('full_time', 'part_time')),
    credits_attempted        INTEGER NOT NULL CHECK (credits_attempted >= 0),
    credits_completed        INTEGER NOT NULL CHECK (credits_completed >= 0),
    term_gpa                 REAL NOT NULL CHECK (term_gpa BETWEEN 0.0 AND 4.0),
    courses_withdrawn        INTEGER NOT NULL CHECK (courses_withdrawn >= 0),
    courses_repeated         INTEGER NOT NULL CHECK (courses_repeated >= 0),
    advising_contact_flag    INTEGER NOT NULL CHECK (advising_contact_flag IN (0, 1)),
    financial_aid_flag       INTEGER NOT NULL CHECK (financial_aid_flag IN (0, 1)),
    is_graduating_term       INTEGER NOT NULL CHECK (is_graduating_term IN (0, 1)),
    censored_flag            INTEGER NOT NULL CHECK (censored_flag IN (0, 1)),
    persisted_next_term      INTEGER CHECK (persisted_next_term IN (0, 1)),  -- NULL when graduating or censored
    PRIMARY KEY (student_id, term_id)
);

CREATE INDEX idx_enrollments_student ON enrollments(student_id, term_number);
CREATE INDEX idx_enrollments_term ON enrollments(term_id);
