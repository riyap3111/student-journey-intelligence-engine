-- Data validation checks.
-- Each check is a SELECT that should return ZERO rows if the data is clean.
-- Parsed and executed individually by src/student_journey/data/validate.py,
-- which fails the pipeline if any check returns rows.
-- Each check is introduced by a header comment line: name, a pipe, then a
-- human-readable description, immediately above its SELECT.

-- CHECK: duplicate_enrollment_rows | duplicate (student_id, term_id) rows in enrollments
SELECT student_id, term_id, COUNT(*) AS n
FROM enrollments
GROUP BY student_id, term_id
HAVING COUNT(*) > 1;

-- CHECK: duplicate_student_term_number | a student should not have the same term_number twice
SELECT student_id, term_number, COUNT(*) AS n
FROM enrollments
GROUP BY student_id, term_number
HAVING COUNT(*) > 1;

-- CHECK: orphaned_enrollment_student | enrollments referencing a student that doesn't exist
SELECT e.student_id
FROM enrollments e
LEFT JOIN students s ON e.student_id = s.student_id
WHERE s.student_id IS NULL;

-- CHECK: orphaned_enrollment_term | enrollments referencing a term that doesn't exist
SELECT e.term_id
FROM enrollments e
LEFT JOIN terms t ON e.term_id = t.term_id
WHERE t.term_id IS NULL;

-- CHECK: invalid_gpa_range | term_gpa outside the valid 0.0-4.0 range
SELECT student_id, term_id, term_gpa
FROM enrollments
WHERE term_gpa < 0.0 OR term_gpa > 4.0;

-- CHECK: credits_completed_exceeds_attempted | completed more credits than attempted
SELECT student_id, term_id, credits_attempted, credits_completed
FROM enrollments
WHERE credits_completed > credits_attempted;

-- CHECK: negative_values | any negative credits/withdrawals/repeats
SELECT student_id, term_id
FROM enrollments
WHERE credits_attempted < 0 OR credits_completed < 0 OR courses_withdrawn < 0 OR courses_repeated < 0;

-- CHECK: missing_required_fields | NULLs in columns that must always be populated
SELECT student_id, term_id
FROM enrollments
WHERE term_gpa IS NULL
   OR credits_attempted IS NULL
   OR credits_completed IS NULL
   OR enrollment_intensity IS NULL;

-- CHECK: graduating_or_censored_has_label | graduating/censored rows must have a NULL target (excluded from training)
SELECT student_id, term_id, is_graduating_term, censored_flag, persisted_next_term
FROM enrollments
WHERE (is_graduating_term = 1 OR censored_flag = 1)
  AND persisted_next_term IS NOT NULL;

-- CHECK: active_row_missing_label | a non-graduating, non-censored row must have a non-NULL target
SELECT student_id, term_id
FROM enrollments
WHERE is_graduating_term = 0
  AND censored_flag = 0
  AND persisted_next_term IS NULL;

-- CHECK: label_matches_term_order_adjacency | persisted_next_term must exactly equal whether a row exists at term_order+1, in both directions.
-- (Students can stop out for a term or two and return later — see generate_synthetic_data.py —
-- so "persisted_next_term=0" no longer means "no future rows at all," only "no row at the
-- immediate next calendar term." This single bidirectional check is the real invariant.)
WITH e AS (
    SELECT en.student_id, en.term_id, en.persisted_next_term, t.term_order
    FROM enrollments en
    JOIN terms t ON en.term_id = t.term_id
)
SELECT e1.student_id, e1.term_id, e1.term_order, e1.persisted_next_term
FROM e e1
LEFT JOIN e e2 ON e1.student_id = e2.student_id AND e2.term_order = e1.term_order + 1
WHERE e1.persisted_next_term IS NOT NULL
  AND (
        (e1.persisted_next_term = 1 AND e2.student_id IS NULL)
     OR (e1.persisted_next_term = 0 AND e2.student_id IS NOT NULL)
  );
