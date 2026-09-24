-- Transform layer: joins the three raw tables and computes the features that
-- are natural to express as SQL window functions (running totals, lag-based
-- deltas). Rolling-trend features that need more than a single window
-- function (e.g. academic_momentum) are finished in pandas — see
-- src/student_journey/features/build_features.py.
--
-- Re-run safely: this recreates the view each time it's executed.

DROP VIEW IF EXISTS enrollment_base;

CREATE VIEW enrollment_base AS
SELECT
    e.student_id,
    e.term_id,
    t.term_order,
    e.term_number,

    s.program,
    s.entry_type,
    s.age_band,
    s.gender,
    s.first_gen_flag,
    s.distance_from_campus_band,

    e.enrollment_intensity,
    CASE WHEN e.enrollment_intensity = 'full_time' THEN 1 ELSE 0 END AS enrollment_intensity_numeric,

    e.credits_attempted,
    e.credits_completed,
    CAST(e.credits_completed AS REAL) / NULLIF(e.credits_attempted, 0) AS credit_completion_rate,

    SUM(e.credits_attempted) OVER w_cum AS cumulative_credits_attempted,
    SUM(e.credits_completed) OVER w_cum AS cumulative_credits_completed,
    CAST(SUM(e.credits_completed) OVER w_cum AS REAL)
        / NULLIF(SUM(e.credits_attempted) OVER w_cum, 0) AS cumulative_credit_completion_rate,

    e.term_gpa,
    AVG(e.term_gpa) OVER w_cum AS cumulative_gpa,
    e.term_gpa - LAG(e.term_gpa) OVER w_ord AS gpa_change,

    e.courses_withdrawn,
    SUM(e.courses_withdrawn) OVER w_cum AS cumulative_withdrawals,
    e.courses_repeated,
    SUM(e.courses_repeated) OVER w_cum AS cumulative_repeats,

    e.advising_contact_flag,
    e.financial_aid_flag,

    -- Whether the student has a row at the immediately preceding CALENDAR
    -- term (term_order - 1), not just the preceding term_number. Students
    -- can stop out for a term or two and return (see
    -- generate_synthetic_data.py), so this is genuinely informative: 0 for
    -- a student's first term OR a term right after returning from a gap;
    -- 1 for consecutive enrollment. Computed via a self-join on terms
    -- (term_order - 1) rather than LAG, since LAG over term_number would
    -- treat a post-gap return as "consecutive" when it isn't.
    CASE WHEN e_prev.student_id IS NOT NULL THEN 1 ELSE 0 END AS prior_term_enrolled_flag,

    e.is_graduating_term,
    e.censored_flag,
    e.persisted_next_term

FROM enrollments e
JOIN students s ON e.student_id = s.student_id
JOIN terms t ON e.term_id = t.term_id
LEFT JOIN terms t_prev ON t_prev.term_order = t.term_order - 1
LEFT JOIN enrollments e_prev ON e_prev.student_id = e.student_id AND e_prev.term_id = t_prev.term_id
WINDOW
    w_ord AS (PARTITION BY e.student_id ORDER BY e.term_number),
    w_cum AS (PARTITION BY e.student_id ORDER BY e.term_number
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW);
