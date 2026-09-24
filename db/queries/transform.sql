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

    -- Whether the student has any row for the immediately preceding
    -- term_number. In the current generator every student enrolls in
    -- consecutive term_numbers with no simulated gap-and-return, so this is
    -- 0 only for a student's first term; a gap-term simulation is a planned
    -- future enhancement that would make this feature more informative.
    CASE WHEN LAG(e.term_number) OVER w_ord IS NOT NULL THEN 1 ELSE 0 END AS prior_term_enrolled_flag,

    e.is_graduating_term,
    e.censored_flag,
    e.persisted_next_term

FROM enrollments e
JOIN students s ON e.student_id = s.student_id
JOIN terms t ON e.term_id = t.term_id
WINDOW
    w_ord AS (PARTITION BY e.student_id ORDER BY e.term_number),
    w_cum AS (PARTITION BY e.student_id ORDER BY e.term_number
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW);
