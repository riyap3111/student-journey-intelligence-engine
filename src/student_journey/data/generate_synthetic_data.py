"""Generate a synthetic, de-identified, multi-term student enrollment dataset.

IMPORTANT: Every value produced here is fabricated by a stochastic simulation.
No row, field, or distribution is copied from or fit to any real student,
institution, or dataset. This exists so the rest of the pipeline (SQL,
feature engineering, modeling, API, dashboard) has realistic longitudinal
data with a genuine-but-noisy learnable signal, without any real-world
privacy concern.

Simulation design (why the target isn't leaked):
    For each synthetic student we simulate term-by-term forward in time.
    At each term we generate that term's academic outcomes (GPA, credits,
    withdrawals, repeats) from the student's latent trajectory so far, then
    use THOSE same-term values to decide (via a logistic function + random
    noise) whether the student enrolls in the NEXT term. The loop then
    either continues (persisted) or stops (attrition) or stops because the
    student graduated. This mirrors how the label must be constructed in
    the real pipeline: strictly from future enrollment existence, never
    from a field describing the future term itself.

Run:
    python -m student_journey.data.generate_synthetic_data
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from student_journey.config import ENROLLMENTS_CSV, RANDOM_SEED, STUDENTS_CSV, TERMS_CSV

PROGRAMS = ["Business", "Engineering", "Social Sciences", "Biology", "Computer Science", "Undeclared"]
AGE_BANDS = ["18-20", "21-24", "25-34", "35+"]
GENDER_CODES = ["A", "B", "C"]  # deliberately non-identifying synthetic codes
DISTANCE_BANDS = ["local", "commuter", "remote"]

# Sequential term calendar the simulation can place students into.
TERM_SEQUENCE = [
    (f"20{yr}{season}", f"20{yr} {label}")
    for yr in range(18, 25)
    for season, label in [("FA", "Fall"), ("SP", "Spring"), ("SU", "Summer")]
][:20]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_terms_table() -> pd.DataFrame:
    rows = [
        {"term_id": term_id, "term_order": i + 1, "term_name": term_name}
        for i, (term_id, term_name) in enumerate(TERM_SEQUENCE)
    ]
    return pd.DataFrame(rows)


def _sample_students(n_students: int, rng: np.random.Generator, max_entry_order: int) -> pd.DataFrame:
    student_ids = [f"S{i:05d}" for i in range(1, n_students + 1)]
    entry_type = rng.choice(["first_time", "transfer"], size=n_students, p=[0.75, 0.25])
    program = rng.choice(PROGRAMS, size=n_students)
    age_band = rng.choice(AGE_BANDS, size=n_students, p=[0.55, 0.25, 0.12, 0.08])
    gender = rng.choice(GENDER_CODES, size=n_students, p=[0.48, 0.48, 0.04])
    first_gen_flag = rng.binomial(1, 0.35, size=n_students)
    distance_band = rng.choice(DISTANCE_BANDS, size=n_students, p=[0.4, 0.4, 0.2])
    expected_program_terms = np.clip(rng.normal(8, 1.3, size=n_students).round().astype(int), 5, 14)
    entry_term_order = rng.integers(1, max_entry_order + 1, size=n_students)

    return pd.DataFrame(
        {
            "student_id": student_ids,
            "entry_type": entry_type,
            "program": program,
            "entry_term_order": entry_term_order,
            "expected_program_terms": expected_program_terms,
            "age_band": age_band,
            "gender": gender,
            "first_gen_flag": first_gen_flag,
            "distance_from_campus_band": distance_band,
        }
    )


def _simulate_enrollments(
    students: pd.DataFrame, terms: pd.DataFrame, rng: np.random.Generator, max_term_number: int = 16
) -> pd.DataFrame:
    max_term_order = terms["term_order"].max()
    term_id_by_order = dict(zip(terms["term_order"], terms["term_id"]))

    records = []

    for row in students.itertuples(index=False):
        # Latent per-student traits (not stored — they only drive the simulation).
        ability = rng.normal(0, 1)
        resilience = rng.normal(0, 1)  # buffers against dropping out despite a bad term
        financial_aid_static = rng.binomial(1, 0.45)

        cumulative_credits_attempted = 0
        cumulative_credits_completed = 0
        prev_gpa = None

        for term_number in range(1, max_term_number + 1):
            term_order = row.entry_term_order + term_number - 1
            if term_order > max_term_order:
                break  # ran off the end of the observation calendar (right-censored)

            full_time = rng.random() < (0.72 if financial_aid_static else 0.58)
            enrollment_intensity = "full_time" if full_time else "part_time"

            credits_attempted = int(np.clip(rng.normal(15 if full_time else 8, 1.5), 3, 19))

            gpa_drift = 0.0 if prev_gpa is None else 0.5 * (prev_gpa - 2.7)
            term_gpa = float(np.clip(rng.normal(2.7 + 0.5 * ability + gpa_drift, 0.55), 0.0, 4.0))

            completion_ratio = float(np.clip(0.55 + 0.15 * term_gpa + rng.normal(0, 0.08), 0.15, 1.0))
            credits_completed = round(credits_attempted * completion_ratio)
            credits_completed = min(credits_completed, credits_attempted)

            withdrawal_rate = max(0.0, 0.35 - 0.09 * term_gpa)
            courses_withdrawn = rng.poisson(withdrawal_rate)
            repeat_rate = max(0.0, 0.25 - 0.07 * term_gpa)
            courses_repeated = rng.poisson(repeat_rate)

            advising_contact_flag = int(rng.random() < (0.5 + 0.1 * resilience if resilience > 0 else 0.4))
            advising_contact_flag = int(np.clip(advising_contact_flag, 0, 1))
            financial_aid_flag = financial_aid_static

            cumulative_credits_attempted += credits_attempted
            cumulative_credits_completed += credits_completed

            on_track_credits = cumulative_credits_attempted > 0 and (
                cumulative_credits_completed / cumulative_credits_attempted > 0.7
            )
            is_final_expected_term = term_number >= row.expected_program_terms

            # Graduation check: only plausible once a student has reached (or passed)
            # their expected program length AND is broadly on track academically.
            if is_final_expected_term and on_track_credits:
                grad_prob = _sigmoid(1.2 * (term_gpa - 2.0) + (0.5 if on_track_credits else -0.5))
                if rng.random() < grad_prob:
                    records.append(
                        _row(
                            row.student_id, term_id_by_order[term_order], term_number,
                            enrollment_intensity, credits_attempted, credits_completed, term_gpa,
                            courses_withdrawn, courses_repeated, advising_contact_flag,
                            financial_aid_flag, is_graduating_term=1, censored_flag=0,
                            persisted_next_term=None,
                        )
                    )
                    break

            # If the next term would fall outside the generated calendar, we cannot
            # observe whether the student would have continued — right-censor this
            # row instead of assigning a label that no future row would back up.
            if term_order + 1 > max_term_order:
                records.append(
                    _row(
                        row.student_id, term_id_by_order[term_order], term_number,
                        enrollment_intensity, credits_attempted, credits_completed, term_gpa,
                        courses_withdrawn, courses_repeated, advising_contact_flag,
                        financial_aid_flag, is_graduating_term=0, censored_flag=1,
                        persisted_next_term=None,
                    )
                )
                break

            # Persistence decision: logistic function of this term's own signals.
            risk_score = (
                -1.4
                + 1.1 * (2.5 - term_gpa)
                + 0.9 * (1 - completion_ratio)
                + 0.35 * courses_withdrawn
                + 0.3 * courses_repeated
                - 0.5 * advising_contact_flag
                - 0.35 * financial_aid_flag
                + 0.25 * row.first_gen_flag
                - 0.4 * resilience
                + rng.normal(0, 0.4)
            )
            p_attrition = _sigmoid(risk_score)
            persisted = int(rng.random() > p_attrition)

            records.append(
                _row(
                    row.student_id, term_id_by_order[term_order], term_number,
                    enrollment_intensity, credits_attempted, credits_completed, term_gpa,
                    courses_withdrawn, courses_repeated, advising_contact_flag,
                    financial_aid_flag, is_graduating_term=0, censored_flag=0,
                    persisted_next_term=persisted,
                )
            )

            prev_gpa = term_gpa
            if not persisted:
                break

    return pd.DataFrame(records)


def _row(
    student_id, term_id, term_number, enrollment_intensity, credits_attempted,
    credits_completed, term_gpa, courses_withdrawn, courses_repeated,
    advising_contact_flag, financial_aid_flag, is_graduating_term, censored_flag,
    persisted_next_term,
) -> dict:
    return {
        "student_id": student_id,
        "term_id": term_id,
        "term_number": term_number,
        "enrollment_intensity": enrollment_intensity,
        "credits_attempted": credits_attempted,
        "credits_completed": credits_completed,
        "term_gpa": round(term_gpa, 3),
        "courses_withdrawn": courses_withdrawn,
        "courses_repeated": courses_repeated,
        "advising_contact_flag": advising_contact_flag,
        "financial_aid_flag": financial_aid_flag,
        "is_graduating_term": is_graduating_term,
        "censored_flag": censored_flag,
        "persisted_next_term": persisted_next_term,
    }


def generate(n_students: int = 4000, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    terms = build_terms_table()
    max_entry_order = max(1, terms["term_order"].max() - 6)  # leave runway for a full program
    students = _sample_students(n_students, rng, max_entry_order)
    enrollments = _simulate_enrollments(students, terms, rng)

    students_out = students.drop(columns=["entry_term_order"]).copy()
    students_out.insert(
        3, "entry_term_id", students["entry_term_order"].map(dict(zip(terms["term_order"], terms["term_id"])))
    )
    return students_out, terms, enrollments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-students", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    students, terms, enrollments = generate(args.n_students, args.seed)

    STUDENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    students.to_csv(STUDENTS_CSV, index=False)
    terms.to_csv(TERMS_CSV, index=False)
    enrollments.to_csv(ENROLLMENTS_CSV, index=False)

    print(f"Generated {len(students)} students, {len(terms)} terms, {len(enrollments)} enrollment rows.")
    print(f"  -> {STUDENTS_CSV}")
    print(f"  -> {TERMS_CSV}")
    print(f"  -> {ENROLLMENTS_CSV}")
    labeled = enrollments["persisted_next_term"].notna().sum()
    positive_rate = enrollments["persisted_next_term"].mean()
    print(f"Labeled rows (excl. graduation/censoring): {labeled} | positive rate: {positive_rate:.3f}")


if __name__ == "__main__":
    main()
