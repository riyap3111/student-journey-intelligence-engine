"""Simulated intervention impact: "if advisors acted on risk flags with some
assumed effectiveness, what's the estimated impact on persistence?"

WHAT IS REAL HERE, AND WHAT IS ASSUMED — read this before using any number
this module produces:

  - REAL, computed from actual model predictions on the held-out test set:
    how many students fall into each risk category, and what fraction of
    each category actually did not persist (the risk-category counts and
    observed attrition rates).
  - ASSUMED, not derived from data: `participation_rate` (what fraction of
    targeted students actually receive/engage with outreach) and
    `relative_risk_reduction` (how much outreach reduces attrition risk for
    those who engage). This dataset has no recorded outreach history to
    estimate a real effect size from — there is no intervention in the
    synthetic simulation this project's data comes from. These two
    parameters are deliberately exposed as adjustable inputs (sliders in
    the dashboard), not hardcoded defaults presented as findings, so the
    output is legible as a sensitivity analysis over stated assumptions,
    not a causal claim.

This turns the risk model from a prediction demo into a business-case
sketch: "if outreach reduced attrition by X% among Y% of high-risk
students, that's an estimated Z additional students retained" — useful for
a conversation with stakeholders about whether an outreach program would be
worth funding, but the X and Z here are illustrative arithmetic over
assumptions, not a validated prediction of what any real program would
achieve. A real answer to "does this outreach work" requires an actual
experiment (e.g. a randomized pilot), not a simulation.

Run (smoke test with illustrative parameters):
    python -m student_journey.analysis.intervention_simulation
"""
from __future__ import annotations


def simulate_intervention_impact(
    risk_category_counts: dict[str, int],
    observed_attrition_rate: dict[str, float],
    target_categories: list[str],
    participation_rate: float,
    relative_risk_reduction: float,
) -> dict:
    """Expected-value arithmetic, not a model prediction: for each targeted
    risk category, `participation_rate` of students are assumed to receive
    outreach that reduces THEIR attrition probability by
    `relative_risk_reduction`; the rest keep their observed baseline rate.

    Args:
        risk_category_counts: real counts per risk category (e.g. from the
            test set), keys among "low"/"medium"/"high".
        observed_attrition_rate: real observed 1 - persistence rate per
            risk category on the same population.
        target_categories: which risk categories the intervention targets
            (e.g. ["high"], or ["medium", "high"]).
        participation_rate: assumed fraction of targeted students who
            actually engage with outreach, in [0, 1].
        relative_risk_reduction: assumed relative reduction in attrition
            probability for participants, in [0, 1] (0 = no effect, 1 =
            eliminates their attrition risk entirely).

    Returns:
        A dict with baseline/scenario expected non-persister counts, the
        estimated additional students retained, and a per-category
        breakdown — every number computed by simple arithmetic from the
        inputs, nothing hidden.
    """
    if not 0.0 <= participation_rate <= 1.0:
        raise ValueError(f"participation_rate must be in [0, 1], got {participation_rate}")
    if not 0.0 <= relative_risk_reduction <= 1.0:
        raise ValueError(f"relative_risk_reduction must be in [0, 1], got {relative_risk_reduction}")
    unknown = set(target_categories) - set(risk_category_counts)
    if unknown:
        raise ValueError(f"target_categories not in risk_category_counts: {unknown}")

    by_category = {}
    baseline_total = 0.0
    scenario_total = 0.0

    for category in target_categories:
        n = risk_category_counts[category]
        p = observed_attrition_rate[category]
        baseline = n * p

        participants = n * participation_rate
        non_participants = n - participants
        scenario = participants * p * (1 - relative_risk_reduction) + non_participants * p

        baseline_total += baseline
        scenario_total += scenario
        by_category[category] = {
            "n": n,
            "observed_attrition_rate": round(p, 4),
            "baseline_expected_non_persisters": round(baseline, 2),
            "scenario_expected_non_persisters": round(scenario, 2),
        }

    return {
        "target_categories": list(target_categories),
        "participation_rate": participation_rate,
        "relative_risk_reduction": relative_risk_reduction,
        "baseline_expected_non_persisters": round(baseline_total, 2),
        "scenario_expected_non_persisters": round(scenario_total, 2),
        "additional_students_retained": round(baseline_total - scenario_total, 2),
        "by_category": by_category,
    }


if __name__ == "__main__":
    import json
    import sqlite3
    import warnings

    import pandas as pd

    from student_journey.config import DB_PATH, FEATURES_TABLE
    from student_journey.models.predict import PersistenceModel
    from student_journey.models.train import TARGET

    model = PersistenceModel()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {FEATURES_TABLE} WHERE persisted_next_term IS NOT NULL", conn)
    conn.close()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        predictions = model.predict_batch(df)
    df["risk_category"] = [p.risk_category for p in predictions]

    counts = df["risk_category"].value_counts().to_dict()
    rates = {
        cat: float(1 - df.loc[df.risk_category == cat, TARGET].astype(int).mean())
        for cat in counts
    }
    print("Real risk-category counts:", counts)
    print("Real observed attrition rates:", {k: round(v, 4) for k, v in rates.items()})

    for participation, effect in [(0.5, 0.15), (0.75, 0.2), (0.9, 0.3)]:
        result = simulate_intervention_impact(
            counts, rates, target_categories=["high"],
            participation_rate=participation, relative_risk_reduction=effect,
        )
        print(
            f"\n[illustrative] {participation:.0%} participation, {effect:.0%} relative risk reduction "
            f"among 'high' risk: {json.dumps(result, indent=2)}"
        )
