"""Streamlit dashboard for the Student Journey Intelligence Engine.

Reuses the exact same model/explainer code the API serves (predict.py,
shap_utils.py) rather than calling the API over HTTP, so the dashboard has
no runtime dependency on the API process being up.

Run:
    streamlit run src/student_journey/dashboard/app.py
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import ks_2samp

from student_journey.analysis.intervention_simulation import simulate_intervention_impact
from student_journey.api.schemas import API_DISCLAIMER
from student_journey.config import DOCS_SCREENSHOTS_DIR, FEATURES_TABLE
from student_journey.data.generate_synthetic_data import PROGRAMS
from student_journey.db import get_engine
from student_journey.explainability.shap_utils import PersistenceExplainer
from student_journey.models.predict import PersistenceModel
from student_journey.models.train import (
    TARGET,
    TRAIN_MAX_TERM_ORDER,
    VAL_MAX_TERM_ORDER,
)

# Validated categorical / status palette — the dataviz skill's reference
# instance, used unmodified (already passes every colorblind-safety and
# contrast check documented there, so it isn't re-validated here).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
RISK_COLOR_MAP = {"low": STATUS["good"], "medium": STATUS["warning"], "high": STATUS["critical"]}
SEQUENTIAL_BLUE = "#2a78d6"

st.set_page_config(page_title="Student Journey Intelligence Engine", layout="wide")

PRIVACY_NOTICE = (
    "**Synthetic data only.** Every student record on this dashboard is fabricated by a "
    "stochastic simulation — no real student, institution, or record is represented. This "
    "is a portfolio/educational project and is not affiliated with, endorsed by, or in use "
    "at any real university. Predictions are for planning and advising support only, never "
    "an automated decision about any student. Full details: `docs/model_card.md`."
)


@st.cache_resource
def get_model():
    try:
        return PersistenceModel()
    except FileNotFoundError:
        return None


@st.cache_resource
def get_explainer():
    try:
        return PersistenceExplainer()
    except FileNotFoundError:
        return None


@st.cache_data
def get_features_df() -> pd.DataFrame:
    return pd.read_sql(f"SELECT * FROM {FEATURES_TABLE}", get_engine())


@st.cache_data
def score_cohort(_model_version: str) -> pd.DataFrame:
    """Score every labeled row once per model version (cache key), for the
    risk-distribution and monitoring views. _model_version is unused beyond
    invalidating the cache when the underlying model changes."""
    model = get_model()
    df = get_features_df()
    labeled = df[df[TARGET].notna()].copy()
    predictions = model.predict_batch(labeled)
    labeled["persistence_probability"] = [p.persistence_probability for p in predictions]
    labeled["risk_probability"] = [p.risk_probability for p in predictions]
    labeled["risk_category"] = [p.risk_category for p in predictions]
    return labeled


def render_overview(features_df, model):
    st.subheader("Dataset")
    if features_df is None:
        st.error("No feature data found. Run the Phase 2 pipeline: `python -m student_journey.features.build_features`.")
        return

    labeled = features_df[features_df[TARGET].notna()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", f"{features_df['student_id'].nunique():,}")
    c2.metric("Student-term rows", f"{len(features_df):,}")
    c3.metric("Labeled rows", f"{len(labeled):,}")
    c4.metric("Persistence rate", f"{labeled[TARGET].mean():.1%}")

    st.caption(
        "Labeled rows exclude graduating-term and end-of-calendar right-censored rows "
        "(see `data/DATA_DICTIONARY.md`). All data is synthetic — see the privacy notice above."
    )

    st.subheader("Model")
    if model is None:
        st.warning("No trained model found. Run `python -m student_journey.models.train`.")
        return

    meta = model.metadata
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model type", meta["selected_model"].replace("_", " ").title())
    c2.metric("Version", meta["model_version"])
    c3.metric("Test ROC-AUC", f"{meta['test_metrics']['roc_auc']:.3f}")
    c4.metric("Test F1", f"{meta['test_metrics']['f1']:.3f}")
    st.caption(
        f"Time-aware split: train ≤ term {TRAIN_MAX_TERM_ORDER}, validation "
        f"{TRAIN_MAX_TERM_ORDER}–{VAL_MAX_TERM_ORDER}, test > {VAL_MAX_TERM_ORDER}. "
        f"Trained: {meta['trained_at_utc']}."
    )


def render_predict(model, explainer):
    if model is None or explainer is None:
        st.warning("No trained model found. Run `python -m student_journey.models.train`.")
        return

    st.subheader("Student-level prediction")
    st.caption("Enter one student-term's engineered features (see `data/DATA_DICTIONARY.md` for field meanings).")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            term_number = st.number_input("Term number", min_value=1, max_value=20, value=3)
            enrollment_intensity = st.selectbox("Enrollment intensity", ["full_time", "part_time"])
            program = st.selectbox("Program", PROGRAMS)
            entry_type = st.selectbox("Entry type", ["first_time", "transfer"])
            term_gpa = st.slider("Term GPA", 0.0, 4.0, 2.7, 0.05)
            cumulative_gpa = st.slider("Cumulative GPA", 0.0, 4.0, 2.8, 0.05)
        with col2:
            credits_attempted = st.number_input("Credits attempted (this term)", 0, 30, 15)
            credits_completed = st.number_input("Credits completed (this term)", 0, 30, 12)
            cumulative_credits_attempted = st.number_input("Cumulative credits attempted", 0, 300, 40)
            cumulative_credits_completed = st.number_input("Cumulative credits completed", 0, 300, 34)
            gpa_change = st.slider("GPA change vs. recent terms", -4.0, 4.0, -0.2, 0.05)
            academic_momentum = st.slider("Academic momentum", -4.0, 4.0, -0.2, 0.05)
        with col3:
            courses_withdrawn = st.number_input("Courses withdrawn (this term)", 0, 10, 0)
            cumulative_withdrawals = st.number_input("Cumulative withdrawals", 0, 30, 1)
            courses_repeated = st.number_input("Courses repeated (this term)", 0, 10, 0)
            cumulative_repeats = st.number_input("Cumulative repeats", 0, 30, 0)
            advising_contact_flag = st.checkbox("Had advising contact this term")
            financial_aid_flag = st.checkbox("Receiving financial aid")
            prior_term_enrolled_flag = st.checkbox("Enrolled in immediately prior term", value=True)

        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    credit_completion_rate = credits_completed / credits_attempted if credits_attempted else 0.0
    cumulative_credit_completion_rate = (
        cumulative_credits_completed / cumulative_credits_attempted if cumulative_credits_attempted else 0.0
    )
    record = {
        "term_number": term_number,
        "enrollment_intensity_numeric": 1 if enrollment_intensity == "full_time" else 0,
        "credits_attempted": credits_attempted,
        "credits_completed": credits_completed,
        "credit_completion_rate": credit_completion_rate,
        "cumulative_credits_attempted": cumulative_credits_attempted,
        "cumulative_credits_completed": cumulative_credits_completed,
        "cumulative_credit_completion_rate": cumulative_credit_completion_rate,
        "term_gpa": term_gpa,
        "cumulative_gpa": cumulative_gpa,
        "gpa_change": gpa_change,
        "academic_momentum": academic_momentum,
        "courses_withdrawn": courses_withdrawn,
        "cumulative_withdrawals": cumulative_withdrawals,
        "courses_repeated": courses_repeated,
        "cumulative_repeats": cumulative_repeats,
        "advising_contact_flag": int(advising_contact_flag),
        "financial_aid_flag": int(financial_aid_flag),
        "prior_term_enrolled_flag": int(prior_term_enrolled_flag),
        "program": program,
        "entry_type": entry_type,
    }

    prediction = model.predict_one(record)
    explanation = explainer.explain(record)

    c1, c2, c3 = st.columns(3)
    c1.metric("Persistence probability", f"{prediction.persistence_probability:.1%}")
    c2.metric("Risk probability", f"{prediction.risk_probability:.1%}")
    c3.markdown(
        f"<div style='padding:0.5rem;border-radius:0.4rem;background:{RISK_COLOR_MAP[prediction.risk_category]};"
        f"color:white;text-align:center;font-weight:600'>Risk: {prediction.risk_category.upper()}</div>",
        unsafe_allow_html=True,
    )

    factors_df = pd.DataFrame(explanation["top_factors"])
    factors_df["color"] = factors_df["direction"].map(
        {"increases_persistence_likelihood": STATUS["good"], "increases_risk": STATUS["critical"]}
    )
    scale_label = "log-odds scale" if explanation["score_scale"] == "log_odds" else "probability scale"
    fig = px.bar(
        factors_df.sort_values("contribution"),
        x="contribution", y="feature", orientation="h",
        color="direction",
        color_discrete_map={"increases_persistence_likelihood": STATUS["good"], "increases_risk": STATUS["critical"]},
        labels={"contribution": f"Contribution ({scale_label})", "feature": "", "direction": ""},
        title="Top contributing factors for this prediction",
    )
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.info(API_DISCLAIMER)


def render_risk_distribution(model):
    if model is None:
        st.warning("No trained model found. Run `python -m student_journey.models.train`.")
        return

    scored = score_cohort(model.model_version)

    c1, c2 = st.columns(2)
    with c1:
        counts = scored["risk_category"].value_counts().reindex(["low", "medium", "high"]).reset_index()
        counts.columns = ["risk_category", "count"]
        fig = px.bar(
            counts, x="risk_category", y="count", color="risk_category",
            color_discrete_map=RISK_COLOR_MAP,
            category_orders={"risk_category": ["low", "medium", "high"]},
            title="Students by risk category (held-out cohort)",
            labels={"risk_category": "Risk category", "count": "Students"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(
            scored, x="risk_probability", nbins=30,
            color_discrete_sequence=[SEQUENTIAL_BLUE],
            title="Risk probability distribution",
            labels={"risk_probability": "Risk probability"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Actual attrition rate by predicted risk category (a sanity check that the ranking is meaningful): "
        + ", ".join(
            f"{cat}: {1 - scored.loc[scored.risk_category == cat, TARGET].astype(int).mean():.1%}"
            for cat in ["low", "medium", "high"]
        )
    )


def render_feature_importance(explainer):
    if explainer is None:
        st.warning("No trained model found. Run `python -m student_journey.models.train`.")
        return

    scale_label = "log-odds scale" if explainer.score_scale == "log_odds" else "probability scale"
    st.caption("Mean |SHAP value| over a sample of the held-out test set — one hue, since this ranks a single series by magnitude.")
    importance = explainer.global_importance(sample_size=300).sort_values()
    fig = px.bar(
        importance, orientation="h",
        color_discrete_sequence=[SEQUENTIAL_BLUE],
        labels={"value": f"Mean |SHAP value| ({scale_label})", "index": ""},
        title="Global feature importance",
    )
    fig.update_layout(showlegend=False, margin={"l": 220}, yaxis={"automargin": True})
    st.plotly_chart(fig, use_container_width=True)


def render_persistence_trends(features_df):
    if features_df is None:
        return
    labeled = features_df[features_df[TARGET].notna()]
    by_term = labeled.groupby("term_number")[TARGET].agg(["mean", "count"]).reset_index()
    by_term.columns = ["term_number", "persistence_rate", "n_students"]
    by_term = by_term[by_term["n_students"] >= 20]  # drop sparse late terms — too few students for a stable rate

    fig = px.line(
        by_term, x="term_number", y="persistence_rate", markers=True,
        color_discrete_sequence=[SEQUENTIAL_BLUE],
        title="Term-over-term persistence rate",
        labels={"term_number": "Student's term number", "persistence_rate": "Persistence rate"},
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Terms with fewer than 20 labeled students are dropped as too sparse for a stable rate.")


def render_bottleneck_analysis(features_df):
    if features_df is None:
        return
    st.caption(
        "This synthetic dataset models student-terms, not individual courses, so there is no "
        "course-level enrollment table to analyze — a planned enhancement. As the closest available "
        "proxy, this section shows where withdrawals and repeats concentrate by term number and by "
        "program."
    )
    labeled = features_df[features_df[TARGET].notna()]

    by_term = labeled.groupby("term_number")[["courses_withdrawn", "courses_repeated"]].mean().reset_index()
    by_term_melted = by_term.melt(id_vars="term_number", var_name="metric", value_name="mean_count")
    fig = px.bar(
        by_term_melted, x="term_number", y="mean_count", color="metric", barmode="group",
        color_discrete_map={"courses_withdrawn": CATEGORICAL[0], "courses_repeated": CATEGORICAL[1]},
        title="Mean withdrawals / repeats by term number",
        labels={"term_number": "Student's term number", "mean_count": "Mean count", "metric": ""},
    )
    st.plotly_chart(fig, use_container_width=True)

    by_program = labeled.groupby("program")[["courses_withdrawn", "courses_repeated"]].mean().reset_index()
    by_program_melted = by_program.melt(id_vars="program", var_name="metric", value_name="mean_count")
    fig = px.bar(
        by_program_melted, x="program", y="mean_count", color="metric", barmode="group",
        color_discrete_map={"courses_withdrawn": CATEGORICAL[0], "courses_repeated": CATEGORICAL[1]},
        title="Mean withdrawals / repeats by program",
        labels={"program": "", "mean_count": "Mean count", "metric": ""},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_model_performance(model):
    if model is None:
        st.warning("No trained model found. Run `python -m student_journey.models.train`.")
        return

    metrics = model.metadata["test_metrics"]
    cols = st.columns(5)
    for col, key in zip(cols, ["precision", "recall", "f1", "roc_auc", "average_precision"]):
        col.metric(key.replace("_", " ").title(), f"{metrics[key]:.3f}")
    st.caption(f"Held-out test set: {metrics['n_samples']:,} rows, {metrics['positive_rate']:.1%} positive rate.")

    c1, c2 = st.columns(2)
    cm_path = DOCS_SCREENSHOTS_DIR / "confusion_matrix.png"
    cal_path = DOCS_SCREENSHOTS_DIR / "calibration_curve.png"
    if cm_path.exists():
        c1.image(str(cm_path), caption="Confusion matrix (from evaluate.py)")
    if cal_path.exists():
        c2.image(str(cal_path), caption="Calibration curve (from evaluate.py)")
    if not (cm_path.exists() and cal_path.exists()):
        st.info("Run `python -m student_journey.models.evaluate` to generate these plots.")


def render_monitoring(features_df):
    st.caption(
        "**Illustrative only.** This is a portfolio project with no live deployment generating new "
        "production traffic to monitor. This section demonstrates the drift-check *method* by comparing "
        "the training period's feature distributions against the held-out test period's — a real "
        "deployment would instead compare training data against actual incoming requests over time."
    )
    if features_df is None:
        return

    labeled = features_df[features_df[TARGET].notna()]
    train = labeled[labeled["term_order"] <= TRAIN_MAX_TERM_ORDER]
    test = labeled[labeled["term_order"] > VAL_MAX_TERM_ORDER]

    for feature in ["term_gpa", "credit_completion_rate"]:
        stat, p_value = ks_2samp(train[feature], test[feature])
        combined = pd.concat([
            train[[feature]].assign(period="train"),
            test[[feature]].assign(period="test"),
        ])
        fig = px.histogram(
            combined, x=feature, color="period", barmode="overlay", opacity=0.6, nbins=30,
            color_discrete_map={"train": CATEGORICAL[0], "test": CATEGORICAL[1]},
            title=f"{feature}: train vs. test distribution (KS statistic={stat:.3f}, p={p_value:.3f})",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_intervention_impact(model):
    st.caption(
        "**Sensitivity analysis over stated assumptions, not a prediction.** The risk-category counts and "
        "observed attrition rates below are real, computed from actual model predictions. The participation "
        "rate and effectiveness are assumptions **you set** with the sliders — this dataset has no recorded "
        "outreach history to estimate a real effect size from. Use this to sketch a business case for "
        "discussion, not as a forecast of what any real program would achieve."
    )
    if model is None:
        st.warning("No trained model found. Run `python -m student_journey.models.train`.")
        return

    scored = score_cohort(model.model_version)
    counts = scored["risk_category"].value_counts().to_dict()
    rates = {
        cat: float(1 - scored.loc[scored.risk_category == cat, TARGET].astype(int).mean())
        for cat in counts
    }

    c1, c2, c3 = st.columns(3)
    with c1:
        target_categories = st.multiselect(
            "Target risk categories", ["low", "medium", "high"], default=["high"]
        )
    with c2:
        participation_rate = st.slider("Participation rate (assumed)", 0.0, 1.0, 0.6, 0.05)
    with c3:
        relative_risk_reduction = st.slider("Relative risk reduction (assumed)", 0.0, 1.0, 0.2, 0.05)

    if not target_categories:
        st.info("Select at least one risk category to simulate.")
        return

    result = simulate_intervention_impact(
        counts, rates, target_categories, participation_rate, relative_risk_reduction
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline expected non-persisters", f"{result['baseline_expected_non_persisters']:.0f}")
    c2.metric("Scenario expected non-persisters", f"{result['scenario_expected_non_persisters']:.0f}")
    c3.metric("Estimated additional students retained", f"{result['additional_students_retained']:.0f}")

    breakdown = pd.DataFrame(result["by_category"]).T
    st.dataframe(breakdown, use_container_width=True)


def main():
    st.title("Student Journey Intelligence Engine")
    st.info(PRIVACY_NOTICE)

    model = get_model()
    explainer = get_explainer()
    try:
        features_df = get_features_df()
    except Exception:
        features_df = None

    tabs = st.tabs([
        "Overview", "Predict", "Risk Distribution", "Feature Importance",
        "Persistence Trends", "Bottleneck Analysis", "Model Performance", "Monitoring",
        "Intervention Impact",
    ])
    with tabs[0]:
        render_overview(features_df, model)
    with tabs[1]:
        render_predict(model, explainer)
    with tabs[2]:
        render_risk_distribution(model)
    with tabs[3]:
        render_feature_importance(explainer)
    with tabs[4]:
        render_persistence_trends(features_df)
    with tabs[5]:
        render_bottleneck_analysis(features_df)
    with tabs[6]:
        render_model_performance(model)
    with tabs[7]:
        render_monitoring(features_df)
    with tabs[8]:
        render_intervention_impact(model)


if __name__ == "__main__":
    main()
