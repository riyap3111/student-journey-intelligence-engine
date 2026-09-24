"""Tests for SHAP-based explainability: the additivity property that makes
these explanations trustworthy (base_value + sum(shap_values) reproduces the
model's own raw score exactly, for the linear model), and basic sanity of
the per-prediction and global explanation outputs.

Requires a trained model at models/model_pipeline.joblib — run
`python -m student_journey.models.train` first if these are skipped/failing
for that reason.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from student_journey.config import DB_PATH, FEATURES_TABLE
from student_journey.explainability.shap_utils import PersistenceExplainer
from student_journey.models.predict import PersistenceModel
from student_journey.models.train import FEATURE_COLUMNS, MODELS_DIR

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "model_pipeline.joblib").exists(),
    reason="No trained model found; run `python -m student_journey.models.train` first.",
)


@pytest.fixture(scope="module")
def sample_records():
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {FEATURES_TABLE} WHERE persisted_next_term IS NOT NULL LIMIT 5", conn)
    conn.close()
    return df.to_dict(orient="records")


@pytest.fixture(scope="module")
def explainer():
    return PersistenceExplainer()


def test_shap_values_reconstruct_model_probability(sample_records, explainer):
    """The core trust property: for the linear model, base_value + sum of
    the UNAGGREGATED shap values must equal the raw decision function, and
    its sigmoid must equal predict_proba — not an approximation."""
    model = PersistenceModel()
    for record in sample_records:
        raw_df = pd.DataFrame([record])[FEATURE_COLUMNS]
        transformed = explainer.preprocessor.transform(raw_df)
        explanation = explainer.explainer(transformed)

        reconstructed_logit = float(explanation.base_values[0]) + float(explanation.values[0].sum())
        reconstructed_prob = expit(reconstructed_logit)

        actual_prob = model.pipeline.predict_proba(raw_df)[0, 1]
        assert reconstructed_prob == pytest.approx(actual_prob, abs=1e-6)


def test_explain_returns_top_factors_with_disclaimer(sample_records, explainer):
    result = explainer.explain(sample_records[0])
    assert "disclaimer" in result and len(result["disclaimer"]) > 0
    assert 1 <= len(result["top_factors"]) <= 5
    for factor in result["top_factors"]:
        assert factor["direction"] in ("increases_persistence_likelihood", "increases_risk")


def test_global_importance_is_nonnegative_and_sorted(explainer):
    importance = explainer.global_importance(sample_size=50)
    assert (importance >= 0).all()
    assert list(importance) == sorted(importance, reverse=True)
