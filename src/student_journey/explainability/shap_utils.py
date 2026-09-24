"""Per-prediction and global explainability using SHAP.

Explains predictions in terms of the model's log-odds output (the additive
scale SHAP is exact on for linear models): for a given student-term row,
each feature gets a signed contribution that sums with a baseline to
reproduce exactly the model's raw score, and therefore its predicted
probability after a sigmoid. A positive contribution pushed the prediction
toward "persisted"; a negative contribution pushed it toward "not
persisted" (i.e. increased risk). This is verified in
tests/test_explainability.py (base_value + sum(shap_values) == the
pipeline's raw decision function output, exactly, for the linear model).

The explainer type is chosen by the model actually selected in training
(see train.py):
  - LogisticRegression -> shap.LinearExplainer (exact for linear models)
  - RandomForest / XGBoost -> shap.TreeExplainer (exact for tree ensembles)
This matters: using the wrong explainer for a model family gives
approximate, sometimes misleading attributions, so the model type in
model_metadata.json drives the choice rather than hardcoding one.

One-hot encoded categorical columns (e.g. "program") are reported back
aggregated to their original feature name (summing the SHAP values of that
feature's dummy columns), since "program" is one human-meaningful decision
factor, not six.

IMPORTANT: these explanations describe what the model's own pattern-matching
weighted for THIS row, on a synthetic dataset. They are not a claim about
what causally determines any student's actual future — see the disclaimer
text in EXPLANATION_DISCLAIMER, reused verbatim by the API (Phase 5) and
dashboard (Phase 6).

Run (smoke test on a few real rows):
    python -m student_journey.explainability.shap_utils
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
import shap

from student_journey.config import MODELS_DIR
from student_journey.models.train import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    load_modeling_data,
    time_aware_split,
)

MODEL_PATH = MODELS_DIR / "model_pipeline.joblib"
BACKGROUND_SAMPLE_SIZE = 100
TOP_N_FACTORS = 5

EXPLANATION_DISCLAIMER = (
    "This explanation shows which factors the model weighted most heavily for this "
    "specific prediction. It describes a statistical pattern the model found in "
    "historical-style data — it does not determine, guarantee, or fully explain any "
    "individual student's actual future. Use it as one input to human judgment, not "
    "as a decision by itself."
)


def _aggregate_to_original_features(shap_values: np.ndarray, transformed_names: list[str]) -> dict[str, float]:
    """Sum one-hot dummy-column SHAP values back to their original feature name.

    Transformed names look like 'num__term_gpa' or 'cat__program_Business'.
    Numeric features map 1:1; categorical dummies for the same original
    column (matched against CATEGORICAL_FEATURES) are summed together.
    """
    aggregated: dict[str, float] = {}
    for name, value in zip(transformed_names, shap_values):
        if name.startswith("num__"):
            original = name[len("num__"):]
        elif name.startswith("cat__"):
            stripped = name[len("cat__"):]
            # stripped looks like "<column>_<category>"; match the longest
            # known categorical column name that prefixes it.
            original = next(
                (col for col in CATEGORICAL_FEATURES if stripped.startswith(col + "_")),
                stripped,
            )
        else:
            original = name
        aggregated[original] = aggregated.get(original, 0.0) + float(value)
    return aggregated


class PersistenceExplainer:
    """Loads the saved pipeline once and builds the right SHAP explainer for
    whichever model type was actually selected during training."""

    def __init__(self, model_path=MODEL_PATH, background_size: int = BACKGROUND_SAMPLE_SIZE):
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found. Run `python -m student_journey.models.train` first.")
        self.pipeline = joblib.load(model_path)
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier = self.pipeline.named_steps["classifier"]
        self.transformed_feature_names = list(self.preprocessor.get_feature_names_out())

        df = load_modeling_data()
        train_df, _, _ = time_aware_split(df)
        background_raw = train_df[FEATURE_COLUMNS].sample(
            min(background_size, len(train_df)), random_state=42
        )
        self.background_transformed = self.preprocessor.transform(background_raw)

        model_type = type(self.classifier).__name__
        if model_type == "LogisticRegression":
            self.explainer = shap.LinearExplainer(self.classifier, self.background_transformed)
        elif model_type in ("RandomForestClassifier", "XGBClassifier"):
            self.explainer = shap.TreeExplainer(self.classifier)
        else:
            raise ValueError(f"No SHAP explainer configured for model type: {model_type}")
        self._model_type = model_type

    def explain(self, record: dict) -> dict:
        """Explain a single student-term record (a dict with FEATURE_COLUMNS keys)."""
        raw_df = pd.DataFrame([record])[FEATURE_COLUMNS]
        transformed = self.preprocessor.transform(raw_df)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            explanation = self.explainer(transformed)

        values = explanation.values[0]
        base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
        if self._model_type != "LogisticRegression" and values.ndim > 1:
            # Some TreeExplainer outputs are (n_features, n_classes); take positive class.
            values = values[:, 1]

        by_feature = _aggregate_to_original_features(values, self.transformed_feature_names)
        sorted_factors = sorted(by_feature.items(), key=lambda kv: abs(kv[1]), reverse=True)

        top_factors = [
            {
                "feature": name,
                "contribution": round(value, 4),
                "raw_value": record.get(name),
                "direction": "increases_persistence_likelihood" if value > 0 else "increases_risk",
            }
            for name, value in sorted_factors[:TOP_N_FACTORS]
        ]

        return {
            "base_value_log_odds": round(base_value, 4),
            "total_contribution_log_odds": round(sum(by_feature.values()), 4),
            "top_factors": top_factors,
            "disclaimer": EXPLANATION_DISCLAIMER,
        }

    def global_importance(self, sample_size: int = 300) -> pd.Series:
        """Mean |SHAP value| per original feature over a sample of test rows —
        a global view of what the model relies on most, for the dashboard."""
        df = load_modeling_data()
        _, _, test_df = time_aware_split(df)
        sample = test_df[FEATURE_COLUMNS].sample(min(sample_size, len(test_df)), random_state=42)
        transformed = self.preprocessor.transform(sample)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            explanation = self.explainer(transformed)

        values = explanation.values
        if self._model_type != "LogisticRegression" and values.ndim > 2:
            values = values[:, :, 1]

        abs_by_feature: dict[str, list] = {}
        for row in values:
            agg = _aggregate_to_original_features(row, self.transformed_feature_names)
            for feature, val in agg.items():
                abs_by_feature.setdefault(feature, []).append(abs(val))

        means = {feature: float(np.mean(vals)) for feature, vals in abs_by_feature.items()}
        return pd.Series(means).sort_values(ascending=False)


if __name__ == "__main__":
    import sqlite3

    from student_journey.config import DB_PATH, FEATURES_TABLE

    conn = sqlite3.connect(DB_PATH)
    sample = pd.read_sql(
        f"SELECT * FROM {FEATURES_TABLE} WHERE persisted_next_term IS NOT NULL LIMIT 3", conn
    )
    conn.close()

    explainer = PersistenceExplainer()
    for record in sample.to_dict(orient="records"):
        result = explainer.explain(record)
        print(f"\nstudent {record['student_id']} term {record['term_number']}:")
        print(f"  base_value (log-odds): {result['base_value_log_odds']}")
        for factor in result["top_factors"]:
            print(f"  {factor['feature']:35s} contribution={factor['contribution']:+.4f}  ({factor['direction']})")

    print("\nGlobal feature importance (mean |SHAP|, sample of test set):")
    print(explainer.global_importance())
