"""Per-prediction and global explainability using SHAP.

Explains predictions in terms of the model's own additive score: for a
given student-term row, each feature gets a signed contribution that sums
with a baseline to reproduce exactly the model's raw output. A positive
contribution pushed the prediction toward "persisted"; a negative
contribution pushed it toward "not persisted" (i.e. increased risk). This
additivity is verified in tests/test_explainability.py.

The explainer type is chosen by the model actually selected in training
(see train.py):
  - LogisticRegression -> shap.LinearExplainer (exact for linear models)
  - RandomForest / XGBoost -> shap.TreeExplainer (exact for tree ensembles)
This matters: using the wrong explainer for a model family gives
approximate, sometimes misleading attributions, so the model type in
model_metadata.json drives the choice rather than hardcoding one.

The additive scale is NOT the same for every model type: LinearExplainer is
exact on the log-odds scale (needs a sigmoid to recover a probability), but
shap.TreeExplainer's scale depends on the specific tree algorithm —
probability-space for RandomForestClassifier, typically log-odds/margin for
gradient-boosted trees like XGBoost. Rather than hardcode an assumption per
model name, PersistenceExplainer detects the scale empirically at load time
(see _detect_score_scale) and reports it in every explanation's
"score_scale" field, so callers always know how to read "base_value" and
"contribution" without guessing.

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
from student_journey.data.generate_synthetic_data import PROGRAMS
from student_journey.models.train import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    load_modeling_data,
    time_aware_split,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# For the ensemble explainer path only: SHAP's default tabular masker calls
# np.isclose on the background data to detect invariant features, which
# raises a TypeError on string columns (can't subtract strings). Categorical
# columns are encoded to fixed integer codes before reaching SHAP, and
# decoded back to strings inside the wrapped predict_proba callable — a
# fixed mapping (not derived from the background sample's observed values)
# so codes stay consistent between the background and any later query row,
# even if a category happens not to appear in the background sample.
_CATEGORY_MAPS = {"program": PROGRAMS, "entry_type": ["first_time", "transfer"]}


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, categories in _CATEGORY_MAPS.items():
        code_by_category = {category: i for i, category in enumerate(categories)}
        df[col] = df[col].map(code_by_category).astype(float)
    return df[FEATURE_COLUMNS]


def _decode_categoricals(x) -> pd.DataFrame:
    df = pd.DataFrame(np.asarray(x, dtype=float), columns=FEATURE_COLUMNS)
    for col, categories in _CATEGORY_MAPS.items():
        codes = df[col].round().astype(int).clip(0, len(categories) - 1)
        df[col] = codes.map(dict(enumerate(categories)))
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col])
    return df


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

        # train.py now saves a CalibratedClassifierCV wrapping the base
        # model (see its calibration comment), not the base model directly —
        # predict_proba works identically either way, but SHAP needs the
        # actual preprocessor/classifier steps, which only the base model
        # has. Calibration is a monotonic remapping applied after the base
        # model's score, so explaining the base model's decision is still
        # the correct, meaningful explanation of what drove the prediction.
        if hasattr(self.pipeline, "named_steps"):
            base_model = self.pipeline
        else:
            base_model = self.pipeline.calibrated_classifiers_[0].estimator

        df = load_modeling_data()
        train_df, _, _ = time_aware_split(df)
        background_raw = train_df[FEATURE_COLUMNS].sample(min(background_size, len(train_df)), random_state=42)

        if hasattr(base_model, "named_steps"):
            # Single preprocessor+classifier Pipeline (logistic_regression,
            # random_forest, or xgboost — see train.py's model types).
            self.is_ensemble = False
            self.preprocessor = base_model.named_steps["preprocessor"]
            self.classifier = base_model.named_steps["classifier"]
            self.transformed_feature_names = list(self.preprocessor.get_feature_names_out())
            self.background_transformed = self.preprocessor.transform(background_raw)

            model_type = type(self.classifier).__name__
            if model_type == "LogisticRegression":
                self.explainer = shap.LinearExplainer(self.classifier, self.background_transformed)
            elif model_type in ("RandomForestClassifier", "XGBClassifier"):
                self.explainer = shap.TreeExplainer(self.classifier)
            else:
                raise ValueError(f"No SHAP explainer configured for model type: {model_type}")
            self._model_type = model_type
        else:
            # Ensemble (sklearn VotingClassifier, soft-voting over the tuned
            # models — see train.py). Each member pipeline handles its own
            # preprocessing internally, so there's no single shared
            # preprocessor to explain in transformed space. Explained
            # directly on raw features with a model-agnostic Permutation
            # explainer over the ensemble's own predict_proba — slower than
            # LinearExplainer/TreeExplainer, but correct for an arbitrary
            # black-box combiner rather than silently crashing or guessing.
            self.is_ensemble = True
            self.preprocessor = None
            self.classifier = base_model
            self.transformed_feature_names = list(FEATURE_COLUMNS)  # no one-hot expansion at this level
            # SHAP's masker/permutation logic operates on this — must be
            # purely numeric (see _encode_categoricals), not raw strings.
            self.background_transformed = _encode_categoricals(background_raw)

            def _ensemble_predict_proba(x) -> np.ndarray:
                """x is ENCODED (numeric) rows, matching background_transformed's
                format — decoded back to real categories before scoring."""
                decoded_df = _decode_categoricals(x)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    return base_model.predict_proba(decoded_df)

            self._ensemble_predict_proba = _ensemble_predict_proba
            self.explainer = shap.Explainer(
                lambda x: _ensemble_predict_proba(x)[:, 1], self.background_transformed, algorithm="permutation"
            )
            self._model_type = "VotingClassifier"

        # SHAP's additive output scale is NOT the same for every model type:
        # shap.LinearExplainer on a logistic regression is exact on the
        # log-odds scale (needs a sigmoid to recover a probability), but
        # shap.TreeExplainer's scale depends on the specific tree algorithm —
        # probability-space for RandomForestClassifier's probability-
        # averaging trees, but typically log-odds/margin-space for gradient-
        # boosted trees like XGBoost. Rather than hardcode an assumption per
        # model name (wrong for whichever model ISN'T currently selected —
        # and XGBoost couldn't even be verified locally, see train.py), the
        # scale is detected empirically: check which formula (raw sum, or
        # its sigmoid) actually reconstructs a real predict_proba call.
        self.score_scale = self._detect_score_scale()

    def _detect_score_scale(self) -> str:
        sample_transformed = self.background_transformed[:1]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            sample_explanation = self.explainer(sample_transformed, silent=True)
            if self.is_ensemble:
                actual_prob = float(self._ensemble_predict_proba(sample_transformed)[0, 1])
            else:
                actual_prob = float(self.classifier.predict_proba(sample_transformed)[0, 1])

        values, base = sample_explanation.values[0], sample_explanation.base_values[0]
        if values.ndim > 1:  # (n_features, n_classes) — take the positive class
            values = values[:, 1]
            base = np.asarray(base).reshape(-1)[1] if hasattr(base, "__len__") else base
        raw_reconstruction = float(base) + float(values.sum())

        log_odds_error = abs(_sigmoid(np.array(raw_reconstruction)) - actual_prob)
        probability_error = abs(raw_reconstruction - actual_prob)
        return "log_odds" if log_odds_error < probability_error else "probability"

    def explain(self, record: dict) -> dict:
        """Explain a single student-term record (a dict with FEATURE_COLUMNS keys)."""
        raw_df = pd.DataFrame([record])[FEATURE_COLUMNS]
        if self.preprocessor is not None:
            model_input = self.preprocessor.transform(raw_df)
        elif self.is_ensemble:
            model_input = _encode_categoricals(raw_df)
        else:
            model_input = raw_df

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            explanation = self.explainer(model_input, silent=True)

        values, base = explanation.values[0], explanation.base_values[0]
        if values.ndim > 1:  # (n_features, n_classes) — take the positive (persistence) class
            values = values[:, 1]
            base = np.asarray(base).reshape(-1)[1] if hasattr(base, "__len__") else base
        base_value = float(base)

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
            "score_scale": self.score_scale,
            "base_value": round(base_value, 4),
            "total_contribution": round(sum(by_feature.values()), 4),
            "top_factors": top_factors,
            "disclaimer": EXPLANATION_DISCLAIMER,
        }

    def global_importance(self, sample_size: int = 300) -> pd.Series:
        """Mean |SHAP value| per original feature over a sample of test rows —
        a global view of what the model relies on most, for the dashboard."""
        if self.is_ensemble:
            # The Permutation explainer re-evaluates the model many times per
            # row (roughly 2x features x background), so a large sample here
            # is expensive for a black-box ensemble in a way it isn't for
            # LinearExplainer/TreeExplainer's closed-form attributions.
            sample_size = min(sample_size, 50)

        df = load_modeling_data()
        _, _, test_df = time_aware_split(df)
        sample = test_df[FEATURE_COLUMNS].sample(min(sample_size, len(test_df)), random_state=42)
        if self.preprocessor is not None:
            model_input = self.preprocessor.transform(sample)
        elif self.is_ensemble:
            model_input = _encode_categoricals(sample)
        else:
            model_input = sample

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            explanation = self.explainer(model_input, silent=True)

        values = explanation.values
        if values.ndim > 2:
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
    print(f"Detected score scale: {explainer.score_scale}")
    for record in sample.to_dict(orient="records"):
        result = explainer.explain(record)
        print(f"\nstudent {record['student_id']} term {record['term_number']}:")
        print(f"  base_value ({result['score_scale']}): {result['base_value']}")
        for factor in result["top_factors"]:
            print(f"  {factor['feature']:35s} contribution={factor['contribution']:+.4f}  ({factor['direction']})")

    print("\nGlobal feature importance (mean |SHAP|, sample of test set):")
    print(explainer.global_importance())
