"""Train baseline (Logistic Regression), Random Forest, and XGBoost models to
predict `persisted_next_term`, using a time-aware split, MLflow tracking, and
class-imbalance handling. Selects the best model by validation ROC-AUC, does
one final unbiased evaluation on the held-out test set, and saves the winning
pipeline + metadata to models/.

Design decisions (documented here so they're not just implicit in code):

  - Time-aware split by `term_order` (the calendar sequence), NOT a random
    row split. Train on earlier terms, validate and test on strictly later
    terms. A random split would let the model see a student's later-term
    outcomes while "predicting" an earlier one, which is unrealistic for a
    deployed system that only ever has the past to work with.

  - Demographic proxy fields (age_band, gender, first_gen_flag,
    distance_from_campus_band) are deliberately EXCLUDED from the model's
    input features. They're kept in the dataset only to slice evaluation
    metrics after prediction (see evaluate.py), not to predict from — using
    them as direct predictive inputs would risk encoding indirect
    discrimination into the score. Only academic/behavioral/enrollment
    signals are used to predict.

  - `student_id`, `term_id`, `term_order` are identifiers/time-index, not
    features. `is_graduating_term` and `censored_flag` are always 0 for the
    modeling population (rows with a non-null label), so they carry no
    signal and are dropped to avoid confusion.

  - Class imbalance (roughly 5:1 persisted:not-persisted) is handled with
    class weighting (class_weight="balanced" / XGBoost's scale_pos_weight)
    rather than resampling, so no synthetic rows are introduced beyond the
    already-synthetic dataset.

  - Every metric printed or logged here comes from actually running these
    models on this data — nothing is hardcoded.

Run:
    python -m student_journey.models.train
"""
from __future__ import annotations

import json
import sqlite3
import warnings
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from student_journey.config import DB_PATH, FEATURES_TABLE, MLRUNS_DIR, MODELS_DIR, RANDOM_SEED

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as exc:  # xgboost raises its own XGBoostError, not just ImportError/OSError
    # On some local dev machines (e.g. macOS without Homebrew's libomp) the
    # xgboost native library can't load. XGBoost is still fully wired up
    # below and runs normally in the Docker image (Phase 7, Linux), where
    # this dependency is present. We degrade gracefully here rather than
    # hard-failing the whole training run over an environment gap.
    XGBOOST_AVAILABLE = False
    _XGBOOST_IMPORT_ERROR = exc

NUMERIC_FEATURES = [
    "term_number", "enrollment_intensity_numeric",
    "credits_attempted", "credits_completed", "credit_completion_rate",
    "cumulative_credits_attempted", "cumulative_credits_completed", "cumulative_credit_completion_rate",
    "term_gpa", "cumulative_gpa", "gpa_change", "academic_momentum",
    "courses_withdrawn", "cumulative_withdrawals", "courses_repeated", "cumulative_repeats",
    "advising_contact_flag", "financial_aid_flag", "prior_term_enrolled_flag",
]
CATEGORICAL_FEATURES = ["program", "entry_type"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Kept in the dataframe for subgroup fairness slicing in evaluate.py, never
# passed to the model as an input feature.
DEMOGRAPHIC_PROXY_COLUMNS = ["age_band", "gender", "first_gen_flag", "distance_from_campus_band"]

TARGET = "persisted_next_term"

TRAIN_MAX_TERM_ORDER = 12
VAL_MAX_TERM_ORDER = 14
MODEL_VERSION = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def load_modeling_data(db_path=DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(f"SELECT * FROM {FEATURES_TABLE}", conn)
    finally:
        conn.close()
    return df[df[TARGET].notna()].copy()


def time_aware_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["term_order"] <= TRAIN_MAX_TERM_ORDER]
    val = df[(df["term_order"] > TRAIN_MAX_TERM_ORDER) & (df["term_order"] <= VAL_MAX_TERM_ORDER)]
    test = df[df["term_order"] > VAL_MAX_TERM_ORDER]
    return train, val, test


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_candidate_models(scale_pos_weight: float) -> dict[str, Pipeline]:
    models = {
        "logistic_regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=300, max_depth=8, class_weight="balanced",
                random_state=RANDOM_SEED, n_jobs=-1,
            )),
        ]),
    }

    if XGBOOST_AVAILABLE:
        models["xgboost"] = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=-1,
            )),
        ])
    else:
        print(
            f"\n[warning] Skipping XGBoost — native library failed to load "
            f"in this environment ({_XGBOOST_IMPORT_ERROR!r}). "
            "It will run in the Docker image (Phase 7). Continuing with "
            "Logistic Regression and Random Forest only.\n"
        )

    return models


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }


def main() -> None:
    df = load_modeling_data()
    train_df, val_df, test_df = time_aware_split(df)
    print(f"Train: {len(train_df)} rows (term_order <= {TRAIN_MAX_TERM_ORDER})")
    print(f"Val:   {len(val_df)} rows ({TRAIN_MAX_TERM_ORDER} < term_order <= {VAL_MAX_TERM_ORDER})")
    print(f"Test:  {len(test_df)} rows (term_order > {VAL_MAX_TERM_ORDER})")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET].astype(int).values
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[TARGET].astype(int).values
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET].astype(int).values

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"Train class balance: {pos} positive / {neg} negative (scale_pos_weight={scale_pos_weight:.3f})")

    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    mlflow.set_experiment("student_persistence")

    models = build_candidate_models(scale_pos_weight)
    val_results = {}

    for name, pipeline in models.items():
        with mlflow.start_run(run_name=name):
            # LogisticRegression's lbfgs solver emits benign transient
            # RuntimeWarnings (matmul overflow) while exploring large
            # coefficient updates under class_weight="balanced" on this
            # data; verified no inf/NaN in the input and the converged
            # metrics are stable, so these are suppressed rather than
            # left to look like a real data bug in the console output.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                pipeline.fit(X_train, y_train)
                val_prob = pipeline.predict_proba(X_val)[:, 1]
            metrics = compute_metrics(y_val, val_prob)

            mlflow.log_param("model_type", name)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("val_rows", len(X_val))
            mlflow.log_metrics({
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float))
            })
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            val_results[name] = {"pipeline": pipeline, "metrics": metrics, "val_prob": val_prob}
            print(f"\n[{name}] validation metrics: {json.dumps(metrics, indent=2)}")

    best_name = max(val_results, key=lambda k: val_results[k]["metrics"]["roc_auc"])
    best_pipeline = val_results[best_name]["pipeline"]
    print(f"\nSelected best model by validation ROC-AUC: {best_name}")

    # Risk-category thresholds are derived from the VALIDATION set's score
    # distribution (never the test set, to keep test purely for unbiased
    # final reporting). Round-number thresholds (e.g. "high risk if risk >=
    # 0.35") turned out to flag ~44% of students as high-risk against an
    # actual ~14% attrition rate — unusable alert-fatigue for an advisor
    # tool. Instead we bucket by percentile of the model's own risk score,
    # so "high" always means "the ~10% highest-risk students by this
    # model's ranking" regardless of how the raw probabilities are shaped
    # by class-weighting.
    val_risk_prob = 1.0 - val_results[best_name]["val_prob"]
    risk_thresholds = {
        "low_max": float(np.percentile(val_risk_prob, 60)),
        "medium_max": float(np.percentile(val_risk_prob, 90)),
    }
    print(f"Risk thresholds (from validation set percentiles): {risk_thresholds}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        test_prob = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_prob)
    print(f"\n[{best_name}] FINAL held-out test metrics: {json.dumps(test_metrics, indent=2)}")

    with mlflow.start_run(run_name=f"{best_name}_final_test"):
        mlflow.log_param("model_type", best_name)
        mlflow.log_param("selection_metric", "val_roc_auc")
        mlflow.log_metrics({k: v for k, v in test_metrics.items() if isinstance(v, (int, float))})
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "model_pipeline.joblib"
    joblib.dump(best_pipeline, model_path)

    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": best_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_demographic_proxy_columns": DEMOGRAPHIC_PROXY_COLUMNS,
        "target": TARGET,
        "train_max_term_order": TRAIN_MAX_TERM_ORDER,
        "val_max_term_order": VAL_MAX_TERM_ORDER,
        "validation_metrics_by_model": {
            name: r["metrics"] for name, r in val_results.items()
        },
        "selected_model": best_name,
        "test_metrics": test_metrics,
        "risk_thresholds": risk_thresholds,
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved pipeline -> {model_path}")
    print(f"Saved metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
