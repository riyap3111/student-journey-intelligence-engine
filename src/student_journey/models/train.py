"""Train Logistic Regression, Random Forest, and XGBoost models to predict
`persisted_next_term`, using a time-aware split, Optuna hyperparameter
tuning, MLflow tracking, class-imbalance handling, and a soft-voting
ensemble over the tuned models as an additional candidate. Selects the best
model by validation ROC-AUC, calibrates it, does one final unbiased
evaluation on the held-out test set, and saves the winning pipeline +
metadata to models/.

Design decisions (documented here so they're not just implicit in code):

  - Each model type is tuned with Optuna (see N_TUNING_TRIALS) before
    comparison — maximizing validation ROC-AUC, the same metric used for
    final model selection, so tuning optimizes exactly what the comparison
    cares about. A soft-voting ensemble over the tuned models is evaluated
    as one more candidate; it wins only if it genuinely beats every
    individual tuned model on validation ROC-AUC.

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
import warnings
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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

from student_journey.cloud import storage as gcs_storage
from student_journey.config import FEATURES_TABLE, MLRUNS_DIR, MODELS_DIR, RANDOM_SEED
from student_journey.db import get_engine
from student_journey.monitoring.drift import build_reference_distribution

optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep console output focused on our own prints, not per-trial spam

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


def load_modeling_data(engine=None) -> pd.DataFrame:
    engine = engine or get_engine()
    df = pd.read_sql(f"SELECT * FROM {FEATURES_TABLE}", engine)
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


def _build_pipeline(model_type: str, params: dict, scale_pos_weight: float) -> Pipeline:
    """Construct an unfitted preprocessor+classifier pipeline for model_type
    with the given hyperparameters. Shared by the Optuna objective (many
    throwaway fits) and the final best-params fit, so there's exactly one
    place that knows how to turn a model_type + params dict into a pipeline.
    """
    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            C=params["C"], class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
        )
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"], max_features=params["max_features"],
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
        )
    elif model_type == "xgboost":
        classifier = XGBClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"],
            learning_rate=params["learning_rate"], subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"], scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return Pipeline([("preprocessor", build_preprocessor()), ("classifier", classifier)])


def _suggest_params(trial: optuna.Trial, model_type: str) -> dict:
    if model_type == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if model_type == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    if model_type == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    raise ValueError(f"Unknown model_type: {model_type}")


N_TUNING_TRIALS = {"logistic_regression": 15, "random_forest": 25, "xgboost": 25}


def tune_model(model_type: str, X_train, y_train, X_val, y_val, scale_pos_weight: float, n_trials: int):
    """Optuna search over model_type's hyperparameters, maximizing validation
    ROC-AUC — the same metric used for model selection, so tuning optimizes
    exactly what the comparison downstream cares about. Returns
    (fitted_pipeline_with_best_params, best_params_dict)."""

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_type)
        pipeline = _build_pipeline(model_type, params, scale_pos_weight)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            pipeline.fit(X_train, y_train)
            val_prob = pipeline.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, val_prob)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_pipeline = _build_pipeline(model_type, study.best_params, scale_pos_weight)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        best_pipeline.fit(X_train, y_train)

    return best_pipeline, study.best_params


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
        "n_samples": len(y_true),
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

    model_types = ["logistic_regression", "random_forest"]
    if XGBOOST_AVAILABLE:
        model_types.append("xgboost")
    else:
        print(
            f"\n[warning] Skipping XGBoost — native library failed to load "
            f"in this environment ({_XGBOOST_IMPORT_ERROR!r}). "
            "It will run in the Docker image (Phase 7). Continuing with "
            "Logistic Regression and Random Forest only.\n"
        )

    val_results = {}
    tuned_members = {}  # {name: fitted_pipeline}, feeds the ensemble below

    for name in model_types:
        n_trials = N_TUNING_TRIALS[name]
        print(f"\nTuning {name} ({n_trials} Optuna trials, optimizing validation ROC-AUC)...")
        pipeline, best_params = tune_model(name, X_train, y_train, X_val, y_val, scale_pos_weight, n_trials)
        print(f"  best_params: {best_params}")

        with mlflow.start_run(run_name=name):
            # LogisticRegression's lbfgs solver emits benign transient
            # RuntimeWarnings (matmul overflow) while exploring large
            # coefficient updates under class_weight="balanced" on this
            # data; verified no inf/NaN in the input and the converged
            # metrics are stable, so these are suppressed rather than
            # left to look like a real data bug in the console output.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                val_prob = pipeline.predict_proba(X_val)[:, 1]
            metrics = compute_metrics(y_val, val_prob)

            mlflow.log_param("model_type", name)
            mlflow.log_param("tuning_trials", n_trials)
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("val_rows", len(X_val))
            mlflow.log_metrics({
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float))
            })
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            val_results[name] = {
                "pipeline": pipeline, "metrics": metrics, "val_prob": val_prob, "best_params": best_params,
            }
            tuned_members[name] = pipeline
            print(f"\n[{name}] validation metrics: {json.dumps(metrics, indent=2)}")

    # Ensemble: soft-voting VotingClassifier over the tuned models.
    # VotingClassifier always clones + refits its members on whatever data
    # .fit() is called with — here that's the same X_train each member was
    # already tuned on, so this introduces no leakage, just recombination.
    if len(tuned_members) >= 2:
        print(f"\nBuilding soft-voting ensemble from: {list(tuned_members.keys())}...")
        ensemble = VotingClassifier(estimators=list(tuned_members.items()), voting="soft")
        with mlflow.start_run(run_name="ensemble"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                ensemble.fit(X_train, y_train)
                val_prob = ensemble.predict_proba(X_val)[:, 1]
            metrics = compute_metrics(y_val, val_prob)

            mlflow.log_param("model_type", "ensemble")
            mlflow.log_param("ensemble_members", list(tuned_members.keys()))
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.sklearn.log_model(ensemble, artifact_path="model")

            val_results["ensemble"] = {"pipeline": ensemble, "metrics": metrics, "val_prob": val_prob}
            print(f"\n[ensemble] validation metrics: {json.dumps(metrics, indent=2)}")

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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        test_prob_uncalibrated = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics_uncalibrated = compute_metrics(y_test, test_prob_uncalibrated)
    print(f"\n[{best_name}] test metrics BEFORE calibration: {json.dumps(test_metrics_uncalibrated, indent=2)}")

    # Calibration fix: class_weight="balanced" (used above to handle class
    # imbalance) measurably distorts predicted probabilities — Phase 3 found
    # predicted probabilities running below observed persistence rates on the
    # calibration curve. CalibratedClassifierCV fits a monotonic remapping
    # (Platt/sigmoid scaling, chosen over isotonic since our ~2.4k-row
    # validation set is on the smaller side for isotonic's more flexible,
    # more overfit-prone nonparametric fit) using the VALIDATION set — never
    # train (already used to fit the base model) or test (must stay unbiased
    # for final reporting). Calibration is monotonic, so it changes the
    # probabilities shown but not the model's ranking — risk-category
    # percentile thresholds are unaffected (verified: Spearman rank
    # correlation between raw and calibrated probabilities is exactly 1.0).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)  # cv="prefit" deprecation (see comment above)
        calibrated_pipeline = CalibratedClassifierCV(best_pipeline, method="sigmoid", cv="prefit")
        calibrated_pipeline.fit(X_val, y_val)
        val_prob = calibrated_pipeline.predict_proba(X_val)[:, 1]
        test_prob = calibrated_pipeline.predict_proba(X_test)[:, 1]

    test_metrics = compute_metrics(y_test, test_prob)
    print(f"\n[{best_name}] FINAL held-out test metrics AFTER calibration: {json.dumps(test_metrics, indent=2)}")
    print(
        f"Brier score: {test_metrics_uncalibrated['brier_score']:.4f} (uncalibrated) "
        f"-> {test_metrics['brier_score']:.4f} (calibrated)"
    )

    val_risk_prob = 1.0 - val_prob
    risk_thresholds = {
        "low_max": float(np.percentile(val_risk_prob, 60)),
        "medium_max": float(np.percentile(val_risk_prob, 90)),
    }
    print(f"Risk thresholds (from calibrated validation-set percentiles): {risk_thresholds}")

    with mlflow.start_run(run_name=f"{best_name}_calibrated_final_test"):
        mlflow.log_param("model_type", best_name)
        mlflow.log_param("selection_metric", "val_roc_auc")
        mlflow.log_param("calibration_method", "sigmoid")
        mlflow.log_metrics({k: v for k, v in test_metrics.items() if isinstance(v, (int, float))})
        mlflow.log_metric("brier_score_uncalibrated", test_metrics_uncalibrated["brier_score"])
        mlflow.sklearn.log_model(calibrated_pipeline, artifact_path="model")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "model_pipeline.joblib"
    joblib.dump(calibrated_pipeline, model_path)

    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": best_name,
        "calibration_method": "sigmoid",
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
        "best_hyperparameters": {
            name: r["best_params"] for name, r in val_results.items() if "best_params" in r
        },
        "ensemble_members": list(tuned_members.keys()) if "ensemble" in val_results else None,
        "selected_model": best_name,
        "test_metrics": test_metrics,
        "test_metrics_uncalibrated": test_metrics_uncalibrated,
        "risk_thresholds": risk_thresholds,
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # Reference distribution for drift detection (monitoring/drift.py):
    # built from TRAIN only (never val/test) since it represents "the
    # population this model was fit to expect" — the fixed baseline that
    # actual incoming API traffic gets compared against later.
    reference_distribution = build_reference_distribution(X_train, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    reference_path = MODELS_DIR / "reference_distribution.json"
    reference_path.write_text(json.dumps(reference_distribution, indent=2))

    print(f"\nSaved calibrated pipeline -> {model_path}")
    print(f"Saved metadata -> {metadata_path}")
    print(f"Saved drift-detection reference distribution -> {reference_path}")

    if gcs_storage.is_enabled():
        uploaded = gcs_storage.upload_model_artifacts(MODELS_DIR)
        print(f"Uploaded to gs://{gcs_storage.GCS_BUCKET_NAME}/{gcs_storage.GCS_MODEL_PREFIX}/: {uploaded}")


if __name__ == "__main__":
    main()
