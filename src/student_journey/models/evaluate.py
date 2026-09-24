"""Full evaluation report for the saved model: the metric suite, confusion
matrix and calibration plots, error analysis, and subgroup performance
slicing on the demographic proxy fields that were deliberately excluded from
the model's input features (see train.py's module docstring for why).

IMPORTANT — subgroup results here are computed on a SYNTHETIC dataset with
fabricated demographic proxy fields. They demonstrate the fairness-evaluation
*method* (how you'd slice metrics by subgroup on a real deployment) and are
not evidence about real-world fairness of anything.

Run (after `python -m student_journey.models.train` has produced a model):
    python -m student_journey.models.evaluate
"""
from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")  # headless-safe backend, no display required
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, brier_score_loss, classification_report

from student_journey.config import DOCS_SCREENSHOTS_DIR, MODELS_DIR
from student_journey.models.train import (
    DEMOGRAPHIC_PROXY_COLUMNS,
    FEATURE_COLUMNS,
    TARGET,
    compute_metrics,
    load_modeling_data,
    time_aware_split,
)

MODEL_PATH = MODELS_DIR / "model_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
REPORT_PATH = MODELS_DIR / "evaluation_report.json"

ERROR_ANALYSIS_FEATURES = [
    "term_gpa", "cumulative_gpa", "credit_completion_rate",
    "cumulative_credit_completion_rate", "courses_withdrawn", "courses_repeated",
    "advising_contact_flag", "financial_aid_flag",
]


def load_model_and_test_set():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run `python -m student_journey.models.train` first.")
    pipeline = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())

    df = load_modeling_data()
    _, _, test_df = time_aware_split(df)
    return pipeline, metadata, test_df


def plot_confusion_matrix(y_true, y_pred, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Not persisted", "Persisted"], ax=ax, colorbar=False
    )
    ax.set_title("Confusion Matrix — held-out test set")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_calibration_curve(y_true, y_prob, out_path, y_prob_uncalibrated=None):
    """If y_prob_uncalibrated is given, plots both curves for a direct
    before/after comparison of the CalibratedClassifierCV fix in train.py."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if y_prob_uncalibrated is not None:
        prob_true_raw, prob_pred_raw = calibration_curve(y_true, y_prob_uncalibrated, n_bins=10, strategy="quantile")
        ax.plot(prob_pred_raw, prob_true_raw, marker="o", label="Before calibration", color="#e34948")
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    label = "After calibration" if y_prob_uncalibrated is not None else "Model"
    ax.plot(prob_pred, prob_true, marker="o", label=label, color="#2a78d6")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed persistence rate")
    ax.set_title("Calibration curve — held-out test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def get_uncalibrated_probabilities(pipeline, X_test):
    """If the saved pipeline is a CalibratedClassifierCV (see train.py),
    returns the base (pre-calibration) model's probabilities for
    before/after comparison; None if the saved pipeline isn't calibrated."""
    if not hasattr(pipeline, "calibrated_classifiers_"):
        return None
    base_pipeline = pipeline.calibrated_classifiers_[0].estimator
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return base_pipeline.predict_proba(X_test)[:, 1]


def error_analysis(test_df: pd.DataFrame, y_true, y_pred) -> dict:
    labels = pd.Series(
        [
            "true_positive" if t == 1 and p == 1 else
            "true_negative" if t == 0 and p == 0 else
            "false_positive" if t == 0 and p == 1 else
            "false_negative"
            for t, p in zip(y_true, y_pred)
        ],
        index=test_df.index,
    )
    summary = test_df[ERROR_ANALYSIS_FEATURES].groupby(labels).mean()
    counts = labels.value_counts()
    return {
        "counts": counts.to_dict(),
        "feature_means_by_outcome": summary.round(3).to_dict(orient="index"),
    }


def subgroup_performance(test_df: pd.DataFrame, y_true, y_prob, y_pred) -> dict:
    results = {}
    for col in DEMOGRAPHIC_PROXY_COLUMNS:
        by_group = {}
        for value, group_idx in test_df.groupby(col).groups.items():
            positions = test_df.index.get_indexer(group_idx)
            g_true, g_prob = y_true[positions], y_prob[positions]
            if len(set(g_true)) < 2:
                # ROC-AUC is undefined with only one class present in the slice
                by_group[str(value)] = {"n": len(positions), "note": "insufficient class diversity for ROC-AUC"}
                continue
            by_group[str(value)] = compute_metrics(g_true, g_prob)
        results[col] = by_group
    return results


def main() -> None:
    pipeline, metadata, test_df = load_model_and_test_set()

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET].astype(int).values

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_prob)
    print(f"Model: {metadata['selected_model']} (version {metadata['model_version']})")
    print(f"Test set: {len(test_df)} rows\n")
    print("Overall metrics:")
    print(json.dumps(metrics, indent=2))

    print("\nsklearn classification_report:")
    print(classification_report(y_test, y_pred, target_names=["not_persisted", "persisted"]))

    y_prob_uncalibrated = get_uncalibrated_probabilities(pipeline, X_test)
    if y_prob_uncalibrated is not None:
        print(
            f"\nCalibration fix: Brier score {brier_score_loss(y_test, y_prob_uncalibrated):.4f} (before) "
            f"-> {metrics['brier_score']:.4f} (after)"
        )

    DOCS_SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(y_test, y_pred, DOCS_SCREENSHOTS_DIR / "confusion_matrix.png")
    plot_calibration_curve(
        y_test, y_prob, DOCS_SCREENSHOTS_DIR / "calibration_curve.png", y_prob_uncalibrated=y_prob_uncalibrated
    )
    print(f"\nSaved confusion matrix -> {DOCS_SCREENSHOTS_DIR / 'confusion_matrix.png'}")
    print(f"Saved calibration curve -> {DOCS_SCREENSHOTS_DIR / 'calibration_curve.png'}")

    errors = error_analysis(test_df, y_test, y_pred)
    print("\nError analysis — outcome counts:")
    print(json.dumps(errors["counts"], indent=2))
    print("\nError analysis — mean feature values by outcome:")
    print(json.dumps(errors["feature_means_by_outcome"], indent=2))

    print(
        "\n--- Subgroup performance (SYNTHETIC demographic proxies; "
        "illustrates method only, not a real fairness finding) ---"
    )
    subgroups = subgroup_performance(test_df, y_test, y_prob, y_pred)
    print(json.dumps(subgroups, indent=2))

    report = {
        "model_version": metadata["model_version"],
        "model_type": metadata["selected_model"],
        "overall_metrics": metrics,
        "error_analysis": errors,
        "subgroup_performance": subgroups,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved full evaluation report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
