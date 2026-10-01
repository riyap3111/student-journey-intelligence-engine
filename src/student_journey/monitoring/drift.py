"""Population Stability Index (PSI) based feature drift detection.

PSI is the standard drift metric used in production credit-risk and ML
scoring systems: it compares a "reference" distribution (here, the training
set a model was fit on) against a "current" distribution (here, the actual
feature values the API has been asked to score recently) over a shared set
of bins, and sums a divergence term per bin:

    PSI = sum_over_bins( (actual_pct - expected_pct) * ln(actual_pct / expected_pct) )

Conventional thresholds (used industry-wide, not invented here):
    PSI < 0.10            -> no significant population shift
    0.10 <= PSI < 0.25     -> moderate shift, worth investigating
    PSI >= 0.25            -> significant shift, the model may need retraining

Numeric features are bucketed into deciles of the REFERENCE distribution
(so bin edges are fixed at reference-build time, not recomputed per
comparison — otherwise two different "current" samples would be judged
against different bins and their PSI values wouldn't be comparable).
Categorical features compare category frequency instead of quantile bins.

This module is pure computation with no I/O: `build_reference_distribution`
is called once at training time (train.py) and its output is saved to
reference_distribution.json; `compute_drift_report` is called at request
time (the API's /monitoring/drift endpoint, and the dashboard) against
whatever recent feature rows are available (see prediction_log.py for where
those come from).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PSI_WARNING_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25
DEFAULT_N_BINS = 10
DEFAULT_MIN_SAMPLES = 30
_EPSILON = 1e-4  # floor on bin proportions so PSI's log term never divides by zero


def build_reference_distribution(
    df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str], n_bins: int = DEFAULT_N_BINS
) -> dict:
    """Summarizes df's feature distributions into a JSON-serializable
    reference. For numeric features, bin edges are the reference data's own
    quantiles (so ~n_bins/n_bins of the reference falls in each bin by
    construction); for categorical features, it's just observed frequency.
    """
    reference = {"numeric": {}, "categorical": {}, "n_reference_rows": len(df)}

    for feature in numeric_features:
        series = df[feature].dropna().astype(float)
        edges = np.unique(np.quantile(series, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 2:
            # A degenerate (near-constant) feature has no meaningful quantile
            # spread — fall back to a single bin covering its full range.
            edges = np.array([series.min() - 1.0, series.max() + 1.0])
        edges = edges.copy()
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(series, bins=edges)
        proportions = counts / counts.sum()
        reference["numeric"][feature] = {
            "bin_edges": edges.tolist(),
            "proportions": proportions.tolist(),
        }

    for feature in categorical_features:
        counts = df[feature].value_counts(normalize=True)
        reference["categorical"][feature] = {
            "categories": counts.index.tolist(),
            "proportions": counts.values.tolist(),
        }

    return reference


def load_reference_distribution(models_dir: Path) -> dict | None:
    """Loads the reference_distribution.json saved by train.py, or None if
    no model has been trained yet (mirrors PersistenceModel's own
    file-not-found-is-a-valid-state handling in predict.py)."""
    path = Path(models_dir) / "reference_distribution.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _psi_from_proportions(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(np.asarray(expected, dtype=float), _EPSILON, None)
    actual = np.clip(np.asarray(actual, dtype=float), _EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _numeric_psi(series: pd.Series, reference: dict) -> float:
    edges = np.array(reference["bin_edges"])
    counts, _ = np.histogram(series.dropna().astype(float), bins=edges)
    actual_proportions = counts / counts.sum()
    return _psi_from_proportions(reference["proportions"], actual_proportions)


def _categorical_psi(series: pd.Series, reference: dict) -> float:
    categories = reference["categories"]
    expected = dict(zip(categories, reference["proportions"]))
    observed = series.value_counts(normalize=True)
    actual = [observed.get(c, 0.0) for c in categories]
    return _psi_from_proportions([expected[c] for c in categories], actual)


def classify_psi(psi: float | None) -> str:
    if psi is None:
        return "insufficient_data"
    if psi < PSI_WARNING_THRESHOLD:
        return "stable"
    if psi < PSI_ALERT_THRESHOLD:
        return "moderate_shift"
    return "significant_shift"


_STATUS_SEVERITY = {"stable": 0, "insufficient_data": 0, "moderate_shift": 1, "significant_shift": 2}


def compute_drift_report(
    current_df: pd.DataFrame, reference: dict, min_samples: int = DEFAULT_MIN_SAMPLES
) -> dict:
    """Compares current_df's feature distributions against a reference built
    by build_reference_distribution. Returns a JSON-serializable report with
    a per-feature PSI + status and an overall status (the worst of any
    individual feature's status, since one badly-drifted feature is enough
    to warrant investigating even if most features look stable).
    """
    n_current = len(current_df)
    features = []

    for feature, feature_ref in reference.get("numeric", {}).items():
        if feature not in current_df.columns or n_current < min_samples:
            psi = None
        else:
            psi = _numeric_psi(current_df[feature], feature_ref)
        features.append({"feature": feature, "feature_type": "numeric", "psi": psi, "status": classify_psi(psi)})

    for feature, feature_ref in reference.get("categorical", {}).items():
        if feature not in current_df.columns or n_current < min_samples:
            psi = None
        else:
            psi = _categorical_psi(current_df[feature], feature_ref)
        features.append({"feature": feature, "feature_type": "categorical", "psi": psi, "status": classify_psi(psi)})

    overall_status = max(
        (f["status"] for f in features), key=lambda s: _STATUS_SEVERITY[s], default="insufficient_data"
    )

    return {
        "overall_status": overall_status,
        "n_reference_rows": reference.get("n_reference_rows"),
        "n_current_rows": n_current,
        "min_samples_required": min_samples,
        "features": sorted(features, key=lambda f: (-_STATUS_SEVERITY[f["status"]], f["feature"])),
    }
