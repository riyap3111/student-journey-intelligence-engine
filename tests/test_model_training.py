"""Unit tests for training-pipeline logic that don't require a full training
run: metric computation, the time-aware split boundaries, pipeline/param
construction for hyperparameter tuning, a fast end-to-end tuning smoke test
on fabricated data, and risk-category bucketing.
"""
import warnings

import numpy as np
import optuna
import pandas as pd
import pytest

from student_journey.models.predict import risk_category
from student_journey.models.train import (
    CATEGORICAL_FEATURES,
    N_TUNING_TRIALS,
    NUMERIC_FEATURES,
    TRAIN_MAX_TERM_ORDER,
    VAL_MAX_TERM_ORDER,
    XGBOOST_AVAILABLE,
    _build_pipeline,
    _suggest_params,
    compute_metrics,
    time_aware_split,
    tune_model,
)


def test_compute_metrics_on_known_values():
    y_true = np.array([1, 1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.4, 0.3, 0.6])  # threshold 0.5 -> pred = [1,1,0,0,1]
    metrics = compute_metrics(y_true, y_prob)

    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 2}
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 2 / 3
    assert metrics["n_samples"] == 5
    assert metrics["positive_rate"] == 0.6


def test_time_aware_split_respects_term_order_boundaries():
    df = pd.DataFrame({"term_order": [1, TRAIN_MAX_TERM_ORDER, TRAIN_MAX_TERM_ORDER + 1, VAL_MAX_TERM_ORDER, VAL_MAX_TERM_ORDER + 1]})
    train, val, test = time_aware_split(df)

    assert (train["term_order"] <= TRAIN_MAX_TERM_ORDER).all()
    assert ((val["term_order"] > TRAIN_MAX_TERM_ORDER) & (val["term_order"] <= VAL_MAX_TERM_ORDER)).all()
    assert (test["term_order"] > VAL_MAX_TERM_ORDER).all()
    assert len(train) + len(val) + len(test) == len(df)


def test_n_tuning_trials_covers_all_model_types():
    assert set(N_TUNING_TRIALS.keys()) == {"logistic_regression", "random_forest", "xgboost"}


def test_suggest_params_returns_expected_keys():
    study = optuna.create_study()
    lr_params = _suggest_params(study.ask(), "logistic_regression")
    assert set(lr_params.keys()) == {"C"}

    rf_params = _suggest_params(study.ask(), "random_forest")
    assert set(rf_params.keys()) == {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}

    xgb_params = _suggest_params(study.ask(), "xgboost")
    assert set(xgb_params.keys()) == {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"}


def test_build_pipeline_known_model_types():
    lr_pipeline = _build_pipeline("logistic_regression", {"C": 1.0}, scale_pos_weight=1.0)
    assert set(lr_pipeline.named_steps) == {"preprocessor", "classifier"}

    rf_pipeline = _build_pipeline(
        "random_forest",
        {"n_estimators": 50, "max_depth": 3, "min_samples_leaf": 2, "max_features": "sqrt"},
        scale_pos_weight=1.0,
    )
    assert set(rf_pipeline.named_steps) == {"preprocessor", "classifier"}


def test_build_pipeline_rejects_unknown_model_type():
    with pytest.raises(ValueError):
        _build_pipeline("not_a_real_model", {}, scale_pos_weight=1.0)


@pytest.fixture
def fabricated_features():
    """Small, self-contained fabricated dataset with the right columns/dtypes
    — not going through the full generator/feature pipeline, just enough
    structure for tune_model to run its Optuna loop end to end quickly."""
    rng = np.random.default_rng(0)
    n = 200
    data = {col: rng.normal(size=n) for col in NUMERIC_FEATURES}
    df = pd.DataFrame(data)
    df["program"] = rng.choice(["Business", "Engineering"], size=n)
    df["entry_type"] = rng.choice(["first_time", "transfer"], size=n)
    y = rng.integers(0, 2, size=n)
    return df, y


def test_tune_model_runs_end_to_end_and_returns_fitted_pipeline(fabricated_features):
    X, y = fabricated_features
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        pipeline, best_params = tune_model(
            "logistic_regression", X, y, X, y, scale_pos_weight=1.0, n_trials=2
        )
    assert "C" in best_params
    assert hasattr(pipeline, "predict_proba")
    probs = pipeline.predict_proba(X)[:, 1]
    assert ((probs >= 0) & (probs <= 1)).all()


def test_xgboost_flag_is_boolean():
    assert isinstance(XGBOOST_AVAILABLE, bool)


def test_categorical_and_numeric_features_disjoint():
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)


def test_risk_category_thresholds():
    thresholds = {"low_max": 0.2, "medium_max": 0.5}
    assert risk_category(0.1, thresholds) == "low"
    assert risk_category(0.2, thresholds) == "medium"  # boundary is exclusive on low side
    assert risk_category(0.35, thresholds) == "medium"
    assert risk_category(0.5, thresholds) == "high"
    assert risk_category(0.9, thresholds) == "high"
