"""Unit tests for training-pipeline logic that don't require a full training
run: metric computation, the time-aware split boundaries, candidate-model
construction, and risk-category bucketing.
"""
import numpy as np
import pandas as pd

from student_journey.models.predict import risk_category
from student_journey.models.train import (
    TRAIN_MAX_TERM_ORDER,
    VAL_MAX_TERM_ORDER,
    XGBOOST_AVAILABLE,
    build_candidate_models,
    compute_metrics,
    time_aware_split,
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


def test_build_candidate_models_always_includes_baseline_and_forest():
    models = build_candidate_models(scale_pos_weight=1.0)
    assert "logistic_regression" in models
    assert "random_forest" in models
    if XGBOOST_AVAILABLE:
        assert "xgboost" in models
    else:
        assert "xgboost" not in models


def test_risk_category_thresholds():
    thresholds = {"low_max": 0.2, "medium_max": 0.5}
    assert risk_category(0.1, thresholds) == "low"
    assert risk_category(0.2, thresholds) == "medium"  # boundary is exclusive on low side
    assert risk_category(0.35, thresholds) == "medium"
    assert risk_category(0.5, thresholds) == "high"
    assert risk_category(0.9, thresholds) == "high"
