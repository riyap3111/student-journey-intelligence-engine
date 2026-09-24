"""Tests for the Streamlit dashboard using Streamlit's own headless testing
harness (streamlit.testing.v1.AppTest), which actually executes the script
and surfaces real exceptions — not just an import check.

Requires a trained model for the tabs that depend on one; run
`python -m student_journey.models.train` first if these are skipped.
"""
import pytest
from streamlit.testing.v1 import AppTest

from student_journey.models.train import MODELS_DIR

MODEL_EXISTS = (MODELS_DIR / "model_pipeline.joblib").exists()
APP_PATH = "src/student_journey/dashboard/app.py"

TAB_NAMES = [
    "Overview", "Predict", "Risk Distribution", "Feature Importance",
    "Persistence Trends", "Bottleneck Analysis", "Model Performance", "Monitoring",
]


@pytest.mark.skipif(not MODEL_EXISTS, reason="No trained model; run `python -m student_journey.models.train`.")
def test_app_loads_all_tabs_without_exception():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception
    assert len(at.tabs) == len(TAB_NAMES)


@pytest.mark.skipif(not MODEL_EXISTS, reason="No trained model; run `python -m student_journey.models.train`.")
def test_overview_tab_shows_dataset_and_model_stats():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    overview = at.tabs[0]
    assert not at.exception
    assert len(overview.metric) == 8  # 4 dataset stats + 4 model stats


@pytest.mark.skipif(not MODEL_EXISTS, reason="No trained model; run `python -m student_journey.models.train`.")
def test_predict_form_submission_produces_a_prediction():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.tabs[1].button[0].click().run()
    assert not at.exception

    predict_tab = at.tabs[1]
    metrics = {m.label: m.value for m in predict_tab.metric}
    assert "Persistence probability" in metrics
    assert metrics["Persistence probability"].endswith("%")
    assert len(predict_tab.info) == 1  # the disclaimer


@pytest.mark.skipif(not MODEL_EXISTS, reason="No trained model; run `python -m student_journey.models.train`.")
def test_model_performance_tab_shows_metrics():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    perf_tab = at.tabs[6]
    assert not at.exception
    assert len(perf_tab.metric) == 5  # precision, recall, f1, roc_auc, average_precision
