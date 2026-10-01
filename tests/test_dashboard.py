"""Tests for the Streamlit dashboard using Streamlit's own headless testing
harness (streamlit.testing.v1.AppTest), which actually executes the script
and surfaces real exceptions — not just an import check.

Requires a trained model for the tabs that depend on one; run
`python -m student_journey.models.train` first if these are skipped.
"""
import pytest
from streamlit.testing.v1 import AppTest

from student_journey.config import PROJECT_ROOT
from student_journey.models.train import MODELS_DIR

MODEL_EXISTS = (MODELS_DIR / "model_pipeline.joblib").exists()
# Absolute path: newer Streamlit versions resolve a relative AppTest.from_file
# path against the file that CALLS from_file (this test file, under tests/),
# not the process's working directory — a real behavior difference caught
# when this first ran in CI against a freshly-installed Streamlit.
APP_PATH = str(PROJECT_ROOT / "src" / "student_journey" / "dashboard" / "app.py")

TAB_NAMES = [
    "Overview", "Predict", "Risk Distribution", "Feature Importance",
    "Persistence Trends", "Bottleneck Analysis", "Model Performance", "Monitoring",
    "Intervention Impact",
]


@pytest.mark.skipif(not MODEL_EXISTS, reason="No trained model; run `python -m student_journey.models.train`.")
def test_app_loads_all_tabs_without_exception():
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception
    assert len(at.tabs) == len(TAB_NAMES)


@pytest.mark.skipif(not MODEL_EXISTS, reason="No trained model; run `python -m student_journey.models.train`.")
def test_overview_tab_shows_dataset_and_model_stats():
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    overview = at.tabs[0]
    assert not at.exception
    assert len(overview.metric) == 8  # 4 dataset stats + 4 model stats


@pytest.mark.skipif(not MODEL_EXISTS, reason="No trained model; run `python -m student_journey.models.train`.")
def test_predict_form_submission_produces_a_prediction():
    at = AppTest.from_file(APP_PATH, default_timeout=120)
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
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    perf_tab = at.tabs[6]
    assert not at.exception
    assert len(perf_tab.metric) == 5  # precision, recall, f1, roc_auc, average_precision


@pytest.mark.skipif(not MODEL_EXISTS, reason="No trained model; run `python -m student_journey.models.train`.")
def test_intervention_impact_tab_defaults_to_high_risk_and_shows_metrics():
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    impact_tab = at.tabs[8]
    assert impact_tab.multiselect[0].value == ["high"]
    metrics = {m.label: m.value for m in impact_tab.metric}
    assert "Estimated additional students retained" in metrics


@pytest.mark.skipif(not MODEL_EXISTS, reason="No trained model; run `python -m student_journey.models.train`.")
def test_intervention_impact_responds_to_slider_changes():
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    impact_tab = at.tabs[8]

    # participation_rate is the second of the three sliders on this tab
    participation_slider = impact_tab.slider[0]
    participation_slider.set_value(1.0).run()
    effect_slider = at.tabs[8].slider[1]
    effect_slider.set_value(1.0).run()
    assert not at.exception

    metrics = {m.label: m.value for m in at.tabs[8].metric}
    # 100% participation + 100% effect should retain everyone in the targeted category
    assert metrics["Scenario expected non-persisters"] == "0"
