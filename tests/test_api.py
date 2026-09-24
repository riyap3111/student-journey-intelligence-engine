"""Tests for the FastAPI service: happy paths for all four endpoints, input
validation, and graceful degradation when no trained model is present.

Requires a trained model at models/model_pipeline.joblib for the "happy
path" tests — run `python -m student_journey.models.train` first if those
are skipped.
"""
import pytest
from fastapi.testclient import TestClient

from student_journey.models.train import MODELS_DIR

MODEL_EXISTS = (MODELS_DIR / "model_pipeline.joblib").exists()

VALID_RECORD = {
    "term_number": 3,
    "enrollment_intensity": "full_time",
    "credits_attempted": 15,
    "credits_completed": 12,
    "credit_completion_rate": 0.8,
    "cumulative_credits_attempted": 42,
    "cumulative_credits_completed": 34,
    "cumulative_credit_completion_rate": 0.81,
    "term_gpa": 2.1,
    "cumulative_gpa": 2.4,
    "gpa_change": -0.3,
    "academic_momentum": -0.25,
    "courses_withdrawn": 1,
    "cumulative_withdrawals": 2,
    "courses_repeated": 0,
    "cumulative_repeats": 1,
    "advising_contact_flag": 0,
    "financial_aid_flag": 1,
    "prior_term_enrolled_flag": 1,
    "program": "Business",
    "entry_type": "first_time",
}


@pytest.fixture
def client():
    from student_journey.api.main import app
    with TestClient(app) as c:
        yield c


@pytest.mark.skipif(not MODEL_EXISTS, reason="No trained model; run `python -m student_journey.models.train`.")
def test_health_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]


@pytest.mark.skipif(not MODEL_EXISTS, reason="No trained model; run `python -m student_journey.models.train`.")
def test_model_info_matches_metadata(client):
    r = client.get("/model_info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_type"] in ("logistic_regression", "random_forest", "xgboost")
    assert "test_metrics" in body and "roc_auc" in body["test_metrics"]
    assert set(body["excluded_demographic_proxy_columns"]) == {
        "age_band", "gender", "first_gen_flag", "distance_from_campus_band"
    }


@pytest.mark.skipif(not MODEL_EXISTS, reason="No trained model; run `python -m student_journey.models.train`.")
def test_predict_happy_path(client):
    r = client.post("/predict", json=VALID_RECORD)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["persistence_probability"] <= 1.0
    assert body["risk_probability"] == pytest.approx(1.0 - body["persistence_probability"], abs=1e-9)
    assert body["risk_category"] in ("low", "medium", "high")
    assert 1 <= len(body["top_contributing_factors"]) <= 5
    assert "not automated decision" in body["disclaimer"] or "not, and must not be used as" in body["disclaimer"]


@pytest.mark.skipif(not MODEL_EXISTS, reason="No trained model; run `python -m student_journey.models.train`.")
def test_predict_rejects_invalid_program(client):
    bad = dict(VALID_RECORD, program="Not A Real Program")
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


@pytest.mark.skipif(not MODEL_EXISTS, reason="No trained model; run `python -m student_journey.models.train`.")
def test_predict_rejects_out_of_range_gpa(client):
    bad = dict(VALID_RECORD, term_gpa=9.0)
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


@pytest.mark.skipif(not MODEL_EXISTS, reason="No trained model; run `python -m student_journey.models.train`.")
def test_batch_predict_echoes_record_ids(client):
    records = [dict(VALID_RECORD, record_id="s1"), dict(VALID_RECORD, record_id="s2", term_gpa=3.8)]
    r = client.post("/batch_predict", json={"records": records})
    assert r.status_code == 200
    body = r.json()
    assert [p["record_id"] for p in body["predictions"]] == ["s1", "s2"]
    # Higher GPA should predict a lower risk probability, all else equal.
    assert body["predictions"][1]["risk_probability"] < body["predictions"][0]["risk_probability"]


def test_service_degrades_gracefully_without_a_model(tmp_path, monkeypatch):
    """Simulate no trained model at all: /health stays 200 (liveness), but
    endpoints that need the model return 503 with an actionable message."""
    import importlib

    import student_journey.api.model_loader as model_loader_module

    def raise_not_found(*args, **kwargs):
        raise FileNotFoundError("simulated missing model for this test")

    monkeypatch.setattr(model_loader_module, "PersistenceModel", raise_not_found)
    import student_journey.api.main as main_module
    importlib.reload(main_module)

    with TestClient(main_module.app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is False

        r = client.get("/model_info")
        assert r.status_code == 503

        r = client.post("/predict", json=VALID_RECORD)
        assert r.status_code == 503

    # Undo the monkeypatch before reloading, so the restored main_module
    # picks back up the real PersistenceModel — otherwise this test would
    # leave a permanently broken `app` behind for any test that runs after it.
    monkeypatch.undo()
    importlib.reload(main_module)
