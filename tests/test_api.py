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
REFERENCE_EXISTS = (MODELS_DIR / "reference_distribution.json").exists()

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
    assert body["model_type"] in ("logistic_regression", "random_forest", "xgboost", "ensemble")
    assert "test_metrics" in body and "roc_auc" in body["test_metrics"]
    assert set(body["excluded_demographic_proxy_columns"]) == {
        "age_band", "gender", "first_gen_flag", "distance_from_campus_band"
    }
    # Model-comparison fields (the frontend's Model Info page relies on these to show
    # every candidate's validation metrics, not just the winner's).
    assert body["model_type"] in body["validation_metrics_by_model"]
    assert "roc_auc" in body["validation_metrics_by_model"][body["model_type"]]
    assert "brier_score" in body["test_metrics_uncalibrated"]


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


@pytest.mark.skipif(not MODEL_EXISTS, reason="No trained model; run `python -m student_journey.models.train`.")
def test_predict_requires_api_key_when_configured(client, monkeypatch):
    """When STUDENT_JOURNEY_API_KEY is set, /predict requires a matching
    X-API-Key header; when unset (this project's default), no header is
    required — see security.py's rationale. Only security.py needs
    reloading: main.py's route already holds a reference to
    require_api_key, and that function looks up EXPECTED_API_KEY from its
    module's __dict__ at call time, which reload mutates in place."""
    import importlib

    import student_journey.api.security as security_module

    monkeypatch.setenv("STUDENT_JOURNEY_API_KEY", "test-secret-key")
    importlib.reload(security_module)
    try:
        r = client.post("/predict", json=VALID_RECORD)
        assert r.status_code == 401

        r = client.post("/predict", json=VALID_RECORD, headers={"X-API-Key": "wrong-key"})
        assert r.status_code == 401

        r = client.post("/predict", json=VALID_RECORD, headers={"X-API-Key": "test-secret-key"})
        assert r.status_code == 200
    finally:
        monkeypatch.undo()
        importlib.reload(security_module)


@pytest.mark.skipif(not MODEL_EXISTS, reason="No trained model; run `python -m student_journey.models.train`.")
def test_metrics_endpoint_exposes_prometheus_format(client):
    client.get("/health")  # generate at least one data point first
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "# HELP" in r.text or "# TYPE" in r.text


@pytest.mark.skipif(
    not MODEL_EXISTS or not REFERENCE_EXISTS,
    reason="No trained model/reference distribution; run `python -m student_journey.models.train`.",
)
def test_drift_endpoint_returns_a_well_formed_report(client):
    client.post("/predict", json=VALID_RECORD)  # ensure at least one row is logged to compare against
    r = client.get("/monitoring/drift")
    assert r.status_code == 200
    body = r.json()
    valid_statuses = {"stable", "moderate_shift", "significant_shift", "insufficient_data"}
    assert body["overall_status"] in valid_statuses
    assert body["n_current_rows"] >= 1
    assert len(body["features"]) > 0
    assert all(f["status"] in valid_statuses for f in body["features"])


@pytest.mark.skipif(not MODEL_EXISTS, reason="No trained model; run `python -m student_journey.models.train`.")
def test_drift_endpoint_503_when_no_reference_distribution(client, monkeypatch):
    import student_journey.api.main as main_module

    monkeypatch.setattr(main_module.drift, "load_reference_distribution", lambda models_dir: None)
    r = client.get("/monitoring/drift")
    assert r.status_code == 503


def test_rate_limiting_mechanism_returns_429_when_exceeded():
    """Verifies the slowapi wiring pattern used in main.py actually enforces
    a limit and returns 429 — tested against an isolated minimal app with a
    trivial body and a very low limit, rather than main.py's real endpoints,
    since hitting the real (SHAP-heavy) /predict enough times to exceed its
    60/minute limit would make this test extremely slow."""
    from fastapi import FastAPI, Request
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    mini_app = FastAPI()
    mini_limiter = Limiter(key_func=get_remote_address)
    mini_app.state.limiter = mini_limiter
    mini_app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @mini_app.get("/ping")
    @mini_limiter.limit("3/minute")
    def ping(request: Request):
        return {"ok": True}

    with TestClient(mini_app) as mini_client:
        for _ in range(3):
            r = mini_client.get("/ping")
            assert r.status_code == 200
        r = mini_client.get("/ping")
        assert r.status_code == 429
