"""FastAPI service exposing the trained persistence model.

Endpoints:
    GET  /health          liveness/readiness check
    POST /predict          single student-term prediction + explanation
    POST /batch_predict     up to 500 records in one call
    GET  /model_info        model version, features, held-out test metrics
    GET  /metrics           Prometheus metrics (request counts, latencies)

The model and SHAP explainer are loaded once at startup (see model_loader.py)
and reused across requests. If loading fails (e.g. no trained model yet),
the service still starts so /health can report the problem, but /predict,
/batch_predict, and /model_info return 503 until a model is trained and the
service is restarted.

Production hardening (see security.py, middleware.py):
  - API key auth on /predict and /batch_predict via the X-API-Key header,
    controlled by the STUDENT_JOURNEY_API_KEY environment variable — unset
    means auth is OFF (local-dev default), set means it's enforced.
  - Rate limiting (slowapi): 60/minute on /predict, 20/minute on
    /batch_predict (heavier per-request compute), keyed by client IP.
  - Structured JSON access logging for every request (method, path, status,
    latency, client IP) to stdout.
  - Prometheus-compatible /metrics endpoint.

Run:
    uvicorn student_journey.api.main:app --reload --port 8000
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from student_journey.api.middleware import AccessLogMiddleware
from student_journey.api.model_loader import ModelBundle
from student_journey.api.schemas import (
    API_DISCLAIMER,
    BatchPredictionItem,
    BatchPredictRequest,
    BatchPredictResponse,
    ContributingFactor,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    StudentTermFeatures,
)
from student_journey.api.security import require_api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model_bundle = ModelBundle()
        app.state.load_error = None
    except FileNotFoundError as exc:
        print(f"[startup warning] Could not load model: {exc}", file=sys.stderr)
        app.state.model_bundle = None
        app.state.load_error = str(exc)
    yield


app = FastAPI(
    title="Student Journey Intelligence Engine API",
    description=(
        "Predicts whether a student is likely to continue enrollment next term, "
        "and explains why. Portfolio/educational project using synthetic data — "
        "see /model_info and the project's docs/model_card.md. " + API_DISCLAIMER
    ),
    version="0.1.0",
    lifespan=lifespan,
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(AccessLogMiddleware)

Instrumentator().instrument(app).expose(app)  # adds GET /metrics


def get_model_bundle() -> ModelBundle:
    bundle = app.state.model_bundle
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Run `python -m student_journey.models.train` "
                f"to produce one, then restart the service. ({app.state.load_error})"
            ),
        )
    return bundle


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    bundle = app.state.model_bundle
    return HealthResponse(
        status="ok",
        model_loaded=bundle is not None,
        model_version=bundle.model_version if bundle else None,
    )


@app.get("/model_info", response_model=ModelInfoResponse)
def model_info(bundle: ModelBundle = Depends(get_model_bundle)) -> ModelInfoResponse:
    metadata = bundle.model.metadata
    return ModelInfoResponse(
        model_version=metadata["model_version"],
        model_type=metadata["selected_model"],
        trained_at_utc=metadata["trained_at_utc"],
        feature_columns=metadata["feature_columns"],
        excluded_demographic_proxy_columns=metadata["excluded_demographic_proxy_columns"],
        risk_thresholds=metadata["risk_thresholds"],
        test_metrics=metadata["test_metrics"],
    )


def _score(record_dict: dict, bundle: ModelBundle) -> tuple:
    prediction = bundle.model.predict_one(record_dict)
    explanation = bundle.explainer.explain(record_dict)
    factors = [
        ContributingFactor(feature=f["feature"], contribution=f["contribution"], direction=f["direction"])
        for f in explanation["top_factors"]
    ]
    return prediction, factors, explanation["score_scale"]


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("60/minute")
def predict(
    request: Request, features: StudentTermFeatures, bundle: ModelBundle = Depends(get_model_bundle)
) -> PredictionResponse:
    record = features.to_feature_dict()
    prediction, factors, score_scale = _score(record, bundle)
    return PredictionResponse(
        persistence_probability=prediction.persistence_probability,
        risk_probability=prediction.risk_probability,
        risk_category=prediction.risk_category,
        score_scale=score_scale,
        top_contributing_factors=factors,
        model_version=bundle.model_version,
    )


@app.post("/batch_predict", response_model=BatchPredictResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("20/minute")
def batch_predict(
    request: Request, payload: BatchPredictRequest, bundle: ModelBundle = Depends(get_model_bundle)
) -> BatchPredictResponse:
    items = []
    for record_with_id in payload.records:
        record_id = record_with_id.record_id
        record = record_with_id.to_feature_dict()
        record.pop("record_id", None)
        prediction, factors, score_scale = _score(record, bundle)
        items.append(
            BatchPredictionItem(
                record_id=record_id,
                persistence_probability=prediction.persistence_probability,
                risk_probability=prediction.risk_probability,
                risk_category=prediction.risk_category,
                score_scale=score_scale,
                top_contributing_factors=factors,
                model_version=bundle.model_version,
            )
        )
    return BatchPredictResponse(predictions=items)
