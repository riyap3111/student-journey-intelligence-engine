"""Pydantic request/response models for the prediction API.

Design note on /predict's input shape: this endpoint accepts an already
ENGINEERED student-term feature record (the same fields build_features.py
produces — GPA, cumulative credits, momentum, etc.), not raw multi-term
enrollment history. Computing cumulative/trend features requires a
student's full history, which belongs in the feature-engineering pipeline
(Phase 2), not duplicated here. A realistic deployment has an upstream
service or scheduled job call build_features.py and feed its output to this
scoring API — a standard separation between feature engineering and model
serving. `enrollment_intensity` is the one exception kept human-friendly
(full_time/part_time string) and converted to the model's numeric feature
internally, since asking a caller to pre-compute a 0/1 flag for a plain
category would be needless friction.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from student_journey.data.generate_synthetic_data import PROGRAMS
from student_journey.explainability.shap_utils import EXPLANATION_DISCLAIMER

API_DISCLAIMER = (
    f"{EXPLANATION_DISCLAIMER} This prediction is intended for planning and "
    "advising support only — it is not, and must not be used as, an automated "
    "decision about any student."
)


class StudentTermFeatures(BaseModel):
    """One student-term's engineered features — the model's actual input."""

    term_number: int = Field(..., ge=1, description="Student's Nth term of enrollment")
    enrollment_intensity: Literal["full_time", "part_time"]
    credits_attempted: int = Field(..., ge=0, le=30)
    credits_completed: int = Field(..., ge=0, le=30)
    credit_completion_rate: float = Field(..., ge=0.0, le=1.0)
    cumulative_credits_attempted: int = Field(..., ge=0)
    cumulative_credits_completed: int = Field(..., ge=0)
    cumulative_credit_completion_rate: float = Field(..., ge=0.0, le=1.0)
    term_gpa: float = Field(..., ge=0.0, le=4.0)
    cumulative_gpa: float = Field(..., ge=0.0, le=4.0)
    gpa_change: float = Field(..., ge=-4.0, le=4.0)
    academic_momentum: float = Field(..., ge=-4.0, le=4.0)
    courses_withdrawn: int = Field(..., ge=0, le=20)
    cumulative_withdrawals: int = Field(..., ge=0)
    courses_repeated: int = Field(..., ge=0, le=20)
    cumulative_repeats: int = Field(..., ge=0)
    advising_contact_flag: Literal[0, 1]
    financial_aid_flag: Literal[0, 1]
    prior_term_enrolled_flag: Literal[0, 1]
    program: Literal[tuple(PROGRAMS)]
    entry_type: Literal["first_time", "transfer"]

    def to_feature_dict(self) -> dict:
        data = self.model_dump()
        data["enrollment_intensity_numeric"] = 1 if data.pop("enrollment_intensity") == "full_time" else 0
        return data

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class ContributingFactor(BaseModel):
    feature: str
    contribution: float = Field(
        ..., description="Signed contribution on the model's own additive score — see score_scale for units"
    )
    direction: Literal["increases_persistence_likelihood", "increases_risk"]


class PredictionResponse(BaseModel):
    persistence_probability: float = Field(..., description="Model's estimated probability the student persists")
    risk_probability: float = Field(..., description="1 - persistence_probability")
    risk_category: Literal["low", "medium", "high"]
    score_scale: Literal["log_odds", "probability"] = Field(
        ...,
        description=(
            "Units of top_contributing_factors' contribution values. Detected per model type, not assumed — "
            "log_odds for the linear model, or probability for some tree-ensemble explainers."
        ),
    )
    top_contributing_factors: list[ContributingFactor]
    model_version: str
    disclaimer: str = API_DISCLAIMER


class BatchPredictionItem(PredictionResponse):
    record_id: Optional[str] = Field(None, description="Caller-supplied identifier, echoed back for matching")


class StudentTermFeaturesWithId(StudentTermFeatures):
    record_id: Optional[str] = Field(None, description="Optional caller-supplied identifier (e.g. student_id)")


class BatchPredictRequest(BaseModel):
    records: list[StudentTermFeaturesWithId] = Field(..., min_length=1, max_length=500)


class BatchPredictResponse(BaseModel):
    predictions: list[BatchPredictionItem]
    disclaimer: str = API_DISCLAIMER


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_version: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_version: str
    model_type: str
    trained_at_utc: str
    feature_columns: list[str]
    excluded_demographic_proxy_columns: list[str]
    risk_thresholds: dict
    test_metrics: dict
    disclaimer: str = API_DISCLAIMER
