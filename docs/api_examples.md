# Example API Requests

Start the service first:
```bash
export PYTHONPATH=src
uvicorn student_journey.api.main:app --reload --port 8000
```
Interactive docs (try requests in the browser): http://localhost:8000/docs

## GET /health

```bash
curl -s http://localhost:8000/health
```
```json
{"status": "ok", "model_loaded": true, "model_version": "20260924-065958"}
```
Always returns `200`, even with no trained model (`model_loaded` becomes `false`) — meant for liveness/readiness checks, not "is a prediction possible."

## GET /model_info

```bash
curl -s http://localhost:8000/model_info | python3 -m json.tool
```
Returns model type/version, the exact feature columns used, the demographic proxy columns deliberately excluded from prediction, the calibrated risk thresholds, and the full held-out test metric suite — all read live from `models/model_metadata.json`, never hardcoded.

## POST /predict

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "entry_type": "first_time"
  }' | python3 -m json.tool
```
```json
{
  "persistence_probability": 0.3385,
  "risk_probability": 0.6615,
  "risk_category": "medium",
  "top_contributing_factors": [
    {"feature": "term_gpa", "contribution": -0.7221, "direction": "increases_risk"},
    {"feature": "advising_contact_flag", "contribution": -0.3303, "direction": "increases_risk"},
    {"feature": "courses_withdrawn", "contribution": -0.2975, "direction": "increases_risk"},
    {"feature": "financial_aid_flag", "contribution": 0.1917, "direction": "increases_persistence_likelihood"},
    {"feature": "credits_attempted", "contribution": 0.1638, "direction": "increases_persistence_likelihood"}
  ],
  "model_version": "20260924-065958",
  "disclaimer": "..."
}
```
This is real output from an actual request against a running server — see the Phase 5 section of the main README.

**Invalid input** (unknown `program`, or `term_gpa` outside `[0, 4]`) returns `422` with a field-level Pydantic error, e.g.:
```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"program": "Not A Real Program", ...}'
# 422
```

## POST /batch_predict

```bash
curl -s -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "record_id": "example-1",
        "term_number": 3, "enrollment_intensity": "full_time",
        "credits_attempted": 15, "credits_completed": 12, "credit_completion_rate": 0.8,
        "cumulative_credits_attempted": 42, "cumulative_credits_completed": 34,
        "cumulative_credit_completion_rate": 0.81,
        "term_gpa": 2.1, "cumulative_gpa": 2.4, "gpa_change": -0.3, "academic_momentum": -0.25,
        "courses_withdrawn": 1, "cumulative_withdrawals": 2, "courses_repeated": 0, "cumulative_repeats": 1,
        "advising_contact_flag": 0, "financial_aid_flag": 1, "prior_term_enrolled_flag": 1,
        "program": "Business", "entry_type": "first_time"
      },
      {
        "record_id": "example-2",
        "term_number": 3, "enrollment_intensity": "full_time",
        "credits_attempted": 15, "credits_completed": 15, "credit_completion_rate": 1.0,
        "cumulative_credits_attempted": 42, "cumulative_credits_completed": 42,
        "cumulative_credit_completion_rate": 1.0,
        "term_gpa": 3.8, "cumulative_gpa": 3.7, "gpa_change": 0.2, "academic_momentum": 0.2,
        "courses_withdrawn": 0, "cumulative_withdrawals": 0, "courses_repeated": 0, "cumulative_repeats": 0,
        "advising_contact_flag": 1, "financial_aid_flag": 0, "prior_term_enrolled_flag": 1,
        "program": "Computer Science", "entry_type": "first_time"
      }
    ]
  }' | python3 -m json.tool
```
Each item in `predictions` echoes back its `record_id` for matching against the request, and the higher-GPA example ("example-2") predicts a lower risk probability than the first — a basic sanity property asserted directly in `tests/test_api.py::test_batch_predict_echoes_record_ids`.

## Python (requests)

```python
import requests

BASE_URL = "http://localhost:8000"

record = {
    "term_number": 3, "enrollment_intensity": "full_time",
    "credits_attempted": 15, "credits_completed": 12, "credit_completion_rate": 0.8,
    "cumulative_credits_attempted": 42, "cumulative_credits_completed": 34,
    "cumulative_credit_completion_rate": 0.81,
    "term_gpa": 2.1, "cumulative_gpa": 2.4, "gpa_change": -0.3, "academic_momentum": -0.25,
    "courses_withdrawn": 1, "cumulative_withdrawals": 2, "courses_repeated": 0, "cumulative_repeats": 1,
    "advising_contact_flag": 0, "financial_aid_flag": 1, "prior_term_enrolled_flag": 1,
    "program": "Business", "entry_type": "first_time",
}

response = requests.post(f"{BASE_URL}/predict", json=record, timeout=10)
response.raise_for_status()
result = response.json()
print(f"Risk category: {result['risk_category']} ({result['risk_probability']:.1%})")
for factor in result["top_contributing_factors"]:
    print(f"  {factor['feature']}: {factor['contribution']:+.3f} ({factor['direction']})")
```
