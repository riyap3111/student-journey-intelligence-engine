"""API key authentication for the prediction endpoints.

Reads the expected key from the STUDENT_JOURNEY_API_KEY environment
variable at import time. If it's unset, auth is disabled entirely — every
request is allowed through — which is the right default for local
development and for this portfolio project's demo deployment, but NOT for
a real production deployment with real data, where the environment
variable must be set. This is stated again in the README/Dockerfile.
"""
from __future__ import annotations

import os

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

API_KEY_HEADER_NAME = "X-API-Key"
_api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)

EXPECTED_API_KEY = os.environ.get("STUDENT_JOURNEY_API_KEY")


def require_api_key(provided_key: str = Depends(_api_key_header)) -> None:
    if EXPECTED_API_KEY is None:
        return  # auth disabled — no STUDENT_JOURNEY_API_KEY set in this environment
    if provided_key != EXPECTED_API_KEY:
        raise HTTPException(
            status_code=401,
            detail=f"Missing or invalid API key. Set the '{API_KEY_HEADER_NAME}' header.",
        )
