"""Google Cloud Storage integration for model artifact persistence.

Optional and gracefully degrading, matching this project's established
pattern (XGBoost availability, API auth, etc.): if GCS_BUCKET_NAME is
unset, every function here is a no-op and the pipeline behaves exactly as
in every earlier phase — local files only. Set GCS_BUCKET_NAME (and have
valid GCP credentials available, e.g. via GOOGLE_APPLICATION_CREDENTIALS
locally or the attached service account on Cloud Run) to persist trained
models to Cloud Storage. This matters specifically for Cloud Run: its
filesystem is ephemeral, so a model trained in one container instance and
saved only to local disk would vanish on the next cold start or new
revision — GCS is what makes the model durable across deployments there.

This module was NOT exercised against a real GCS bucket in the environment
this project was developed in (no GCP project/credentials available there)
— the upload/download logic is covered by unit tests that mock the
`google.cloud.storage` client, verifying the calls made are correct, but
you should run `python -m student_journey.cloud.storage` against a real
bucket you own before relying on this in a real deployment.
"""
from __future__ import annotations

import os
from pathlib import Path

from google.cloud import storage

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
GCS_MODEL_PREFIX = os.environ.get("GCS_MODEL_PREFIX", "models")

MODEL_ARTIFACT_FILENAMES = ["model_pipeline.joblib", "model_metadata.json"]


def is_enabled() -> bool:
    return bool(GCS_BUCKET_NAME)


def _bucket():
    client = storage.Client()
    return client.bucket(GCS_BUCKET_NAME)


def upload_file(local_path: Path, blob_name: str) -> bool:
    """Returns True if a file was actually uploaded, False if GCS is
    disabled (a no-op, not an error)."""
    if not is_enabled():
        return False
    blob = _bucket().blob(f"{GCS_MODEL_PREFIX}/{blob_name}")
    blob.upload_from_filename(str(local_path))
    return True


def download_file(local_path: Path, blob_name: str) -> bool:
    """Returns True if a file was downloaded, False if GCS is disabled or
    the blob doesn't exist (never raises for either of those two cases)."""
    if not is_enabled():
        return False
    blob = _bucket().blob(f"{GCS_MODEL_PREFIX}/{blob_name}")
    if not blob.exists():
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    return True


def upload_model_artifacts(model_dir: Path) -> list[str]:
    """Uploads model_pipeline.joblib and model_metadata.json if present
    locally. Returns the filenames actually uploaded (empty if GCS is
    disabled or neither file exists yet)."""
    uploaded = []
    for filename in MODEL_ARTIFACT_FILENAMES:
        local_path = model_dir / filename
        if local_path.exists() and upload_file(local_path, filename):
            uploaded.append(filename)
    return uploaded


def download_model_artifacts(model_dir: Path) -> list[str]:
    """Downloads model_pipeline.joblib and model_metadata.json from GCS if
    they're not already present locally. Returns the filenames actually
    downloaded (empty if GCS is disabled, already present locally, or not
    yet uploaded)."""
    downloaded = []
    for filename in MODEL_ARTIFACT_FILENAMES:
        local_path = model_dir / filename
        if local_path.exists():
            continue
        if download_file(local_path, filename):
            downloaded.append(filename)
    return downloaded


if __name__ == "__main__":
    from student_journey.config import MODELS_DIR

    if not is_enabled():
        print("GCS_BUCKET_NAME is not set — nothing to do (this is the expected local-dev state).")
    else:
        print(f"Uploading model artifacts from {MODELS_DIR} to gs://{GCS_BUCKET_NAME}/{GCS_MODEL_PREFIX}/ ...")
        uploaded = upload_model_artifacts(MODELS_DIR)
        print(f"Uploaded: {uploaded}" if uploaded else "Nothing to upload (no local model artifacts found).")
