#!/usr/bin/env bash
# Deploys the Student Journey Intelligence Engine to Google Cloud Run,
# backed by Cloud SQL (PostgreSQL) and Cloud Storage (model artifacts).
#
# NOT EXECUTED OR VERIFIED against a real GCP project — there was no GCP
# project or credentials available in the environment this project was
# developed in. This script is written against real gcloud syntax and
# reviewed carefully, but treat it as a strong starting point to run and
# debug yourself, not a proven-working deploy button. See
# docs/deployment/gcp.md for the full one-time setup this script assumes
# already exists (APIs enabled, Cloud SQL instance, GCS bucket, Artifact
# Registry repo, IAM bindings) and for troubleshooting notes.
#
# Usage:
#   export GCP_PROJECT_ID=my-project
#   export GCP_REGION=us-central1                    # optional, defaults below
#   export CLOUDSQL_INSTANCE_CONNECTION_NAME=my-project:us-central1:student-journey-db
#   export DB_PASSWORD=...                            # the student_journey Cloud SQL user's password
#   export GCS_BUCKET_NAME=my-project-student-journey-models
#   export STUDENT_JOURNEY_API_KEY=...                # optional but strongly recommended for a public API
#   bash scripts/deploy_gcp.sh
set -euo pipefail

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
: "${CLOUDSQL_INSTANCE_CONNECTION_NAME:?Set CLOUDSQL_INSTANCE_CONNECTION_NAME (PROJECT:REGION:INSTANCE)}"
: "${DB_PASSWORD:?Set DB_PASSWORD}"
: "${GCS_BUCKET_NAME:?Set GCS_BUCKET_NAME}"

GCP_REGION="${GCP_REGION:-us-central1}"
REPO_NAME="${REPO_NAME:-student-journey}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}"
IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${REPO_NAME}/app:${IMAGE_TAG}"
DB_NAME="${DB_NAME:-student_journey}"
DB_USER="${DB_USER:-student_journey}"
API_SERVICE_NAME="${API_SERVICE_NAME:-student-journey-api}"
DASHBOARD_SERVICE_NAME="${DASHBOARD_SERVICE_NAME:-student-journey-dashboard}"
FRONTEND_SERVICE_NAME="${FRONTEND_SERVICE_NAME:-student-journey-frontend}"
FRONTEND_IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${REPO_NAME}/frontend:${IMAGE_TAG}"

echo "== Building and pushing image via Cloud Build =="
gcloud builds submit --project "$GCP_PROJECT_ID" --tag "$IMAGE_URI" .

# Built once here, deployed with whatever API_BASE_URL is current at deploy
# time below — this image never needs rebuilding just because the API's URL
# changes (see frontend/docker-entrypoint.sh).
echo "== Building and pushing frontend image via Cloud Build =="
gcloud builds submit --project "$GCP_PROJECT_ID" --tag "$FRONTEND_IMAGE_URI" ./frontend

# The Unix-socket form Cloud Run uses to reach an attached Cloud SQL
# instance — not a TCP host:port, since Cloud Run mounts the Cloud SQL
# Proxy's socket at this fixed path when --add-cloudsql-instances is set.
DATABASE_URL="postgresql+psycopg2://${DB_USER}:${DB_PASSWORD}@/${DB_NAME}?host=/cloudsql/${CLOUDSQL_INSTANCE_CONNECTION_NAME}"

COMMON_FLAGS=(
  --project "$GCP_PROJECT_ID"
  --region "$GCP_REGION"
  --image "$IMAGE_URI"
  --add-cloudsql-instances "$CLOUDSQL_INSTANCE_CONNECTION_NAME"
  --set-env-vars "DATABASE_URL=${DATABASE_URL},GCS_BUCKET_NAME=${GCS_BUCKET_NAME}"
  --memory 1Gi
  --allow-unauthenticated  # the API has its own optional API-key layer (see security.py); tighten with
                           # --no-allow-unauthenticated + IAM if you don't want the service itself public
)

if [ -n "${STUDENT_JOURNEY_API_KEY:-}" ]; then
  COMMON_FLAGS+=(--set-env-vars "STUDENT_JOURNEY_API_KEY=${STUDENT_JOURNEY_API_KEY}")
fi

echo "== Deploying API service =="
gcloud run deploy "$API_SERVICE_NAME" \
  "${COMMON_FLAGS[@]}" \
  --set-env-vars "SERVICE_TYPE=api" \
  --port 8080

echo "== Deploying dashboard service =="
gcloud run deploy "$DASHBOARD_SERVICE_NAME" \
  "${COMMON_FLAGS[@]}" \
  --set-env-vars "SERVICE_TYPE=dashboard" \
  --port 8080

# The frontend needs the API's URL, which only exists after the deploy
# above — this ordering (not a build-time arg) is exactly why the frontend
# image takes its backend URL at container startup, not at build time.
API_URL="$(gcloud run services describe "$API_SERVICE_NAME" --project "$GCP_PROJECT_ID" --region "$GCP_REGION" --format='value(status.url)')"

echo "== Deploying frontend service =="
gcloud run deploy "$FRONTEND_SERVICE_NAME" \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION" \
  --image "$FRONTEND_IMAGE_URI" \
  --set-env-vars "API_BASE_URL=${API_URL}" \
  --memory 256Mi \
  --allow-unauthenticated \
  --port 80

echo
echo "Done. Note: a freshly deployed API/dashboard has no trained model yet —"
echo "run the pipeline once (e.g. via a Cloud Run Job, or locally with"
echo "DATABASE_URL/GCS_BUCKET_NAME pointed at the same Cloud SQL instance and"
echo "GCS bucket) so a model lands in Cloud Storage before serving traffic."
echo
echo "Also: the API's CORS_ALLOWED_ORIGINS env var defaults to local dev"
echo "origins only (see main.py) — set it to the frontend's URL below for the"
echo "deployed frontend to actually be able to call the deployed API."
echo
gcloud run services describe "$API_SERVICE_NAME" --project "$GCP_PROJECT_ID" --region "$GCP_REGION" --format="value(status.url)"
gcloud run services describe "$DASHBOARD_SERVICE_NAME" --project "$GCP_PROJECT_ID" --region "$GCP_REGION" --format="value(status.url)"
gcloud run services describe "$FRONTEND_SERVICE_NAME" --project "$GCP_PROJECT_ID" --region "$GCP_REGION" --format="value(status.url)"
