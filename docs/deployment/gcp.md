# Deploying to Google Cloud (Cloud Run + Cloud SQL + Cloud Storage)

> **This deployment path was written and reviewed, but not executed** — there was no
> GCP project or credentials available in the environment this project was developed
> in. Treat everything below as a carefully-written starting point to run and debug
> yourself against a real project, not a proven-working deploy button. If something
> here is wrong, it's a documentation bug — the code it deploys (the API, dashboard,
> `db.py`, `cloud/storage.py`) is exercised by the test suite and local Docker
> Compose, which *were* actually run.

## What this deploys

- Two Cloud Run services from one container image (`scripts/docker-entrypoint.sh`
  picks API vs. dashboard via the `SERVICE_TYPE` env var, and always respects the
  `PORT` env var Cloud Run injects).
- A Cloud SQL (PostgreSQL) instance as the database, in place of local SQLite —
  the application code doesn't change; `DATABASE_URL` does (`db.py`).
- A Cloud Storage bucket holding the trained model artifacts, since Cloud Run's
  filesystem is ephemeral — a model trained in one container instance needs
  somewhere durable to land for the next one to find it (`cloud/storage.py`).

## One-time setup

Everything in this section is run once per GCP project, before the first deploy.

```bash
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-central1

gcloud config set project "$GCP_PROJECT_ID"

# 1. Enable the APIs this deployment uses.
gcloud services enable \
  run.googleapis.com \
  sqladmin.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

# 2. An Artifact Registry repo to hold the built image.
gcloud artifacts repositories create student-journey \
  --repository-format=docker --location="$GCP_REGION" \
  --description="Student Journey Intelligence Engine images"

# 3. A Cloud SQL for PostgreSQL instance. db-f1-micro is the smallest/cheapest
#    tier, fine for a portfolio deployment; size up for anything real.
gcloud sql instances create student-journey-db \
  --database-version=POSTGRES_16 --tier=db-f1-micro --region="$GCP_REGION"

export DB_PASSWORD="$(openssl rand -base64 24)"   # save this — you need it again below
gcloud sql users create student_journey \
  --instance=student-journey-db --password="$DB_PASSWORD"
gcloud sql databases create student_journey --instance=student-journey-db

export CLOUDSQL_INSTANCE_CONNECTION_NAME="$(gcloud sql instances describe student-journey-db --format='value(connectionName)')"
echo "CLOUDSQL_INSTANCE_CONNECTION_NAME=$CLOUDSQL_INSTANCE_CONNECTION_NAME"

# 4. A GCS bucket for model artifacts. Bucket names are globally unique.
export GCS_BUCKET_NAME="${GCP_PROJECT_ID}-student-journey-models"
gcloud storage buckets create "gs://${GCS_BUCKET_NAME}" --location="$GCP_REGION"
```

**IAM:** the identity Cloud Run runs as (by default, the project's Compute Engine
default service account, unless you've set up a dedicated one — recommended for
anything beyond a portfolio demo) needs:
- `roles/cloudsql.client` on the project, to reach the Cloud SQL instance through the
  Cloud SQL Proxy Cloud Run attaches automatically via `--add-cloudsql-instances`.
- `roles/storage.objectAdmin` on the GCS bucket (or the project), to read/write model
  artifacts.

```bash
export RUNTIME_SA="$(gcloud iam service-accounts list --filter='displayName:Compute Engine default service account' --format='value(email)')"

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" --role="roles/cloudsql.client"

gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET_NAME}" \
  --member="serviceAccount:${RUNTIME_SA}" --role="roles/storage.objectAdmin"
```

## Produce a trained model before the first deploy

A freshly deployed API/dashboard has no model until one is trained *against the same
Cloud SQL database and GCS bucket the deployed services will use*. The simplest way,
without setting up a Cloud Run Job: run the pipeline locally, pointed at the cloud
resources via the [Cloud SQL Auth Proxy](https://cloud.google.com/sql/docs/postgres/sql-proxy):

```bash
# In one terminal: start the proxy (downloads a small binary the first time)
cloud-sql-proxy "$CLOUDSQL_INSTANCE_CONNECTION_NAME" --port 5432

# In another terminal, pointed at the proxy and the real bucket:
export DATABASE_URL="postgresql+psycopg2://student_journey:${DB_PASSWORD}@localhost:5432/student_journey"
export GCS_BUCKET_NAME
source .venv/bin/activate
export PYTHONPATH=src
bash scripts/run_pipeline.sh   # generates data, trains, evaluates — ends with the model uploaded to GCS
```

A more production-appropriate approach — a scheduled Cloud Run Job re-running
`scripts/run_pipeline.sh` on a cadence — is a natural next step but isn't built here.

## Deploy

```bash
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-central1                              # optional, this is the default
export CLOUDSQL_INSTANCE_CONNECTION_NAME=...                # from setup step 3
export DB_PASSWORD=...                                      # from setup step 3
export GCS_BUCKET_NAME=...                                  # from setup step 4
export STUDENT_JOURNEY_API_KEY="$(openssl rand -base64 32)" # optional but recommended — see security.py

bash scripts/deploy_gcp.sh
```

This builds the image with Cloud Build, pushes it to Artifact Registry, and deploys
both Cloud Run services. It prints each service's URL at the end.

## Verify

```bash
curl -s "$(gcloud run services describe student-journey-api --region "$GCP_REGION" --format='value(status.url)')/health"
```
Should return `{"status": "ok", "model_loaded": true, ...}`. If `model_loaded` is
`false`, the model-training step above hasn't landed a model in GCS yet, or the
service's GCS/Cloud SQL access isn't wired up correctly — check Cloud Run's logs
(`gcloud run services logs read student-journey-api --region "$GCP_REGION"`) for the
`[startup warning]` line `main.py` prints when it can't load a model.

## Cost note

Cloud Run scales to zero when idle (you only pay for actual request time), but Cloud
SQL does **not** — the `db-f1-micro` instance from setup step 3 runs (and costs
money) continuously until you delete it. For a portfolio demo you don't need running
indefinitely:
```bash
gcloud sql instances delete student-journey-db
gcloud run services delete student-journey-api --region "$GCP_REGION"
gcloud run services delete student-journey-dashboard --region "$GCP_REGION"
gcloud storage rm -r "gs://${GCS_BUCKET_NAME}"
```

## What's deliberately not built here

- A CI/CD pipeline that deploys automatically on push (the existing
  `.github/workflows/ci.yml` runs tests, not deployment — wiring `deploy_gcp.sh` into
  a workflow triggered on merge to `main` is a natural extension, not built).
- A scheduled retraining job (Cloud Run Jobs + Cloud Scheduler, or Cloud Composer for
  something more elaborate).
- Secret Manager for `DB_PASSWORD`/`STUDENT_JOURNEY_API_KEY` instead of plain
  `--set-env-vars` — fine for a portfolio demo, not how you'd want to handle real
  credentials in production.
- A load balancer / custom domain / Cloud CDN in front of the two Cloud Run services.
