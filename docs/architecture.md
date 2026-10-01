# Architecture

```mermaid
flowchart TB
    subgraph DataLayer["Data Layer"]
        A[Synthetic data generator - incl. gap terms] --> B[(DB via SQLAlchemy: SQLite or Cloud SQL)]
        B --> C[SQL validation and transform queries]
        C --> D[Feature engineering - pandas]
        D --> E[(Feature table)]
    end

    subgraph MLLayer["ML Layer"]
        E --> F[Time-aware train/val/test split]
        F --> F2[Optuna tuning per model type]
        F2 --> G[Logistic Regression]
        F2 --> H[Random Forest]
        F2 --> I[XGBoost]
        G --> G2[Soft-voting ensemble]
        H --> G2
        I --> G2
        G --> J[MLflow tracking]
        H --> J
        I --> J
        G2 --> J
        J --> K2[Best candidate by validation ROC-AUC]
        K2 --> K3[CalibratedClassifierCV]
        K3 --> K[Final pipeline .joblib + reference_distribution.json]
        K --> L[SHAP explainer - matched to model type]
        K -.-> GCS[(Cloud Storage: model artifacts)]
    end

    subgraph ServingLayer["Serving Layer"]
        K --> M["FastAPI: /health /predict /batch_predict /model_info /monitoring/drift /metrics"]
        L --> M
        GCS -.-> M
        M --> M2[API key auth + rate limiting + access logging]
        M --> M3[(prediction_log table)]
        M3 --> M4[PSI drift report vs. reference_distribution.json]
    end

    subgraph PresentationLayer["Presentation Layer"]
        M --> N[Streamlit dashboard - 9 tabs incl. Intervention Impact + live drift monitoring]
        E --> N
        J --> N
        M4 --> N
        M -->|HTTP, CORS| N2["React frontend (Vite + TS + Tailwind): Overview, Predict, Model Info, Monitoring"]
    end

    subgraph Infra["Infra"]
        O[Docker: single image, SERVICE_TYPE switches api/dashboard]
        O2["Docker: frontend image (nginx), API_BASE_URL injected at container startup"]
        P[GitHub Actions: lint + tests + frontend build]
        Q["GitHub Actions: deploy job (opt-in) -> Cloud Run"]
        R[Cloud Run: api + dashboard + frontend services]
        O --> R
        O2 --> R
        P --> Q
    end
```

## Component notes

- **Synthetic data generator** produces multi-term student-enrollment records with
  realistic, documented distributions, including stop-out-and-return (gap-term)
  behavior (see `data/DATA_DICTIONARY.md`). This is the only dataset used for
  training — a UCI-dataset benchmark was considered during planning but never built
  (see the README's Data section for that honest correction).
- **SQLite by default, PostgreSQL/Cloud SQL when `DATABASE_URL` is set** — a single
  SQLAlchemy engine (`db.py`) every script uses, so the database backend is a
  connection-string change, not a fork in the code. The schema is written in plain SQL
  (`db/schema.sql`).
- **Optuna** tunes each model type's hyperparameters (maximizing validation ROC-AUC)
  before comparison; a soft-voting ensemble over the tuned models is evaluated as one
  more candidate and wins the final selection only if it genuinely beats every
  individual model.
- **MLflow** tracks every tuning trial's parameters, metrics, and artifacts so no
  reported metric is hand-typed — it's queried from actual run logs.
- **CalibratedClassifierCV** corrects a measured probability-calibration distortion
  from class weighting, fit on the validation set only, before the final test
  evaluation.
- **FastAPI** loads the serialized model + SHAP explainer once at startup and serves
  predictions plus explanations, behind optional API-key auth, per-endpoint rate
  limits, structured access logging, and a Prometheus `/metrics` endpoint.
- **Streamlit** dashboard is a separate process that reuses the same model/explainer
  code directly (not over HTTP) for exploration, including a sensitivity-analysis tab
  for simulated intervention impact.
- **Monitoring**: every scored request is best-effort logged to a `prediction_log`
  table; `/monitoring/drift` and the dashboard's Monitoring tab compare recently
  logged requests against `reference_distribution.json` (saved by `train.py`) using
  the Population Stability Index, falling back to an illustrative train/test
  comparison until enough live traffic has been logged.
- **Cloud Storage** persists model artifacts (including the drift reference) since
  Cloud Run's container filesystem is ephemeral — optional, gracefully disabled when
  `GCS_BUCKET_NAME` is unset.
- **CI/CD**: GitHub Actions runs lint + the full test suite (and, in a separate job,
  the frontend's lint/test/build) on every push/PR; a third, opt-in job redeploys all
  three Cloud Run services automatically after tests pass on a push to `main`.
- **React frontend** (`frontend/`) is a separate SPA (Vite + TypeScript + Tailwind)
  calling the FastAPI backend over HTTP/CORS — a product-style UI alongside the
  Streamlit dashboard, not a replacement for it. Gets the API's base URL from a
  runtime-injected `window.__ENV__` (written by its own `docker-entrypoint.sh` from an
  `API_BASE_URL` env var at container startup) rather than a Vite build-time variable,
  so the same built image works against any backend without rebuilding — necessary
  because Cloud Run doesn't assign the API's URL until after it's deployed.
