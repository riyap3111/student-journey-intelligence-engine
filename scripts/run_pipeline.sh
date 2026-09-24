#!/usr/bin/env bash
# Runs the full pipeline end to end: generate synthetic data -> ingest into
# SQLite -> validate -> build features -> train -> evaluate. Used for local
# first-time setup and inside the Docker image (see Dockerfile/README).
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"

echo "== Phase 2: data pipeline =="
python -m student_journey.data.generate_synthetic_data
python -m student_journey.data.ingest
python -m student_journey.data.validate
python -m student_journey.features.build_features

echo
echo "== Phase 3: training =="
python -m student_journey.models.train

echo
echo "== Phase 4: evaluation =="
python -m student_journey.models.evaluate

echo
echo "Pipeline complete. Model + evaluation artifacts are in models/."
echo "Start the API:       uvicorn student_journey.api.main:app --host 0.0.0.0 --port 8000"
echo "Start the dashboard: streamlit run src/student_journey/dashboard/app.py --server.address 0.0.0.0"
