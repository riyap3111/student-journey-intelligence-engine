#!/bin/sh
# Single entrypoint for both services this image can run, switched by the
# SERVICE_TYPE env var (api|dashboard). Always respects $PORT rather than a
# hardcoded port — required by Cloud Run, which injects PORT and expects the
# container to listen on it; defaults to 8000 for local Docker Compose use,
# where PORT isn't set.
#
# If the container is started with an explicit command (e.g.
# `docker compose run --rm api bash scripts/run_pipeline.sh`), run THAT
# instead of the default service — standard entrypoint-script convention.
# Without this, ENTRYPOINT (unlike CMD) doesn't get replaced by a `run`
# command, it gets that command appended as arguments the script would
# otherwise silently ignore.
set -e

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

PORT="${PORT:-8000}"
SERVICE_TYPE="${SERVICE_TYPE:-api}"

if [ "$SERVICE_TYPE" = "dashboard" ]; then
    exec streamlit run src/student_journey/dashboard/app.py \
        --server.address 0.0.0.0 --server.port "$PORT" --server.headless true
else
    exec uvicorn student_journey.api.main:app --host 0.0.0.0 --port "$PORT"
fi
