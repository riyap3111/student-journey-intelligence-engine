# Single image used for both the API and the dashboard (docker-compose.yml
# runs two containers from it with different commands). Linux base, so
# XGBoost's native library works out of the box (unlike some local macOS
# dev setups without Homebrew's libomp — see README/train.py).
FROM python:3.11-slim

WORKDIR /app

# libgomp1: OpenMP runtime required by XGBoost at import time on Linux.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x scripts/docker-entrypoint.sh

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

EXPOSE 8000 8501

# Which service to run (api|dashboard) and which port to listen on are both
# set via env vars (SERVICE_TYPE, PORT) at container-start time — see
# scripts/docker-entrypoint.sh. This is what makes the same image work for
# docker-compose (PORT unset, defaults to 8000) and Cloud Run (PORT injected
# by the platform, must be respected — Cloud Run refuses traffic otherwise).
ENTRYPOINT ["scripts/docker-entrypoint.sh"]
