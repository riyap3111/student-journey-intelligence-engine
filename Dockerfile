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

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

EXPOSE 8000 8501

# Default: serve the API. docker-compose overrides this for the dashboard service.
CMD ["uvicorn", "student_journey.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
