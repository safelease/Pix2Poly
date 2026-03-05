#!/bin/bash
set -e

# Start RunPod base services (Jupyter, SSH) — uses system Python with jupyter
echo "Starting RunPod services (Jupyter, SSH)..."
/start.sh &

sleep 3

# Start API with venv PATH only for this process
echo "Starting Pix2Poly API server on :8080..."
env PATH="/opt/program/.venv/bin:$PATH" \
  uv run uvicorn api:app --host 0.0.0.0 --port 8080 --workers 1 --backlog 10 &

wait
