#!/bin/bash

# Exit on error
set -e

# Start the API server
echo "Starting API server"
uvicorn api:app --host 0.0.0.0 --port 8080 --workers 1 --backlog 10
