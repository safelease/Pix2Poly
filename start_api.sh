#!/bin/bash

# Exit on error
set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pix2poly

# Start the API server
echo "Starting API server"
uvicorn api:app --port 8080 --workers 1 --backlog 10
