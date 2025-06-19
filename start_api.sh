#!/bin/bash

# Exit on error
set -e

# Set experiment path only if not already set
if [ -z "$EXPERIMENT_PATH" ]; then
    export EXPERIMENT_PATH=runs_share/Pix2Poly_inria_coco_224
fi

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pix2poly

# Start the API server
echo "Starting API server with experiment path: $EXPERIMENT_PATH"
uvicorn api:app --reload --port 8080 --workers 1
