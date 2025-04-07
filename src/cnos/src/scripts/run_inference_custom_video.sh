#!/bin/bash

# Add project root to PYTHONPATH
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Check if environment variables are set



python src/scripts/inference_custom_video.py \
    --template_dir "$CAD_PATH" \
    --rgb_path "$RGB_PATH" \
    --num_max_dets 3 \
    --conf_threshold 0.5 \
    --stability_score_thresh 0.7

echo "Processing complete. Results saved"
