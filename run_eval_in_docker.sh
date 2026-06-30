#!/bin/bash
# Run BigCodeBench evaluation inside a Docker container.
#
# This isolates the execution of model-generated code from the host system.
#
# Usage:
#   ./run_eval_in_docker.sh
#   ./run_eval_in_docker.sh results/gpt-4.1-mini/vanilla/bigcodebench_predictions.json

set -e

IMAGE_NAME="baseline-eval"
DOCKERFILE="Dockerfile.eval"

# Build the image if it doesn't exist.
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "Building Docker image '$IMAGE_NAME'..."
    docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" .
fi

PREDICTIONS_PATH="${1:-}"

# Create a temporary empty .env to mask the host's real .env inside the container.
# The evaluator doesn't need API keys, so an empty file is fine.
ENV_MASK_DIR=$(mktemp -d)
touch "$ENV_MASK_DIR/.env"
trap 'rm -rf "$ENV_MASK_DIR"' EXIT

# Mount the whole project so the summary JSON is written back to the host.
# The real .env is hidden by mounting an empty file at the same path.
if [[ -n "$PREDICTIONS_PATH" ]]; then
    docker run --rm \
        -v "$(pwd):/workspace" \
        -v "$ENV_MASK_DIR/.env:/workspace/.env" \
        -w /workspace \
        "$IMAGE_NAME" \
        python evaluate_bigcodebench.py "$PREDICTIONS_PATH"
else
    docker run --rm \
        -v "$(pwd):/workspace" \
        -v "$ENV_MASK_DIR/.env:/workspace/.env" \
        -w /workspace \
        "$IMAGE_NAME"
fi
