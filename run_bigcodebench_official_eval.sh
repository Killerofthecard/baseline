#!/bin/bash
# run_bigcodebench_official_eval.sh
# Evaluate BigCodeBench predictions using the official BigCodeBench evaluation tool.
#
# This script supports TWO evaluation modes:
#   1. Docker mode (default): Uses official bigcodebench/bigcodebench-evaluate Docker image
#   2. Local mode (--local): Uses locally installed bigcodebench package
#
# Usage:
#     bash run_bigcodebench_official_eval.sh <predictions.json> [output_name] [--local]
#
# Examples:
#     # Docker mode (recommended, requires Docker and the image)
#     bash run_bigcodebench_official_eval.sh \
#         results/gpt-4.1-mini/TacoMAS-MultiAgent/bigcodebench_predictions.json \
#         tacomas_eval
#
#     # Local mode (if Docker image is unavailable)
#     bash run_bigcodebench_official_eval.sh \
#         results/gpt-4.1-mini/TacoMAS-MultiAgent/bigcodebench_predictions.json \
#         tacomas_eval \
#         --local

set -e

# Parse arguments
PREDICTIONS_JSON=""
OUTPUT_NAME="official_eval"
USE_LOCAL=false

for arg in "$@"; do
    if [ "$arg" = "--local" ]; then
        USE_LOCAL=true
    elif [ -z "$PREDICTIONS_JSON" ] && [ -f "$arg" ]; then
        PREDICTIONS_JSON="$arg"
    elif [ "$arg" != "--local" ]; then
        OUTPUT_NAME="$arg"
    fi
done

if [ -z "$PREDICTIONS_JSON" ]; then
    echo "Usage: bash run_bigcodebench_official_eval.sh <predictions.json> [output_name] [--local]"
    echo ""
    echo "Options:"
    echo "  --local    Use locally installed bigcodebench instead of Docker"
    echo ""
    echo "Examples:"
    echo "  # Docker mode (default)"
    echo "  bash run_bigcodebench_official_eval.sh \\"
    echo "    results/gpt-4.1-mini/TacoMAS-MultiAgent/bigcodebench_predictions.json \\"
    echo "    tacomas_eval"
    echo ""
    echo "  # Local mode"
    echo "  bash run_bigcodebench_official_eval.sh \\"
    echo "    results/gpt-4.1-mini/TacoMAS-MultiAgent/bigcodebench_predictions.json \\"
    echo "    tacomas_eval --local"
    exit 1
fi

if [ ! -f "$PREDICTIONS_JSON" ]; then
    echo "Error: Predictions file not found: $PREDICTIONS_JSON"
    exit 1
fi

# Create output directory
RESULTS_DIR="bcb_results"
mkdir -p "$RESULTS_DIR"

# Convert to official format
JSONL_FILE="$RESULTS_DIR/${OUTPUT_NAME}.jsonl"
echo "Converting predictions to official BigCodeBench format..."
python convert_to_bigcodebench_official.py \
    --input "$PREDICTIONS_JSON" \
    --output "$JSONL_FILE"

if [ "$USE_LOCAL" = true ]; then
    # Local mode: use pip-installed bigcodebench
    echo "Running evaluation in LOCAL mode..."
    
    # Check if bigcodebench is installed
    if ! python -c "import bigcodebench" 2>/dev/null; then
        echo "Error: bigcodebench package not installed."
        echo "Install it with: pip install bigcodebench"
        echo "Or install eval dependencies:"
        echo "  pip install -r https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt"
        exit 1
    fi
    
    # Run official evaluation locally
    bigcodebench.evaluate \
        --execution local \
        --split instruct \
        --subset hard \
        --samples "$JSONL_FILE" \
        --save_pass_rate \
        --parallel $(nproc) \
        --root "$RESULTS_DIR"
    
else
    # Docker mode
    echo "Running evaluation in DOCKER mode..."
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker is not installed. Use --local flag for local evaluation."
        exit 1
    fi
    
    # Check if image exists, try to pull if not
    if ! docker images bigcodebench/bigcodebench-evaluate:latest 2>/dev/null | grep -q bigcodebench; then
        echo "Pulling official BigCodeBench evaluation Docker image..."
        echo "(This may take a while, please wait...)"
        if ! docker pull bigcodebench/bigcodebench-evaluate:latest; then
            echo ""
            echo "Warning: Failed to pull Docker image. Falling back to local mode..."
            echo "To use local mode, install bigcodebench: pip install bigcodebench"
            echo "Then run with --local flag"
            exit 1
        fi
    fi
    
    # Run official evaluation in Docker
    docker run --rm \
        -v "$(pwd)/$RESULTS_DIR:/app" \
        bigcodebench/bigcodebench-evaluate:latest \
        --execution local \
        --split instruct \
        --subset hard \
        --samples "/app/${OUTPUT_NAME}.jsonl" \
        --save_pass_rate \
        --parallel $(nproc) \
        --root /app
fi

echo ""
echo "Evaluation complete!"
echo "Results saved in: $RESULTS_DIR/"
echo ""
echo "Look for files like:"
echo "  - ${OUTPUT_NAME}_eval_results.json"
echo "  - ${OUTPUT_NAME}_pass_at_k.json"
