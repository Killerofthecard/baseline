#!/bin/bash
# run_eval_in_docker.sh
# Evaluate AgentNet BigCodeBench predictions in Docker (no bind mount pollution)

set -e

# Configuration
PREDICTIONS_FILE="results/gpt-4.1-mini/AgentNet/bigcodebench_predictions.json"
BENCHMARK_FILE="benchmark/bigcodebench_hard.json"
OUTPUT_SUMMARY="bigcodebench_eval_summary.json"

# Check if predictions file exists
if [ ! -f "$PREDICTIONS_FILE" ]; then
    echo "Error: Predictions file not found: $PREDICTIONS_FILE"
    echo "Please run AgentNet inference first:"
    echo "  cd baseline_method/AgentNet/AgentNet_Code"
    echo "  python run_bigcodebench.py"
    exit 1
fi

# Check if benchmark file exists
if [ ! -f "$BENCHMARK_FILE" ]; then
    echo "Error: Benchmark file not found: $BENCHMARK_FILE"
    exit 1
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Build Docker image if not exists
echo "Building Docker evaluation image..."
docker build -f Dockerfile.eval -t bigcodebench-eval .

# Run evaluation in Docker WITHOUT bind mount (copy files in/out instead)
CONTAINER_NAME="bigcodebench-eval-$(date +%s)"

echo "Running evaluation in isolated container..."

# Create container with a long-running entrypoint
docker create \
    --name "$CONTAINER_NAME" \
    bigcodebench-eval \
    sleep infinity

# Start container in background
docker start "$CONTAINER_NAME"

# Wait a moment for container to be ready
sleep 2

# Copy predictions and benchmark INTO container
docker cp "$PREDICTIONS_FILE" "$CONTAINER_NAME:/workspace/predictions.json"

# Ensure benchmark directory exists, then copy benchmark file
docker exec "$CONTAINER_NAME" mkdir -p /workspace/benchmark
docker cp "$BENCHMARK_FILE" "$CONTAINER_NAME:/workspace/benchmark/bigcodebench_hard.json"

# Also copy latest evaluate script into container (in case Dockerfile is stale)
docker cp "evaluate_bigcodebench.py" "$CONTAINER_NAME:/workspace/evaluate_bigcodebench.py"

# Now run the actual evaluation
docker exec "$CONTAINER_NAME" python evaluate_bigcodebench.py /workspace/predictions.json

# Get exit code
EXIT_CODE=$?

# Copy results OUT of container
docker cp "$CONTAINER_NAME:/workspace/$OUTPUT_SUMMARY" "$OUTPUT_SUMMARY" 2>/dev/null || true

# Cleanup container
docker rm -f "$CONTAINER_NAME" > /dev/null

echo ""
echo "Evaluation complete!"
if [ -f "$OUTPUT_SUMMARY" ]; then
    echo "Results saved to: $OUTPUT_SUMMARY"
else
    echo "Results were printed to stdout above."
fi

exit $EXIT_CODE
