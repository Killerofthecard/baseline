#!/bin/bash
# Run all 4 baselines on a chosen benchmark.
#
# Usage:
#   ./run_baselines.sh gaia [num_cases] [model]
#   ./run_baselines.sh bigcodebench [num_cases] [model]
#   ./run_baselines.sh hle [num_cases] [model]
#
# Examples:
#   ./run_baselines.sh gaia
#   ./run_baselines.sh gaia 5 gpt-4o
#   ./run_baselines.sh bigcodebench 10 gpt-4o-mini
#   ./run_baselines.sh hle 20 gpt-4o-mini

set -e

# Use the project virtual environment
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
else
    echo "Error: .venv/bin/activate not found. Please create a virtual environment first."
    exit 1
fi

BENCHMARK=${1:-gaia}
NUM_CASES=${2:-}
MODEL=${3:-}

if [[ "$BENCHMARK" != "gaia" && "$BENCHMARK" != "bigcodebench" && "$BENCHMARK" != "hle" ]]; then
    echo "Error: benchmark must be 'gaia', 'bigcodebench', or 'hle'"
    echo "Usage: $0 gaia|bigcodebench|hle [num_cases] [model]"
    exit 1
fi

# Build optional arguments
ARGS=""
if [[ -n "$NUM_CASES" ]]; then
    ARGS="$ARGS -n $NUM_CASES"
fi
if [[ -n "$MODEL" ]]; then
    ARGS="$ARGS -m $MODEL"
    export OPENAI_MODEL="$MODEL"
fi

echo "=================================================="
echo "Running baselines on: $BENCHMARK"
if [[ -n "$NUM_CASES" ]]; then
    echo "Number of cases: $NUM_CASES"
else
    echo "Number of cases: all"
fi
if [[ -n "$MODEL" ]]; then
    echo "Model: $MODEL"
fi
echo "=================================================="

BASELINES=("vanilla" "cot" "react" "multi_agent_debate")
for baseline in "${BASELINES[@]}"; do
    echo ""
    echo "--------------------------------------------------"
    echo "Running $baseline on $BENCHMARK"
    echo "--------------------------------------------------"

    if [[ "$BENCHMARK" == "gaia" ]]; then
        python "$baseline/run_gaia.py" $ARGS
    elif [[ "$BENCHMARK" == "hle" ]]; then
        python "$baseline/run_hle.py" $ARGS
    else
        python "$baseline/run_bigcodebench.py" $ARGS
    fi
done

echo ""
echo "=================================================="
echo "All baselines completed for $BENCHMARK"
echo "=================================================="
