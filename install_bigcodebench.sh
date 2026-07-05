#!/bin/bash
# install_bigcodebench.sh
# Install official BigCodeBench evaluation tools for local evaluation.
#
# Usage:
#     bash install_bigcodebench.sh

set -e

echo "Installing official BigCodeBench evaluation tools..."
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if not exists
if [ ! -d ".venv_bigcodebench" ]; then
    echo "Creating virtual environment .venv_bigcodebench..."
    python3 -m venv .venv_bigcodebench
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv_bigcodebench/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install bigcodebench package
echo "Installing bigcodebench package..."
pip install bigcodebench --upgrade

# Install evaluation dependencies
echo "Installing evaluation dependencies..."
pip install -I -r https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt

echo ""
echo "Installation complete!"
echo ""
echo "To use the official evaluation tool:"
echo "  1. Activate the virtual environment:"
echo "     source .venv_bigcodebench/bin/activate"
echo ""
echo "  2. Convert your predictions to official format:"
echo "     python convert_to_bigcodebench_official.py \\"
echo "       --input results/gpt-4.1-mini/YOUR_METHOD/bigcodebench_predictions.json \\"
echo "       --output bcb_results/your_predictions.jsonl"
echo ""
echo "  3. Run evaluation:"
echo "     bigcodebench.evaluate \\"
echo "       --execution local \\"
echo "       --split instruct \\"
echo "       --subset hard \\"
echo "       --samples bcb_results/your_predictions.jsonl \\"
echo "       --save_pass_rate \\"
echo "       --parallel $(nproc)"
echo ""
echo "Or simply use the wrapper script:"
echo "  bash run_bigcodebench_official_eval.sh \\"
echo "    results/gpt-4.1-mini/YOUR_METHOD/bigcodebench_predictions.json \\"
echo "    your_eval --local"
