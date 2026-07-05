#!/usr/bin/env python3
"""
Convert project prediction format to BigCodeBench official evaluation format.

The official format is a JSONL file where each line is:
    {"task_id": "BigCodeBench/1", "solution": "def f():\n    return 1"}

Usage:
    python convert_to_bigcodebench_official.py \
        --input results/gpt-4.1-mini/TacoMAS-MultiAgent/bigcodebench_predictions.json \
        --output bcb_results/tacomas_predictions.jsonl

Then evaluate with:
    docker run -v $(pwd)/bcb_results:/app bigcodebench/bigcodebench-evaluate:latest \
        --execution local --split instruct --subset hard \
        --samples /app/tacomas_predictions.jsonl
"""

import argparse
import json
from pathlib import Path


def convert_predictions(input_path: str, output_path: str) -> None:
    """Convert project's JSON predictions to BigCodeBench official JSONL format."""
    
    with open(input_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            task_id = pred.get("task_id", "")
            solution = pred.get("solution", "")
            
            # BigCodeBench official format
            official_record = {
                "task_id": task_id,
                "solution": solution,
            }
            f.write(json.dumps(official_record, ensure_ascii=False) + "\n")
    
    print(f"Converted {len(predictions)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert project predictions to BigCodeBench official format"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to project's bigcodebench_predictions.json"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output JSONL file (e.g., bcb_results/predictions.jsonl)"
    )
    args = parser.parse_args()
    
    convert_predictions(args.input, args.output)


if __name__ == "__main__":
    main()
