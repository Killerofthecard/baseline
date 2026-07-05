#!/usr/bin/env python3
"""
Convert SelfOrg inference results (JSONL) to BigCodeBench predictions format (JSON).

Extracts code from markdown ```python ... ``` blocks in the 'response' field
and outputs a JSON array with {"task_id": ..., "solution": ...} entries.

Usage:
    python convert_selforg_to_predictions.py \
        baseline_method/selforg/results/bigcodebench_hard/gpt-4.1-mini/selforg_infer.jsonl \
        --output results/gpt-4.1-mini/selforg/bigcodebench_predictions.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


def extract_code(text: str) -> str:
    """Extract Python code from markdown or angle-bracket blocks."""
    for pattern in (
        r"```python\s*\n(.*?)\n```",
        r"```\s*\n(.*?)\n```",
        r"<<<(.*?)>>>",
    ):
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return text.strip()


def convert_jsonl_to_predictions(input_path: str, output_path: str) -> None:
    """Read SelfOrg JSONL and write BigCodeBench predictions JSON."""
    predictions = []
    total = 0
    success = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            item = json.loads(line)

            task_id = item.get("task_id", "")
            response = item.get("response", "")
            error = item.get("error", "")

            if error:
                # If there was an inference error, use empty solution
                solution = ""
            else:
                solution = extract_code(response)
                if solution:
                    success += 1

            predictions.append({
                "task_id": task_id,
                "solution": solution,
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"Converted {total} entries → {output_path}")
    print(f"  With code extracted: {success}/{total}")
    print(f"  Empty/failed: {total - success}/{total}")


def main():
    parser = argparse.ArgumentParser(description="Convert SelfOrg JSONL to BigCodeBench predictions JSON")
    parser.add_argument("input", help="Path to selforg_infer.jsonl")
    parser.add_argument("--output", "-o", help="Output predictions JSON path (default: auto-detect)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Auto-detect output path if not specified
    if args.output:
        output_path = args.output
    else:
        # Try to infer from input path structure
        # e.g., baseline_method/selforg/results/bigcodebench_hard/gpt-4.1-mini/selforg_infer.jsonl
        # → results/gpt-4.1-mini/selforg/bigcodebench_predictions.json
        parts = input_path.parts
        if "baseline_method" in parts and "selforg" in parts:
            # Find the model name and method name from path
            try:
                results_idx = parts.index("results")
                model_name = parts[results_idx + 2]  # e.g., gpt-4.1-mini
                method_name = parts[results_idx + 3].replace("_infer.jsonl", "")
                output_path = f"results/{model_name}/{method_name}/bigcodebench_predictions.json"
            except (IndexError, ValueError):
                output_path = str(input_path.with_suffix("")) + "_predictions.json"
        else:
            output_path = str(input_path.with_suffix("")) + "_predictions.json"

    convert_jsonl_to_predictions(str(input_path), output_path)


if __name__ == "__main__":
    main()
