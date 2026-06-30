"""
Minimal end-to-end script for BigCodeBench-hard.

It loads the dataset, saves all problems to a local JSON file, asks an LLM to
produce Python code for the first problem, and saves the prediction in the
format expected by the bigcodebench evaluation harness.

Usage:
    source .venv/bin/activate
    python test_run_bigcodebench_hard.py

To evaluate (after installing bigcodebench):
    pip install bigcodebench
    python -m bigcodebench.evaluate --dataset bigcodebench-hard --samples predictions.json
"""

import json
import os
import re

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def extract_code(text: str) -> str:
    """Extract Python code from the LLM output, falling back to the full text."""
    match = re.search(r"```python\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip()


def save_predictions(predictions: list[dict], path: str = "predictions.json") -> None:
    """Save predictions in the bigcodebench evaluation format."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(predictions)} prediction(s) to {path}")


def save_dataset(problems: list[dict], path: str = "bigcodebench_hard.json") -> None:
    """Save the raw BigCodeBench-hard problems to a local JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(problems)} problem(s) to {path}")


def main() -> None:
    # 1. Load BigCodeBench-hard dataset
    dataset = load_dataset("bigcode/bigcodebench-hard", split="v0.1.4")

    # 2. Save all problems to JSON
    problems = []
    for example in dataset:
        problems.append(
            example
        )
    save_dataset(problems, "bigcodebench_hard.json")


if __name__ == "__main__":
    main()
