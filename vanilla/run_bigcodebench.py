"""Vanilla baseline for BigCodeBench-hard code generation."""

import argparse
import json
import os
import re
import sys
from datetime import datetime

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import tool_loop_solve


SYSTEM_PROMPT = (
    "You are an expert programmer. Generate correct, concise Python code. "
    "Use tools only if you need to look up external information. "
    "Output the final solution wrapped in triple angle brackets like <<<code>>>."
)


def load_problems(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def generate_code(instruct_prompt: str, code_prompt: str, model: str | None = None) -> dict:
    full_prompt = (
        f"{instruct_prompt}\n\n"
        f"Complete the following Python function:\n\n"
        f"{code_prompt}"
        "\nPlease output ONLY the completed function code, wrapped in a ```python ... ``` block."
    )
    result = tool_loop_solve(full_prompt, SYSTEM_PROMPT, model=model)
    result["final_answer"] = extract_code(result.get("final_answer") or result.get("full_response", ""))
    return result


def save_predictions(path: str, predictions: list[dict]) -> None:
    """Write the official-format predictions list to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def save_result(path: str, metadata: dict, predictions: list[dict]) -> None:
    """Write the canonical result (metadata + predictions) to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "results": predictions}, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vanilla baseline for BigCodeBench-hard")
    parser.add_argument("-d", "--dataset", default="benchmark/bigcodebench_hard.json", help="Path to BigCodeBench dataset JSON")
    parser.add_argument("-n", "--num_cases", type=int, default=None, help="Number of cases (default: all)")
    parser.add_argument("-o", "--output", default=None, help="Output predictions JSON path")
    parser.add_argument("-m", "--model", default=None, help="Override model name")
    args = parser.parse_args()

    problems = load_problems(args.dataset)
    if args.num_cases is not None:
        problems = problems[: args.num_cases]

    model_name = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if args.model:
        os.environ["OPENAI_MODEL"] = args.model

    predictions_path = args.output or os.path.join("results", model_name, "vanilla", "bigcodebench_predictions.json")
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    result_path = os.path.join(os.path.dirname(predictions_path), "bigcodebench_result.json")

    start_time = datetime.now()
    predictions = []

    metadata = {
        "benchmark": "bigcodebench-hard",
        "baseline": "vanilla",
        "model": model_name,
        "num_cases": len(problems),
        "predictions_path": predictions_path,
        "start_time": start_time.isoformat(),
        "end_time": start_time.isoformat(),
        "runtime_seconds": 0.0,
    }

    # Save initial empty files so outputs exist immediately
    save_predictions(predictions_path, predictions)
    save_result(result_path, metadata, predictions)

    for problem in tqdm(problems, desc="Vanilla"):
        result = generate_code(problem["instruct_prompt"], problem["code_prompt"], model=args.model)
        predictions.append(
            {
                "task_id": problem["task_id"],
                "solution": result["final_answer"],
            }
        )

        now = datetime.now()
        metadata.update(
            {
                "end_time": now.isoformat(),
                "runtime_seconds": (now - start_time).total_seconds(),
            }
        )
        save_predictions(predictions_path, predictions)
        save_result(result_path, metadata, predictions)

    print(f"Saved {len(predictions)} predictions to {predictions_path}")
    print(f"Saved result (with metadata) to {result_path}")


if __name__ == "__main__":
    main()
