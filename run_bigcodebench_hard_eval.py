"""
Run LLM on 5 BigCodeBench-hard cases and save predictions.

Usage:
    source .venv/bin/activate
    python run_bigcodebench_hard_eval.py

Outputs:
    - bigcodebench_hard_5samples.json : the 5 selected problems
    - predictions_5samples.json       : model predictions in evaluation format

To evaluate separately:
    python evaluate_bigcodebench_hard.py
"""

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

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


def generate_solution(
    client: OpenAI,
    model: str,
    instruct_prompt: str,
    code_prompt: str,
) -> str:
    """Ask the LLM to complete the function."""
    full_prompt = (
        f"{instruct_prompt}\n\n"
        f"Complete the following Python function:\n\n"
        f"{code_prompt}"
        "\nPlease output ONLY the completed function code, wrapped in a ```python ... ``` block."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert programmer. Write correct, concise Python code."},
            {"role": "user", "content": full_prompt},
        ],
        temperature=0.0,
        max_tokens=2048,
    )

    raw_output = response.choices[0].message.content or ""
    return extract_code(raw_output)


def main() -> None:
    num_cases = 5
    local_dataset_path = "bigcodebench_hard.json"

    # 1. Load dataset from local JSON
    with open(local_dataset_path, "r", encoding="utf-8") as f:
        all_problems = json.load(f)
    problems = all_problems[:num_cases]

    # 2. Save selected problems
    with open("bigcodebench_hard_5samples.json", "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(problems)} problems to bigcodebench_hard_5samples.json")

    # 3. Generate predictions with LLM
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    predictions = []
    for problem in tqdm(problems, desc="Generating"):
        task_id = problem["task_id"]
        solution = generate_solution(
            client,
            model,
            problem["instruct_prompt"],
            problem["code_prompt"],
        )
        predictions.append({"task_id": task_id, "solution": solution})

    predictions_path = "predictions_5samples.json"
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(predictions)} predictions to {predictions_path}")
    print("\nRun 'python evaluate_bigcodebench_hard.py' to evaluate.")


if __name__ == "__main__":
    main()
