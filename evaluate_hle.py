"""
Evaluate HLE baseline results using an LLM as a judge.

Usage:
    python evaluate_hle.py [results_json_path]

If no path is given, scans results/*/*/hle_result.json and prints a summary table.
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import call_llm


JUDGE_SYSTEM_PROMPT = (
    "You are a strict but fair evaluator for Humanity's Last Exam (HLE). "
    "You will be given a question, the ground-truth answer, and a model's predicted answer. "
    "Determine whether the predicted answer is semantically equivalent to the ground-truth answer. "
    "For multiple-choice questions, accept the correct letter or the correct option text. "
    "For numerical or mathematical answers, accept equivalent values even if formatted differently. "
    "If the predicted answer is empty, nonsensical, or clearly wrong, mark it INCORRECT. "
    "If you cannot confidently decide, mark it UNSURE.\n\n"
    "Output your verdict on a single line as one of: CORRECT, INCORRECT, UNSURE. "
    "**DO NOT provide any explanation.**"
)


def load_results(path: str) -> tuple[list[dict], dict]:
    """Load results and metadata from an HLE result file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        return data["results"], data.get("metadata", {})
    if isinstance(data, list):
        return data, {}
    raise ValueError(f"Unexpected HLE result format in {path}")


def build_judge_prompt(question: str, gold: str, full: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Ground-truth answer:\n{gold}\n\n"
        f"Model's full response:\n{full}\n\n"
        "Verdict (CORRECT / INCORRECT / UNSURE):"
    )


def parse_verdict(text: str) -> str:
    """Extract the last whole-word occurrence of CORRECT, INCORRECT, or UNSURE."""
    matches = re.findall(r"\b(CORRECT|INCORRECT|UNSURE)\b", text.upper())
    if matches:
        return matches[-1]
    return "UNSURE"


def judge_record(record: dict, model: str) -> dict:
    """Ask an LLM judge to evaluate a single record. Returns updated record."""
    question = record.get("question", "")
    gold = record.get("gold_answer", "")
    full = record.get("full_response", "")

    if not str(full).strip():
        return {
            **record,
            "judge_response": "(empty model response)",
            "verdict": "INCORRECT",
        }

    prompt = build_judge_prompt(question, str(gold), str(full))
    try:
        response = call_llm(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.0,
            max_tokens=512,
        )
        verdict = parse_verdict(response)
    except Exception as e:
        response = f"Error calling judge: {e}"
        verdict = "UNSURE"

    return {
        **record,
        "judge_response": response,
        "verdict": verdict,
    }


def evaluate_file(path: str, model: str, max_workers: int) -> dict:
    """Evaluate a single HLE results file using an LLM judge."""
    records, metadata = load_results(path)

    evaluated_records = []
    counts = {"CORRECT": 0, "INCORRECT": 0, "UNSURE": 0}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for record in tqdm(
            executor.map(lambda r: judge_record(r, model), records),
            total=len(records),
            desc="Judging",
        ):
            evaluated_records.append(record)
            counts[record["verdict"]] += 1

    total = len(evaluated_records)
    return {
        "path": path,
        "metadata": metadata,
        "judge_model": model,
        "total": total,
        "correct": counts["CORRECT"],
        "incorrect": counts["INCORRECT"],
        "unsure": counts["UNSURE"],
        "accuracy": counts["CORRECT"] / total if total else 0.0,
        "results": evaluated_records,
    }


def scan_results(base_dir: str = "results") -> list[str]:
    """Scan results directory for all hle_result.json files."""
    return [str(p) for p in Path(base_dir).rglob("hle_result.json")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HLE results with an LLM judge")
    parser.add_argument("path", nargs="?", default=None, help="Path to a single hle_result.json")
    parser.add_argument("-m", "--model", default=None, help="Judge model name (default: OPENAI_MODEL or gpt-4o-mini)")
    parser.add_argument("-w", "--max_workers", type=int, default=16, help="Number of parallel judge calls (default: 16)")
    parser.add_argument("-o", "--output", default="hle_eval_summary.json", help="Output summary JSON path")
    args = parser.parse_args()

    judge_model = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    if args.path:
        paths = [args.path]
    else:
        paths = scan_results()

    if not paths:
        print("No HLE result files found.")
        return

    print("=" * 80)
    print(f"HLE Evaluation Results (judge: {judge_model})")
    print("=" * 80)

    summary = []
    for path in paths:
        metrics = evaluate_file(path, judge_model, args.max_workers)
        summary.append(metrics)

        print(f"\nFile: {metrics['path']}")
        if metrics["metadata"]:
            meta = metrics["metadata"]
            print(f"  Baseline: {meta.get('baseline', 'unknown')}")
            print(f"  Model:    {meta.get('model', 'unknown')}")
            print(f"  Cases:    {meta.get('num_cases', metrics['total'])}")
        print(f"  Total:     {metrics['total']}")
        print(f"  Correct:   {metrics['correct']}")
        print(f"  Incorrect: {metrics['incorrect']}")
        print(f"  Unsure:    {metrics['unsure']}")
        print(f"  Accuracy:  {metrics['accuracy']:.2%}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved detailed summary to {args.output}")


if __name__ == "__main__":
    main()
