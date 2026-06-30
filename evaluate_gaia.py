"""
Evaluate GAIA baseline results.

Usage:
    python evaluate_gaia.py [results_json_path]

If no path is given, scans results/*/*/gaia_result.json and prints a summary table.
"""

import json
import os
import sys
from pathlib import Path


def normalize_answer(text: str | None) -> str:
    """Normalize answer string for comparison."""
    if text is None:
        return ""
    return str(text).strip().lower().rstrip(".")


def load_results(path: str) -> tuple[list[dict], dict]:
    """Load results and metadata from a GAIA result file.

    Supports both the new format (top-level metadata + results list) and the old
    plain list format.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        return data["results"], data.get("metadata", {})
    if isinstance(data, list):
        return data, {}
    raise ValueError(f"Unexpected GAIA result format in {path}")


def evaluate_file(path: str) -> dict:
    """Evaluate a single GAIA results file and return metrics."""
    records, metadata = load_results(path)

    total = len(records)
    correct = 0
    level_correct: dict[str, int] = {}
    level_total: dict[str, int] = {}

    for record in records:
        gold = normalize_answer(record.get("gold_answer"))
        pred = normalize_answer(record.get("final_answer"))
        level = str(record.get("level", "unknown"))

        level_total[level] = level_total.get(level, 0) + 1
        if gold and pred and gold == pred:
            correct += 1
            level_correct[level] = level_correct.get(level, 0) + 1

    computed_accuracy = correct / total if total else 0.0

    return {
        "path": path,
        "metadata": metadata,
        "total": total,
        "correct": correct,
        "accuracy": computed_accuracy,
        "level_breakdown": {
            level: {
                "correct": level_correct.get(level, 0),
                "total": level_total[level],
                "accuracy": level_correct.get(level, 0) / level_total[level],
            }
            for level in sorted(level_total.keys(), key=lambda x: int(x) if x.isdigit() else 999)
        },
    }


def scan_results(base_dir: str = "results") -> list[dict]:
    """Scan results directory for all gaia_result.json files."""
    metrics = []
    for path in Path(base_dir).rglob("gaia_result.json"):
        metrics.append(evaluate_file(str(path)))
    return metrics


def main() -> None:
    if len(sys.argv) > 1:
        path = sys.argv[1]
        metrics_list = [evaluate_file(path)]
    else:
        metrics_list = scan_results()

    if not metrics_list:
        print("No GAIA result files found.")
        return

    print("=" * 80)
    print("GAIA Evaluation Results")
    print("=" * 80)

    for metrics in metrics_list:
        print(f"\nFile: {metrics['path']}")
        if metrics["metadata"]:
            meta = metrics["metadata"]
            print(f"  Baseline: {meta.get('baseline', 'unknown')}")
            print(f"  Model:    {meta.get('model', 'unknown')}")
            print(f"  Cases:    {meta.get('num_cases', metrics['total'])}")
            print(f"  Runtime:  {meta.get('runtime_seconds', 'unknown')}s")
        print(f"  Total:    {metrics['total']}")
        print(f"  Correct:  {metrics['correct']}")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")

        if metrics["level_breakdown"]:
            print("  Level breakdown:")
            for level, stats in metrics["level_breakdown"].items():
                print(f"    Level {level}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.2%}")


if __name__ == "__main__":
    main()
