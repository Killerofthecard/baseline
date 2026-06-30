"""Re-parse verdicts in an existing hle_eval_summary.json using improved logic."""

import json
import re
import sys
from pathlib import Path


def parse_verdict(text: str) -> str:
    """Extract the last whole-word occurrence of CORRECT, INCORRECT, or UNSURE."""
    matches = re.findall(r"\b(CORRECT|INCORRECT|UNSURE)\b", text.upper())
    if matches:
        return matches[-1]
    return "UNSURE"


def main() -> None:
    summary_path = sys.argv[1] if len(sys.argv) > 1 else "hle_eval_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    for entry in summary:
        counts = {"CORRECT": 0, "INCORRECT": 0, "UNSURE": 0}
        for record in entry.get("results", []):
            response = record.get("judge_response", "")
            verdict = parse_verdict(response)
            record["verdict"] = verdict
            counts[verdict] += 1

        total = len(entry.get("results", []))
        entry["correct"] = counts["CORRECT"]
        entry["incorrect"] = counts["INCORRECT"]
        entry["unsure"] = counts["UNSURE"]
        entry["accuracy"] = counts["CORRECT"] / total if total else 0.0

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Re-parsed verdicts in {summary_path}")


if __name__ == "__main__":
    main()
