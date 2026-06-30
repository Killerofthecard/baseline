"""Strip markdown fences from existing BigCodeBench predictions."""

import json
import re
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


def fix_predictions(path: str) -> int:
    """Fix a single predictions file and return number of updated entries."""
    with open(path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    updated = 0
    for pred in predictions:
        cleaned = extract_code(pred["solution"])
        if cleaned != pred["solution"]:
            pred["solution"] = cleaned
            updated += 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    return updated


def fix_result(path: str) -> int:
    """Fix a single result file and return number of updated entries."""
    if not Path(path).exists():
        return 0

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated = 0
    for pred in data.get("results", []):
        cleaned = extract_code(pred["solution"])
        if cleaned != pred["solution"]:
            pred["solution"] = cleaned
            updated += 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return updated


def main() -> None:
    for predictions_path in Path("results").rglob("bigcodebench_predictions.json"):
        predictions_path = str(predictions_path)
        result_path = str(Path(predictions_path).with_name("bigcodebench_result.json"))

        updated_preds = fix_predictions(predictions_path)
        updated_result = fix_result(result_path)

        print(f"{predictions_path}: fixed {updated_preds} predictions")
        if updated_result:
            print(f"{result_path}: fixed {updated_result} results")


if __name__ == "__main__":
    main()
