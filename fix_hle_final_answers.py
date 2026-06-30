"""Re-extract final answers for existing HLE results using improved fallback logic."""

import json
import re
from pathlib import Path


def extract_final_answer(text: str) -> str | None:
    """Extract answer from <<< >>> block, falling back to last non-empty line."""
    match = re.search(r"<<<(.*?)>>>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def fix_result(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated = 0
    correct = 0
    for record in data.get("results", []):
        full = record.get("full_response", "")
        old_final = record.get("final_answer")
        new_final = extract_final_answer(full)
        if new_final != old_final:
            record["final_answer"] = new_final
            updated += 1

    if updated:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return updated


def main() -> None:
    for path in Path("results").rglob("hle_result.json"):
        path = str(path)
        updated = fix_result(path)
        print(f"{path}: updated {updated} final answers")


if __name__ == "__main__":
    main()
