"""Vanilla baseline with tool support: direct tool-augmented QA."""

import argparse
import json
import os
import sys
from datetime import datetime

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import tool_loop_solve


SYSTEM_PROMPT = (
    "You are a helpful assistant answering GAIA benchmark questions. "
    "Use the fewest tool calls possible. "
    "If the question is simple or you already know the answer, answer directly. "
    "Only search the web or run code when you truly need external facts or computation."
)


def load_text_only_questions(path: str = "benchmark/gaia_text_only.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["text_only_cases"]


def solve(question: str) -> dict:
    """Direct tool-augmented question answering."""
    return tool_loop_solve(question, SYSTEM_PROMPT)


def save_progress(path: str, metadata: dict, results: list[dict]) -> None:
    """Write the current progress to disk (incremental save)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vanilla baseline for GAIA")
    parser.add_argument("-d", "--dataset", default="benchmark/gaia_text_only.json", help="Path to GAIA text-only JSON")
    parser.add_argument("-n", "--num_cases", type=int, default=None, help="Number of cases (default: all)")
    parser.add_argument("-o", "--output", default=None, help="Output JSON path")
    parser.add_argument("-m", "--model", default=None, help="Override model name")
    args = parser.parse_args()

    dataset = load_text_only_questions(args.dataset)
    if args.num_cases is not None:
        dataset = dataset[: args.num_cases]

    model_name = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if args.model:
        os.environ["OPENAI_MODEL"] = args.model

    output_path = args.output or os.path.join("results", model_name, "vanilla", "gaia_result.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    start_time = datetime.now()
    results = []
    correct = 0

    metadata = {
        "benchmark": "gaia",
        "baseline": "vanilla",
        "model": model_name,
        "num_cases": len(dataset),
        "correct": 0,
        "accuracy": 0.0,
        "start_time": start_time.isoformat(),
        "end_time": start_time.isoformat(),
        "runtime_seconds": 0.0,
    }

    # Save initial empty result file so the output exists immediately
    save_progress(output_path, metadata, results)

    pbar = tqdm(dataset, desc="Vanilla")
    for example in pbar:
        question = example["question"]
        gold = example.get("gold_answer")

        result = solve(question)
        final = result["final_answer"]

        if gold and str(gold).strip().lower() == str(final).strip().lower():
            correct += 1

        record = {
            "index": example.get("index"),
            "task_id": example.get("task_id"),
            "level": example.get("level"),
            "question": question,
            "gold_answer": gold,
            "final_answer": final,
            "full_response": result["full_response"],
            "tool_calls": result["tool_calls"],
            "reached_max_iterations": result["reached_max_iterations"],
        }
        results.append(record)

        now = datetime.now()
        metadata.update(
            {
                "correct": correct,
                "accuracy": correct / len(results) if results else 0.0,
                "end_time": now.isoformat(),
                "runtime_seconds": (now - start_time).total_seconds(),
            }
        )
        save_progress(output_path, metadata, results)
        pbar.set_postfix({"acc": f"{correct}/{len(results)}"})

    print(f"Vanilla: {correct}/{len(results)} correct -> {output_path}")


if __name__ == "__main__":
    main()
