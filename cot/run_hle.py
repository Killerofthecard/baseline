"""Chain-of-Thought baseline for HLE (Humanity's Last Exam) gold subset, no tools, multithreaded, resumable."""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import text_solve


SYSTEM_PROMPT = (
    "You are an expert assistant answering difficult questions from Humanity's Last Exam. "
    "Think step by step, but do not use any tools. "
    "After your reasoning, provide only the final answer "
    "wrapped in triple angle brackets like <<<answer>>>."
)


def load_hle_questions(path: str = "benchmark/hle_gold_subset.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_answer(text: str | None) -> str:
    if text is None:
        return ""
    return str(text).strip().lower().rstrip(".")


def solve(question: str) -> dict:
    return text_solve(question, SYSTEM_PROMPT)


def save_progress(path: str, metadata: dict, results: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=2, ensure_ascii=False)


def load_existing_results(path: str) -> tuple[list[dict], dict]:
    """Load partial results if the output file already exists."""
    if not os.path.exists(path):
        return [], {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", []), data.get("metadata", {})


def process_example(example: dict) -> tuple[dict, bool]:
    """Process a single example. Returns (record, is_correct)."""
    question = example["question"]
    gold = example.get("answer")
    result = solve(question)
    final = result["final_answer"]
    is_correct = bool(gold and normalize_answer(gold) == normalize_answer(final))

    record = {
        "id": example.get("id"),
        "question": question,
        "gold_answer": gold,
        "final_answer": final,
        "full_response": result["full_response"],
        "tool_calls": result["tool_calls"],
        "reached_max_iterations": result["reached_max_iterations"],
    }
    return record, is_correct


def main() -> None:
    parser = argparse.ArgumentParser(description="CoT baseline for HLE (no tools, multithreaded, resumable)")
    parser.add_argument("-d", "--dataset", default="benchmark/hle_gold_subset.json", help="Path to HLE gold subset JSON")
    parser.add_argument("-n", "--num_cases", type=int, default=None, help="Number of cases (default: all)")
    parser.add_argument("-o", "--output", default=None, help="Output JSON path")
    parser.add_argument("-m", "--model", default=None, help="Override model name")
    parser.add_argument("-w", "--max_workers", type=int, default=16, help="Number of parallel threads (default: 16)")
    args = parser.parse_args()

    dataset = load_hle_questions(args.dataset)
    if args.num_cases is not None:
        dataset = dataset[: args.num_cases]

    model_name = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if args.model:
        os.environ["OPENAI_MODEL"] = args.model

    output_path = args.output or os.path.join("results", model_name, "cot", "hle_result.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    existing_results, existing_metadata = load_existing_results(output_path)
    results = existing_results.copy()
    processed_ids = {r["id"] for r in results if r.get("id") is not None}
    correct = sum(
        1
        for r in results
        if r.get("gold_answer") and normalize_answer(r["gold_answer"]) == normalize_answer(r.get("final_answer"))
    )

    start_time_str = existing_metadata.get("start_time") if existing_metadata else None
    start_time = datetime.fromisoformat(start_time_str) if start_time_str else datetime.now()
    save_lock = threading.Lock()

    metadata = {
        "benchmark": "hle",
        "baseline": "cot",
        "model": model_name,
        "num_cases": len(dataset),
        "correct": correct,
        "accuracy": correct / len(results) if results else 0.0,
        "start_time": start_time.isoformat(),
        "end_time": start_time.isoformat(),
        "runtime_seconds": 0.0,
    }
    save_progress(output_path, metadata, results)

    remaining = [ex for ex in dataset if ex.get("id") not in processed_ids]
    if not remaining:
        print(f"All {len(dataset)} cases already processed -> {output_path}")
        return

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for record, is_correct in tqdm(
            executor.map(process_example, remaining),
            total=len(remaining),
            desc="CoT",
        ):
            results.append(record)
            if is_correct:
                correct += 1

            now = datetime.now()
            metadata.update(
                {
                    "correct": correct,
                    "accuracy": correct / len(results) if results else 0.0,
                    "end_time": now.isoformat(),
                    "runtime_seconds": (now - start_time).total_seconds(),
                }
            )
            with save_lock:
                save_progress(output_path, metadata, results)

    print(f"CoT: {correct}/{len(results)} correct -> {output_path}")


if __name__ == "__main__":
    main()
