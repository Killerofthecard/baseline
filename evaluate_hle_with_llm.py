#!/usr/bin/env python3
"""
Evaluate HLE benchmark results using LLM as judge (multi-threaded).

Reads from results/gpt-4.1-mini/{method}/hle_result.json for each baseline method
and uses LLM as judge to evaluate each model answer against ground truth.

Usage:
    python evaluate_hle_with_llm.py
    python evaluate_hle_with_llm.py --model gpt-4.1-mini --max-workers 10
    python evaluate_hle_with_llm.py --methods cot,vanilla --limit 50
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc or ""
            self.n = 0
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.n += 1
            if self.total:
                print(f"\r{self.desc} {self.n}/{self.total}", end="", flush=True)
            print()
        def update(self, n=1):
            self.n += n

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from baseline_method/selforg/ directory
_selforg_dir = Path(__file__).parent / "baseline_method" / "selforg"
load_dotenv(_selforg_dir / ".env")


JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for the Humanity's Last Exam (HLE) benchmark.

Your task is to determine whether the model's answer is correct compared to the gold (ground truth) answer.

Question:
{question}

Gold Answer (Ground Truth):
{gold_answer}

Model's Answer:
{model_answer}

Instructions:
1. For multiple-choice questions, the model answer should match the gold answer letter exactly (case-insensitive), or contain the correct option text.
2. For free-response questions, the model answer should be semantically equivalent to the gold answer. Minor formatting differences, extra explanations, or slight paraphrasing are acceptable as long as the core answer is correct.
3. If the model answer is empty, "timeout", "N/A", or clearly indicates no answer was produced, mark it as incorrect.

Respond with ONLY a JSON object in this exact format:
{{"correct": true/false, "reason": "brief explanation"}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HLE results with LLM judge (multi-threaded)")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="Model directory name under results/ (default: gpt-4.1-mini)")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated list of methods to evaluate (default: all found)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Judge temperature")
    parser.add_argument("--max-workers", type=int, default=10, help="Max concurrent API calls (default 10)")
    parser.add_argument("--output-dir", type=Path, default=Path("hle_eval_results"), help="Output directory for evaluation results")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to evaluate per method")
    parser.add_argument("--offset", type=int, default=0, help="Start from this question index")
    parser.add_argument("--resume", action="store_true", help="Skip already evaluated questions")
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-question output")
    return parser.parse_args()


def discover_methods(results_dir: Path) -> List[str]:
    """Discover all methods under results/{model}/ that have hle_result.json."""
    methods = []
    if not results_dir.exists():
        return methods
    for subdir in sorted(results_dir.iterdir()):
        if subdir.is_dir() and (subdir / "hle_result.json").exists():
            methods.append(subdir.name)
    return methods


def load_hle_results(input_file: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load hle_result.json and return (metadata, results_list)."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("metadata", {}), data.get("results", [])


def load_existing_results(output_file: Path) -> Dict[str, Dict[str, Any]]:
    """Load existing evaluation results if resuming."""
    if not output_file.exists():
        return {}
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {r["id"]: r for r in data.get("results", [])}
    except:
        return {}


def call_llm_judge(
    client: OpenAI,
    model: str,
    temperature: float,
    question: str,
    gold_answer: str,
    model_answer: str,
) -> Tuple[bool, str]:
    """Call LLM judge to evaluate correctness. Returns (is_correct, reason)."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        model_answer=model_answer if model_answer else "(empty / no answer)",
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise evaluation assistant. Respond only with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        result = json.loads(raw)
        is_correct = bool(result.get("correct", False))
        reason = str(result.get("reason", "No reason provided"))
        return is_correct, reason
    except Exception as e:
        return False, f"LLM judge error: {str(e)}"


def evaluate_single(
    result_entry: Dict[str, Any],
    client: OpenAI,
    model: str,
    temperature: float,
    verbose: bool,
    global_idx: int,
    total: int,
) -> Optional[Dict[str, Any]]:
    """Evaluate a single question from hle_result.json entry."""
    problem_id = result_entry["id"]
    question = result_entry.get("question", "")
    gold_answer = str(result_entry.get("gold_answer", "")).strip()
    model_answer = str(result_entry.get("final_answer", "")).strip()

    # Call LLM judge
    is_correct, reason = call_llm_judge(
        client=client,
        model=model,
        temperature=temperature,
        question=question,
        gold_answer=gold_answer,
        model_answer=model_answer,
    )

    if verbose:
        status = "✓ CORRECT" if is_correct else "✗ WRONG"
        print(f"\n[{global_idx + 1}/{total}] {problem_id}")
        print(f"  Gold:   {gold_answer}")
        print(f"  Model:  {model_answer[:100]}{'...' if len(model_answer) > 100 else ''}")
        print(f"  {status} — {reason}")

    return {
        "id": problem_id,
        "question": question,
        "gold_answer": gold_answer,
        "final_answer": model_answer,
        "correct": is_correct,
        "reason": reason,
    }


def save_results(
    output_file: Path,
    results: List[Dict[str, Any]],
    model: str,
    method: str,
    judge_model: str,
    start_time: float,
) -> None:
    """Save evaluation results to JSON file."""
    correct_count = sum(1 for r in results if r.get("correct", False))
    total_evaluated = len(results)
    accuracy = correct_count / total_evaluated if total_evaluated > 0 else 0.0

    output = {
        "metadata": {
            "benchmark": "hle",
            "evaluation_method": "llm_judge",
            "model": model,
            "method": method,
            "judge_model": judge_model,
            "num_evaluated": total_evaluated,
            "correct": correct_count,
            "accuracy": accuracy,
            "start_time": datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": time.time() - start_time,
        },
        "results": results,
    }

    # Atomic write: write to temp file first, then rename
    temp_file = output_file.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    temp_file.rename(output_file)


def evaluate_method(
    method: str,
    results_dir: Path,
    client: OpenAI,
    judge_model: str,
    temperature: float,
    max_workers: int,
    limit: Optional[int],
    offset: int,
    resume: bool,
    verbose: bool,
    output_dir: Path,
) -> None:
    """Evaluate a single method's HLE results."""
    input_file = results_dir / method / "hle_result.json"
    output_file = output_dir / f"{method}_hle_eval.json"

    print(f"\n{'='*60}")
    print(f"Evaluating method: {method}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")

    if not input_file.exists():
        print(f"Warning: Input file not found, skipping: {input_file}")
        return

    metadata, hle_results = load_hle_results(input_file)
    print(f"Loaded {len(hle_results)} results from {input_file}")
    if metadata:
        print(f"  Original model: {metadata.get('model', 'unknown')}")
        print(f"  Original accuracy: {metadata.get('accuracy', 'unknown')}")

    # Load existing results for resume
    existing_results = {}
    if resume and output_file.exists():
        existing_results = load_existing_results(output_file)
        print(f"Loaded {len(existing_results)} existing evaluations for resume")

    # Prepare questions to evaluate
    start_idx = offset
    end_idx = offset + limit if limit is not None else len(hle_results)
    results_to_eval = hle_results[start_idx:end_idx]

    # Filter out already evaluated if resuming
    if resume:
        results_to_eval = [r for r in results_to_eval if r["id"] not in existing_results]

    print(f"Will evaluate {len(results_to_eval)} questions [{start_idx}, {end_idx})")
    print(f"Judge model: {judge_model}")
    print(f"Max workers: {max_workers}")

    # Collect existing results if resuming
    results = list(existing_results.values()) if resume else []
    correct_count = sum(1 for r in results if r.get("correct", False))
    total_evaluated = len(results)

    start_time = time.time()
    last_save_time = start_time

    # Multi-threaded evaluation
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for i, result_entry in enumerate(results_to_eval):
            global_idx = start_idx + i
            future = executor.submit(
                evaluate_single,
                result_entry,
                client,
                judge_model,
                temperature,
                verbose,
                global_idx,
                len(hle_results),
            )
            future_to_idx[future] = global_idx

        # Process completed tasks with progress bar
        pbar = tqdm(total=len(results_to_eval), desc=f"Eval {method}", unit="question")
        for future in as_completed(future_to_idx):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    if result.get("correct", False):
                        correct_count += 1
                    total_evaluated += 1
            except Exception as e:
                global_idx = future_to_idx[future]
                print(f"\nError evaluating index {global_idx}: {e}")

            pbar.update(1)

            # Save intermediate results every 30 seconds
            if time.time() - last_save_time > 30:
                save_results(output_file, results, method, method, judge_model, start_time)
                last_save_time = time.time()

        pbar.close()

    # Final save
    save_results(output_file, results, method, method, judge_model, start_time)

    # Print summary
    accuracy = correct_count / total_evaluated if total_evaluated > 0 else 0.0
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Method: {method} — Evaluation Complete!")
    print(f"  Total evaluated: {total_evaluated}")
    print(f"  Correct:         {correct_count}")
    print(f"  Accuracy:        {accuracy:.6f} ({accuracy*100:.2f}%)")
    print(f"  Runtime:         {elapsed:.1f}s")
    if total_evaluated > 0:
        print(f"  Avg per question: {elapsed/total_evaluated:.2f}s")
    print(f"  Results saved:   {output_file}")


def main() -> None:
    args = parse_args()

    # Load API config from .env
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    default_model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    judge_model = default_model

    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env or environment")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Determine results directory
    results_dir = Path("results") / args.model
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    # Discover or use specified methods
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    else:
        methods = discover_methods(results_dir)

    if not methods:
        print(f"Error: No methods found with hle_result.json in {results_dir}")
        sys.exit(1)

    print(f"Found {len(methods)} method(s) to evaluate: {', '.join(methods)}")
    print(f"Results directory: {results_dir}")
    print(f"Judge model: {judge_model}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each method
    for method in methods:
        evaluate_method(
            method=method,
            results_dir=results_dir,
            client=client,
            judge_model=judge_model,
            temperature=args.temperature,
            max_workers=args.max_workers,
            limit=args.limit,
            offset=args.offset,
            resume=args.resume,
            verbose=args.verbose,
            output_dir=args.output_dir,
        )

    # Print overall summary
    print(f"\n{'='*60}")
    print(f"ALL METHODS EVALUATED")
    print(f"{'='*60}")
    for method in methods:
        output_file = args.output_dir / f"{method}_hle_eval.json"
        if output_file.exists():
            _, results = load_hle_results(output_file)
            correct = sum(1 for r in results if r.get("correct", False))
            total = len(results)
            acc = correct / total if total > 0 else 0.0
            print(f"  {method:20s}: {correct}/{total} = {acc*100:.2f}%")


if __name__ == "__main__":
    main()
