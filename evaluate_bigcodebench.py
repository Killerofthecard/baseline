"""
Evaluate BigCodeBench baseline predictions via subprocess isolation.

Usage:
    python evaluate_bigcodebench.py [predictions_json_path]

If no path is given, scans results/*/*/bigcodebench_predictions.json and prints a summary.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run_test(solution: str, test_code: str, timeout: int = 10) -> dict:
    """Execute solution + test code in a subprocess and return pass/fail status."""
    script = f"""\
{solution}

{test_code}

if __name__ == "__main__":
    import sys
    import unittest

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules["__main__"])
    runner = unittest.TextTestRunner(verbosity=0, stream=sys.stdout)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("__PASS__")
    else:
        print("__FAIL__")
        sys.exit(1)
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode == 0 and "__PASS__" in stdout:
            return {"passed": True, "stdout": stdout, "stderr": stderr}
        return {
            "passed": False,
            "stdout": stdout,
            "stderr": stderr,
            "error": f"exit code {result.returncode}",
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "timeout", "stdout": "", "stderr": ""}
    except Exception as e:
        return {"passed": False, "error": str(e), "stdout": "", "stderr": ""}
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def load_problems(path: str = "benchmark/bigcodebench_hard.json") -> dict[str, dict]:
    """Load BigCodeBench problems and index by task_id."""
    with open(path, "r", encoding="utf-8") as f:
        problems = json.load(f)
    return {p["task_id"]: p for p in problems}


def load_metadata(predictions_path: str) -> dict:
    """Load metadata from a sibling result/metadata file if it exists."""
    base = Path(predictions_path)
    for name in ("bigcodebench_result.json", "bigcodebench_metadata.json"):
        candidate = base.with_name(name)
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("metadata", data)
    return {}


def evaluate_file(predictions_path: str, problems: dict[str, dict]) -> dict:
    """Evaluate a single BigCodeBench predictions file."""
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    metadata = load_metadata(predictions_path)
    total = len(predictions)
    passed = 0
    details = []

    for pred in predictions:
        task_id = pred["task_id"]
        problem = problems.get(task_id)
        if problem is None:
            details.append({"task_id": task_id, "passed": False, "error": "Problem not found"})
            continue

        result = run_test(pred["solution"], problem["test"], timeout=10)
        if result["passed"]:
            passed += 1

        details.append(
            {
                "task_id": task_id,
                "passed": result["passed"],
                "error": result.get("error"),
                "stdout": result["stdout"],
                "stderr": result["stderr"],
            }
        )

    return {
        "path": predictions_path,
        "metadata": metadata,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "details": details,
    }


def scan_results(base_dir: str = "results") -> list[str]:
    """Scan results directory for all bigcodebench_predictions.json files."""
    return [str(p) for p in Path(base_dir).rglob("bigcodebench_predictions.json")]


def main() -> None:
    predictions_path = sys.argv[1] if len(sys.argv) > 1 else None

    problems = load_problems()

    if predictions_path:
        results = [evaluate_file(predictions_path, problems)]
    else:
        paths = scan_results()
        results = [evaluate_file(p, problems) for p in paths]

    if not results:
        print("No BigCodeBench prediction files found.")
        return

    print("=" * 80)
    print("BigCodeBench Evaluation Results")
    print("=" * 80)

    for metrics in results:
        print(f"\nFile: {metrics['path']}")
        if metrics["metadata"]:
            meta = metrics["metadata"]
            print(f"  Baseline: {meta.get('baseline', 'unknown')}")
            print(f"  Model:    {meta.get('model', 'unknown')}")
            print(f"  Cases:    {meta.get('num_cases', metrics['total'])}")
            print(f"  Runtime:  {meta.get('runtime_seconds', 'unknown')}s")
        print(f"  Total:    {metrics['total']}")
        print(f"  Passed:   {metrics['passed']}")
        print(f"  Failed:   {metrics['failed']}")
        print(f"  Pass rate: {metrics['pass_rate']:.2%}")

    # Save combined summary
    summary_path = "bigcodebench_eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved detailed summary to {summary_path}")


if __name__ == "__main__":
    main()
