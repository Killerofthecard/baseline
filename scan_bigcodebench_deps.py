"""
Scan BigCodeBench-hard dataset for third-party Python dependencies.

Usage:
    source .venv/bin/activate
    python scan_bigcodebench_deps.py

Outputs:
    - bigcodebench_requirements.txt : inferred requirements
    - Prints dependency summary to stdout
"""

import json
import re
import sys


# Map common import names to their PyPI package names.
IMPORT_TO_PACKAGE = {
    "PIL": "Pillow",
    "Image": "Pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
    "yaml": "PyYAML",
    "skimage": "scikit-image",
    "matplotlib": "matplotlib",
    "mpl_toolkits": "matplotlib",
    "Crypto": "pycryptodome",
    "Cryptodome": "pycryptodomex",
    "openpyxl": "openpyxl",
    "xlsxwriter": "XlsxWriter",
    "docx": "python-docx",
    "faker": "Faker",
}


def get_stdlib_modules() -> set[str]:
    """Return a set of known standard library module names."""
    if hasattr(sys, "stdlib_module_names"):
        return set(sys.stdlib_module_names)

    # Fallback for older Python versions
    return {
        "abc", "argparse", "ast", "asyncio", "base64", "bisect", "builtins",
        "calendar", "collections", "copy", "csv", "datetime", "decimal",
        "difflib", "enum", "functools", "glob", "hashlib", "heapq", "html",
        "http", "importlib", "inspect", "io", "itertools", "json", "logging",
        "math", "mimetypes", "numbers", "operator", "os", "pathlib", "pickle",
        "platform", "pprint", "random", "re", "shutil", "socket", "sqlite3",
        "statistics", "string", "subprocess", "sys", "tempfile", "textwrap",
        "threading", "time", "timeit", "traceback", "typing", "unittest",
        "urllib", "uuid", "warnings", "xml", "zipfile",
    }


def extract_imports(code: str) -> set[str]:
    """Extract top-level module names from import statements."""
    imports = set()

    # Match: import xxx, import xxx.yyy, import xxx as zzz
    for match in re.finditer(r"^\s*import\s+([a-zA-Z_][\w.]*(?:\s*,\s*[a-zA-Z_][\w.]*)*)", code, re.MULTILINE):
        modules = match.group(1).split(",")
        for mod in modules:
            mod = mod.strip().split()[0]  # remove "as xxx"
            imports.add(mod.split(".")[0])

    # Match: from xxx import yyy
    for match in re.finditer(r"^\s*from\s+([a-zA-Z_][\w.]*)\s+import", code, re.MULTILINE):
        imports.add(match.group(1).split(".")[0])

    return imports


def is_third_party(module: str, stdlib: set[str]) -> bool:
    """Return True if module is likely a third-party package."""
    if not module or module.startswith("."):
        return False
    if module in stdlib:
        return False
    # cgi was removed from Python 3.13 but is not a third-party package
    if module in {"cgi"}:
        return False
    # Skip single-letter or obviously local modules
    if len(module) <= 1:
        return False
    return True


def main() -> None:
    dataset_path = "bigcodebench_hard.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        problems = json.load(f)

    stdlib = get_stdlib_modules()
    all_third_party = set()
    per_problem_deps: dict[str, set[str]] = {}

    for problem in problems:
        task_id = problem.get("task_id", "unknown")
        code_parts = [
            problem.get("instruct_prompt", ""),
            problem.get("code_prompt", ""),
            problem.get("test", ""),
            problem.get("canonical_solution", ""),
        ]
        combined = "\n".join(code_parts)

        imports = extract_imports(combined)
        third_party = {mod for mod in imports if is_third_party(mod, stdlib)}

        # Map import names to PyPI package names
        packages = {IMPORT_TO_PACKAGE.get(mod, mod) for mod in third_party}

        all_third_party.update(packages)
        per_problem_deps[task_id] = packages

    print(f"Scanned {len(problems)} problems.")
    print(f"Found {len(all_third_party)} unique third-party dependencies:\n")

    for dep in sorted(all_third_party):
        print(f"  {dep}")

    # Save to requirements file
    output_path = "bigcodebench_requirements.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for dep in sorted(all_third_party):
            f.write(f"{dep}\n")

    print(f"\nSaved inferred requirements to {output_path}")
    print("\nNote: This is a best-effort scan. Some imports may be local modules or "
          "optional dependencies not required for evaluation.")


if __name__ == "__main__":
    main()
