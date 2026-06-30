"""Unified basic tool set for GAIA / BigCodeBench baselines.

Tools provided:
    - web_search(query, max_results)
    - python_execute(code)
    - calculator(expression)
    - read_file(path)
    - read_image(path)
    - list_directory(path)
"""

import json
import os
import subprocess
import sys
import traceback
from functools import lru_cache
from pathlib import Path

from .llm import call_llm


@lru_cache(maxsize=256)
def _web_search_cached(query: str, max_results: int) -> str:
    """Internal cached web search implementation."""
    try:
        from ddgs import DDGS
    except ImportError as e:
        return (
            f"Error: {e}\n"
            "Please install the search dependency: pip install ddgs"
        )

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

    try:
        with DDGS(proxy=proxy, timeout=10) as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"Error during web search: {e}"

    simplified = [
        {
            "title": r.get("title", ""),
            "href": r.get("href", ""),
            "body": (r.get("body", "") or "")[:500],
        }
        for r in results
    ]
    return json.dumps(simplified, ensure_ascii=False, indent=2)


def web_search(query: str, max_results: int = 3) -> str:
    """Search the web using DuckDuckGo and return a JSON string of results."""
    return _web_search_cached(query, max_results)


def python_execute(code: str, timeout: int = 30) -> str:
    """Execute Python code in a subprocess and return stdout/stderr."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout.strip()
        error = result.stderr.strip()
        if error:
            return f"STDOUT:\n{output}\n\nSTDERR:\n{error}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: code execution timed out."
    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()}"


def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression safely."""
    allowed_names = {
        "abs": abs,
        "max": max,
        "min": min,
        "pow": pow,
        "round": round,
        "sum": sum,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def read_file(path: str, max_chars: int = 20000) -> str:
    """Read a text file and return its contents (truncated if too large)."""
    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(max_chars)
        if len(content) >= max_chars:
            content += "\n... (truncated)"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


def read_image(path: str) -> str:
    """Describe an image using the LLM vision capability, if available."""
    if not os.path.exists(path):
        return f"Error: file not found: {path}"

    try:
        import base64

        with open(path, "rb") as f:
            image_bytes = f.read()
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        ext = Path(path).suffix.lower().lstrip(".") or "png"
        mime = f"image/{ext}" if ext in {"png", "jpg", "jpeg", "gif", "webp"} else "image/png"
        data_url = f"data:{mime};base64,{encoded}"
    except Exception as e:
        return f"Error encoding image: {e}"

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Describe what you see in the image concisely.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the contents of this image. If there is text, transcribe it.",
                },
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    try:
        return call_llm(messages, max_tokens=1024)
    except Exception as e:
        return f"Error calling vision model: {e}"


def list_directory(path: str = ".") -> str:
    """List files and directories at the given path."""
    try:
        entries = os.listdir(path)
        return "\n".join(entries)
    except Exception as e:
        return f"Error listing directory: {e}"


TOOL_REGISTRY = {
    "web_search": (web_search, "Search the web using DuckDuckGo. Argument: query string."),
    "python_execute": (python_execute, "Execute Python code. Argument: source code string."),
    "calculator": (calculator, "Evaluate a math expression. Argument: expression string."),
    "read_file": (read_file, "Read a text file. Argument: file path."),
    "read_image": (read_image, "Describe an image. Argument: image file path."),
    "list_directory": (list_directory, "List directory contents. Argument: directory path."),
}


def get_tool(name: str):
    """Return the tool function registered under ``name``, or None."""
    entry = TOOL_REGISTRY.get(name)
    return entry[0] if entry else None


def list_tools() -> dict[str, str]:
    """Return a mapping from tool name to description."""
    return {name: desc for name, (_, desc) in TOOL_REGISTRY.items()}


def format_tools_for_prompt() -> str:
    """Format the tool registry as a prompt-friendly string."""
    lines = ["Available tools:"]
    for name, (_, desc) in TOOL_REGISTRY.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)
