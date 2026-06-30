"""Minimal ReAct agent using the shared tool-augmented solver."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import tool_loop_solve


SYSTEM_PROMPT = (
    "You are a ReAct agent solving coding and reasoning questions. "
    "Reason step by step and use tools only when they provide new, useful information. "
    "Do not loop or repeat the same search. "
    "Provide a concise final answer as soon as you are confident."
)


def react_solve(question: str, file_path: str | None = None) -> dict:
    """Run the ReAct loop and return the result dict."""
    return tool_loop_solve(
        question,
        SYSTEM_PROMPT,
        file_path=file_path,
        max_iterations=6,
    )
