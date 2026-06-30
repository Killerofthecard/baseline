"""Generic tool-augmented solver using OpenAI function calling."""

import json
import os
import time

from .llm import get_client
from .tools import TOOL_REGISTRY, calculator, list_directory, python_execute, read_file, read_image, web_search

MAX_ITERATIONS = 6

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "max_results": {"type": "integer", "description": "Maximum number of results.", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_execute",
            "description": "Execute Python code in a subprocess.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source code."},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a simple arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression."},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_image",
            "description": "Describe an image using vision.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Image file path."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory path.", "default": "."}},
                "required": [],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "web_search": web_search,
    "python_execute": python_execute,
    "calculator": calculator,
    "read_file": read_file,
    "read_image": read_image,
    "list_directory": list_directory,
}


def execute_tool_call(tool_call) -> str:
    """Execute a single tool call and return the string result."""
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    fn = TOOL_FUNCTIONS.get(function_name)
    if fn is None:
        return f"Error: unknown tool '{function_name}'."

    start = time.time()
    try:
        result = fn(**arguments)
    except Exception as e:
        result = f"Error calling {function_name}: {e}"
    elapsed = time.time() - start
    if elapsed > 1.0:
        print(f"  [slow tool] {function_name} took {elapsed:.2f}s")
    return result


def build_system_prompt(base_prompt: str, file_path: str | None = None) -> str:
    """Build the system prompt with tool instructions and final-answer format."""
    prompt = (
        f"{base_prompt}\n\n"
        "You have access to tools. Use them wisely:\n"
        "- Call a tool only when you genuinely need external information, calculation, "
        "file content, or image understanding.\n"
        "- Do NOT repeat the same or nearly-the-same search query multiple times. "
        "If a search already gave you the key fact, stop searching and use it.\n"
        "- Prefer combining reasoning in one step rather than making many tiny tool calls.\n\n"
        "IMPORTANT: As soon as you have enough information to answer, stop using tools "
        "and output the final answer wrapped in triple angle brackets like this:\n"
        "<<<your final answer here>>>\n\n"
        "Do not output the final answer in any other format. Always use <<< >>>."
    )
    if file_path:
        prompt += f"\nAn attachment is available at: {file_path}"
    return prompt


def extract_final_answer(text: str) -> str | None:
    """Extract the content inside the first <<<...>>> block."""
    import re

    match = re.search(r"<<<(.+?)>>>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def text_solve(
    question: str,
    system_prompt: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> dict:
    """Solve a question with a single direct LLM call (no tools).

    Returns a dict with keys:
        - final_answer: extracted answer string, or None
        - full_response: the full assistant message
        - tool_calls: empty list
        - reached_max_iterations: False
    """
    client = get_client()
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=120,
    )
    content = response.choices[0].message.content or ""
    final_answer = extract_final_answer(content)
    if final_answer is None:
        # Fallback: use the last non-empty line if no <<< >>> block is present.
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        final_answer = lines[-1] if lines else None

    return {
        "final_answer": final_answer,
        "full_response": content.strip(),
        "tool_calls": [],
        "reached_max_iterations": False,
    }


def tool_loop_solve(
    question: str,
    system_prompt: str,
    file_path: str | None = None,
    max_iterations: int = MAX_ITERATIONS,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    model: str | None = None,
) -> dict:
    """
    Solve a question by interleaving LLM reasoning with tool calls.

    Returns a dict with keys:
        - final_answer: extracted answer string, or None
        - full_response: concatenation of all assistant messages
        - tool_calls: list of {name, arguments, result}
        - reached_max_iterations: bool
    """
    client = get_client()
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    messages = [
        {"role": "system", "content": build_system_prompt(system_prompt, file_path)},
        {"role": "user", "content": question},
    ]

    full_response_parts = []
    tool_call_records = []

    for iteration in range(max_iterations):
        force_text = iteration == max_iterations - 1
        if force_text:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Stop using tools. Provide your final answer to the question "
                        "wrapped in triple angle brackets like <<<answer>>>."
                    ),
                }
            )
        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS if not force_text else TOOLS,
            tool_choice="none" if force_text else "auto",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=120,
        )
        llm_elapsed = time.time() - start
        if llm_elapsed > 2.0:
            print(f"  [slow llm] iteration {iteration + 1} took {llm_elapsed:.2f}s")

        message = response.choices[0].message
        messages.append(message)

        if message.content:
            full_response_parts.append(message.content)
            final_answer = extract_final_answer(message.content)
            if final_answer is not None:
                return {
                    "final_answer": final_answer,
                    "full_response": "\n".join(full_response_parts).strip(),
                    "tool_calls": tool_call_records,
                    "reached_max_iterations": False,
                }

        if not message.tool_calls:
            return {
                "final_answer": None,
                "full_response": "\n".join(full_response_parts).strip() or "(no response)",
                "tool_calls": tool_call_records,
                "reached_max_iterations": False,
            }

        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call)
            tool_call_records.append(
                {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                    "result": result,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    return {
        "final_answer": None,
        "full_response": "\n".join(full_response_parts).strip() or "(no response)",
        "tool_calls": tool_call_records,
        "reached_max_iterations": True,
    }


def save_results(results: list[dict], output_path: str = "results.json") -> None:
    """Append-friendly save: load existing results, extend, and write back."""
    existing = []
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []

    if not isinstance(existing, list):
        existing = []

    existing.extend(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} result(s) to {output_path}")
