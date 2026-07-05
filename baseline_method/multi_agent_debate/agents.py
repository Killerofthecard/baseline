"""Multi-agent debate baseline with tool support."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import text_solve, tool_loop_solve


PERSONAS = [
    (
        "analyst",
        "You are a careful analyst. Use tools sparingly to verify only critical facts. "
        "Avoid repeated searches. Answer the question accurately and explain your reasoning. "
        "Output your final answer wrapped in triple angle brackets like <<<answer>>>.",
    ),
    (
        "skeptic",
        "You are a skeptical reviewer. Check the question for traps and ambiguities. "
        "Use tools only to verify claims that are not obvious. "
        "Provide your own answer and note potential issues. "
        "Output your final answer wrapped in triple angle brackets like <<<answer>>>.",
    ),
    (
        "creative",
        "You are a creative problem solver. Consider alternative interpretations, "
        "but do not waste tool calls on repeated searches. Provide an answer efficiently. "
        "Output your final answer wrapped in triple angle brackets like <<<answer>>>.",
    ),
]


def agent_answer(question: str, persona: tuple[str, str], file_path: str | None = None) -> dict:
    """Get an answer from one agent persona using tools."""
    name, system_prompt = persona
    return tool_loop_solve(
        question,
        system_prompt,
        file_path=file_path,
        max_iterations=6,
    )


def debate_solve_text(question: str, num_rounds: int = 1, model: str | None = None) -> dict:
    """Run a multi-agent debate without tools and return the moderator's result dict."""
    # Collect initial proposals from all personas.
    proposals = {
        name: text_solve(question, system_prompt, model=model)
        for name, system_prompt in PERSONAS
    }

    # Optional refinement round.
    for _ in range(num_rounds - 1):
        context = "\n\n".join(
            f"--- {name.upper()} ---\n{answer['full_response']}" for name, answer in proposals.items()
        )
        new_proposals = {}
        for name, system_prompt in PERSONAS:
            revised_prompt = (
                f"{system_prompt}\n\n"
                "You are now revising your answer after seeing other agents' proposals. "
                "Provide your revised final answer."
            )
            new_proposals[name] = text_solve(
                f"Question: {question}\n\nOther agents' answers:\n\n{context}\n\n"
                "Given these perspectives, provide your revised final answer.",
                revised_prompt,
                model=model,
            )
        proposals = new_proposals

    # Moderator synthesizes the proposals into one final answer.
    context = "\n\n".join(
        f"--- {name.upper()} ---\n{answer['full_response']}" for name, answer in proposals.items()
    )
    moderator_prompt = (
        "You are a moderator. Review the agents' answers below and produce "
        "the single best final answer to the question. "
        "Output the final answer wrapped in triple angle brackets like <<<answer>>>. "
        "Be concise."
    )
    return text_solve(
        f"Question: {question}\n\nAgent answers:\n\n{context}\n\nFinal answer:",
        moderator_prompt,
        model=model,
    )


def debate_solve(question: str, file_path: str | None = None, num_rounds: int = 1) -> dict:
    """Run a multi-agent debate with tool use and return the moderator's result dict."""
    # Collect initial proposals from all personas.
    proposals = {
        name: agent_answer(question, (name, system), file_path) for name, system in PERSONAS
    }

    # Optional refinement round: let each persona see others' answers and revise.
    for _ in range(num_rounds - 1):
        context = "\n\n".join(
            f"--- {name.upper()} ---\n{answer['full_response']}" for name, answer in proposals.items()
        )
        new_proposals = {}
        for name, system_prompt in PERSONAS:
            revised_prompt = (
                f"{system_prompt}\n\n"
                "You are now revising your answer after seeing other agents' proposals. "
                "Use tools if you need to verify or correct anything."
            )
            new_proposals[name] = tool_loop_solve(
                f"Question: {question}\n\nOther agents' answers:\n\n{context}\n\n"
                "Given these perspectives, provide your revised final answer.",
                revised_prompt,
                file_path=file_path,
                max_iterations=4,
            )
        proposals = new_proposals

    # Moderator synthesizes the proposals into one final answer.
    context = "\n\n".join(
        f"--- {name.upper()} ---\n{answer['full_response']}" for name, answer in proposals.items()
    )
    moderator_prompt = (
        "You are a moderator. Review the agents' answers below and produce "
        "the single best final answer to the question. Do not use any tools. "
        "Output the final answer wrapped in triple angle brackets like <<<answer>>>. "
        "Be concise."
    )
    return tool_loop_solve(
        f"Question: {question}\n\nAgent answers:\n\n{context}\n\nFinal answer:",
        moderator_prompt,
        file_path=file_path,
        max_iterations=2,
    )
