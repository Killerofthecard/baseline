"""Shared OpenAI-compatible LLM client."""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_CLIENT: OpenAI | None = None


def get_client() -> OpenAI:
    """Return a singleton OpenAI client configured from environment variables."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
    return _CLIENT


def call_llm(
    messages,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    **kwargs,
) -> str:
    """Call the chat completion endpoint and return the assistant message content."""
    client = get_client()
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    return response.choices[0].message.content or ""
