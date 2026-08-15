"""LiteLLM completion wrapper — plain chat (no native tool schemas)."""

import time

from litellm import completion

from .messages import DEFAULT_API_BASE, DEFAULT_MODEL
from .models import think_on

_LENGTH_STOPS = {"length", "max_tokens", "length_cutoff"}


class ThinkingTruncated(Exception):
    """Thinking used the output budget and left no user-visible reply."""


def _ollama_chat_model(model: str) -> str:
    """Prefer Ollama /api/chat when we need think= (generate is unreliable).

    LiteLLM ``ollama/…`` hits ``/api/generate`` with a generic prompt template.
    ``ollama_chat/…`` hits ``/api/chat``, which honors top-level ``think`` and
    uses the model's real chat template (needed for Qwen3.5).
    """
    if model.startswith("ollama/") and not model.startswith("ollama_chat/"):
        return "ollama_chat/" + model[len("ollama/") :]
    return model


def _reasoning_text(choice) -> str:
    raw = (
        getattr(choice, "reasoning_content", None)
        or getattr(choice, "reasoning", None)
        or getattr(choice, "thinking", None)
        or ""
    )
    return raw if isinstance(raw, str) else (str(raw) if raw else "")


def _reasoning_token_count(usage_obj, reasoning: str) -> int:
    details = getattr(usage_obj, "completion_tokens_details", None)
    n = getattr(details, "reasoning_tokens", None) if details else None
    if n is not None:
        return int(n)
    return len(reasoning) // 4


def complete(model=None, messages=None, api_base=None, **kwargs):
    """Call litellm and return a plain dict with content, usage, timing."""
    model = model or DEFAULT_MODEL
    api_base = api_base or DEFAULT_API_BASE
    messages = messages or []

    if "think" in kwargs:
        model = _ollama_chat_model(model)

    started = time.perf_counter()
    response = completion(
        model=model,
        messages=messages,
        api_base=api_base,
        **kwargs,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    choice = response.choices[0].message
    content = choice.content or ""
    reasoning = _reasoning_text(choice)
    finish_reason = getattr(response.choices[0], "finish_reason", None)

    usage_obj = getattr(response, "usage", None)
    reasoning_tokens = _reasoning_token_count(usage_obj, reasoning)
    usage = {
        "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
        "reasoning_tokens": reasoning_tokens,
    }

    # think= on + no visible reply: the trace ate max_tokens (or the
    # provider omitted finish_reason). JSON tasks force think=False.
    empty = not str(content).strip()
    cut = finish_reason in _LENGTH_STOPS or bool(reasoning.strip())
    if think_on(kwargs.get("think")) and empty and cut:
        cap = kwargs.get("max_tokens")
        raise ThinkingTruncated(
            f"thinking used the output budget (max_tokens={cap}) and left "
            "no reply; raise max_output_tokens or set think to false/low"
        )

    return {
        "content": content,
        "reasoning": reasoning or None,
        "finish_reason": finish_reason,
        "usage": usage,
        "elapsed_ms": elapsed_ms,
        "raw": response,
    }
