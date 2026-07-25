"""LiteLLM completion wrapper — plain chat (no native tool schemas)."""

import time

from litellm import completion

from .messages import DEFAULT_API_BASE, DEFAULT_MODEL


def complete(model=None, messages=None, api_base=None, **kwargs):
    """Call litellm and return a plain dict with content, usage, timing."""
    model = model or DEFAULT_MODEL
    api_base = api_base or DEFAULT_API_BASE
    messages = messages or []

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
    reasoning = getattr(choice, "reasoning_content", None) or getattr(
        choice, "reasoning", None
    )

    usage_obj = getattr(response, "usage", None)
    usage = {
        "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
    }

    return {
        "content": content,
        "reasoning": reasoning,
        "usage": usage,
        "elapsed_ms": elapsed_ms,
        "raw": response,
    }
