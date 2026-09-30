"""LiteLLM completion wrapper — plain chat (no native tool schemas)."""

import logging
import time

from litellm import completion

from .messages import DEFAULT_API_BASE, DEFAULT_MODEL
from .models import coerce_think, think_on

logger = logging.getLogger(__name__)

_LENGTH_STOPS = {"length", "max_tokens", "length_cutoff"}

# Profile ``think`` values that are not an Ollama flag. OpenRouter reads
# these from ``reasoning.effort``; ``none`` turns the trace off.
_REASONING_EFFORT = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "max",
}


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


def prepare_completion_kwargs(model: str, kwargs: dict) -> dict:
    """Adapt profile ``think`` to the provider that will actually run.

    Ollama keeps the ``think`` flag. OpenAI-compatible hosts (OpenRouter,
    DeepSeek) ignore that flag and think by default, which holds a
    non-streaming call until the whole trace is done. Those hosts get
    ``reasoning.effort`` instead, and ``think: false`` becomes ``none``.
    """
    out = dict(kwargs)
    if "think" not in out:
        return out
    think = out.pop("think")
    if model.startswith(("ollama/", "ollama_chat/")):
        out["think"] = think
        return out
    value = coerce_think(think)
    if value is False:
        effort = "none"
    elif value is True:
        effort = "medium"
    else:
        effort = _REASONING_EFFORT.get(str(value), "medium")
    extra = dict(out.get("extra_body") or {})
    reasoning = {"effort": effort}
    if effort == "none":
        reasoning["enabled"] = False
    extra["reasoning"] = reasoning
    out["extra_body"] = extra
    return out


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
    kwargs = prepare_completion_kwargs(model, kwargs)

    started = time.perf_counter()
    reasoning = (kwargs.get("extra_body") or {}).get("reasoning")
    logger.info(
        "llm complete start model=%s max_tokens=%s reasoning=%s",
        model,
        kwargs.get("max_tokens"),
        reasoning if reasoning is not None else kwargs.get("think"),
    )
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

    logger.info(
        "llm complete done model=%s elapsed_ms=%.0f finish=%s completion_tokens=%s",
        model,
        elapsed_ms,
        finish_reason,
        usage["completion_tokens"],
    )
    return {
        "content": content,
        "reasoning": reasoning or None,
        "finish_reason": finish_reason,
        "usage": usage,
        "elapsed_ms": elapsed_ms,
        "raw": response,
    }
