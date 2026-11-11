"""LiteLLM completion wrapper — plain chat (no native tool schemas)."""

import logging
import math
import time

from litellm import completion

from .attachments import messages_have_images
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

# OpenRouter allocates roughly this share of ``max_tokens`` to reasoning.
# Profile caps are treated as the *visible reply* budget; the wire cap is
# inflated so that share does not eat the reply.
_EFFORT_SHARE = {
    "minimal": 0.10,
    "low": 0.20,
    "medium": 0.50,
    "high": 0.80,
    "max": 0.95,
    "xhigh": 0.95,
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


def _bare_model_id(model: str) -> str:
    """LiteLLM id without ``openai/`` prefix or OpenRouter ``:route`` suffix."""
    name = (model or "").strip()
    if name.startswith("openai/"):
        name = name[len("openai/") :]
    return name.split(":", 1)[0].lower()


def reasoning_mandatory(model: str) -> bool:
    """True when the host rejects ``reasoning.effort=none`` / ``enabled=false``.

    Z.ai GLM endpoints on OpenRouter (and bare ``glm-*`` ids on OpenCode Zen)
    require a reasoning trace; disabling it returns HTTP 400.
    """
    bare = _bare_model_id(model)
    leaf = bare.rsplit("/", 1)[-1]
    return leaf.startswith("glm")


def _effort_for_think(think) -> str | None:
    """Map a profile ``think`` value to an OpenRouter-style effort, or None."""
    value = coerce_think(think)
    if value is None or value is False:
        return None
    if value is True:
        return "medium"
    return _REASONING_EFFORT.get(str(value), "medium")


def wire_max_tokens(reply_budget: int, effort: str | None) -> int:
    """Inflate ``max_tokens`` so ``reply_budget`` remains after reasoning.

    Providers count reasoning against the same output cap. OpenRouter's
    effort levels claim roughly a fixed share of that cap; grow the wire
    limit so the visible reply still gets ``reply_budget`` tokens.
    """
    budget = max(int(reply_budget), 0)
    if not effort or effort == "none" or budget <= 0:
        return budget
    share = _EFFORT_SHARE.get(str(effort), 0.50)
    remaining = max(1.0 - float(share), 0.05)
    return max(budget, int(math.ceil(budget / remaining)))


def prepare_completion_kwargs(model: str, kwargs: dict) -> dict:
    """Adapt profile ``think`` to the provider that will actually run.

    Ollama keeps the ``think`` flag. OpenAI-compatible hosts (OpenRouter,
    DeepSeek) ignore that flag and think by default, which holds a
    non-streaming call until the whole trace is done. Those hosts get
    ``reasoning.effort`` instead, and ``think: false`` becomes ``none``.

    Profile ``max_tokens`` is the visible-reply budget. When reasoning is
    on, the wire cap is inflated so the trace does not consume that budget.
    Models that require reasoning (GLM) never get ``effort=none``; a disable
    request is clamped to ``low`` (with the same inflation) so control/JSON
    and caption calls still leave room for the reply.
    """
    out = dict(kwargs)
    if "think" not in out:
        return out
    think = out.pop("think")
    # Preserve intent for ThinkingTruncated after ``think`` is adapted away.
    out["_think_intent"] = think
    if model.startswith(("ollama/", "ollama_chat/")):
        out["think"] = think
        effort = _effort_for_think(think)
        if effort is not None and "max_tokens" in out:
            out["max_tokens"] = wire_max_tokens(int(out["max_tokens"]), effort)
        return out
    value = coerce_think(think)
    if value is False:
        effort = "none"
    elif value is True:
        effort = "medium"
    else:
        effort = _REASONING_EFFORT.get(str(value), "medium")
    if effort == "none" and reasoning_mandatory(model):
        effort = "low"
    if effort != "none" and "max_tokens" in out:
        out["max_tokens"] = wire_max_tokens(int(out["max_tokens"]), effort)
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

    # Ollama /api/generate cannot attach a picture to one turn. /api/chat can.
    # Other providers already accept OpenAI image_url parts unchanged.
    if messages_have_images(messages) or "think" in kwargs:
        model = _ollama_chat_model(model)
    think_intent = kwargs.get("think")
    kwargs = prepare_completion_kwargs(model, kwargs)
    think_intent = kwargs.pop("_think_intent", think_intent)
    # Strip internal keys before they reach LiteLLM.
    call_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
    request_reasoning = (call_kwargs.get("extra_body") or {}).get("reasoning")
    request_effort = (
        request_reasoning.get("effort")
        if isinstance(request_reasoning, dict)
        else None
    )

    started = time.perf_counter()
    logger.info(
        "llm complete start model=%s max_tokens=%s reasoning=%s",
        model,
        call_kwargs.get("max_tokens"),
        request_reasoning if request_reasoning is not None else call_kwargs.get("think"),
    )
    response = completion(
        model=model,
        messages=messages,
        api_base=api_base,
        **call_kwargs,
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

    # Reasoning on + no visible reply: the trace ate max_tokens (or the
    # provider omitted finish_reason). JSON tasks disable think when the
    # host allows it; mandatory-reasoning models keep a minimum effort and
    # an inflated wire cap instead.
    empty = not str(content).strip()
    thinking = (
        think_on(think_intent)
        or think_on(call_kwargs.get("think"))
        or (request_effort not in (None, "none"))
    )
    cut = finish_reason in _LENGTH_STOPS or bool(reasoning.strip())
    if thinking and empty and cut:
        cap = call_kwargs.get("max_tokens")
        raise ThinkingTruncated(
            f"thinking used the output budget (max_tokens={cap}) and left "
            "no reply; raise max_output_tokens or lower reasoning effort"
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
