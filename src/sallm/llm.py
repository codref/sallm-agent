import json
import re
import time
import uuid

from litellm import completion

from .messages import DEFAULT_API_BASE, DEFAULT_MODEL

_TOOL_CALLS_JSON_RE = re.compile(
    r"\{[^{}]*\"tool_calls\"\s*:\s*\[.*\]\s*\}",
    re.DOTALL,
)


def _normalize_arguments(arguments):
    if arguments is None:
        return "{}"
    if isinstance(arguments, (dict, list)):
        return json.dumps(arguments)
    return str(arguments)


def _serialize_tool_calls(tool_calls):
    """Normalize litellm/openai tool_call objects into plain dicts."""
    if not tool_calls:
        return None
    serialized = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            serialized.append(
                {
                    "id": tc.get("id") or f"call_{uuid.uuid4()}",
                    "type": tc.get("type") or "function",
                    "function": {
                        "name": fn.get("name"),
                        "arguments": _normalize_arguments(fn.get("arguments")),
                    },
                }
            )
            continue
        fn = getattr(tc, "function", None)
        serialized.append(
            {
                "id": getattr(tc, "id", None) or f"call_{uuid.uuid4()}",
                "type": getattr(tc, "type", "function") or "function",
                "function": {
                    "name": getattr(fn, "name", None) if fn is not None else None,
                    "arguments": _normalize_arguments(
                        getattr(fn, "arguments", None) if fn is not None else "{}"
                    ),
                },
            }
        )
    return serialized


def _tool_calls_from_content(content):
    """Recover tool calls when a model dumps them as JSON text in content.

    Some Ollama models (e.g. Gemma) emit:
      {"tool_calls": [{"id": "...", "type": "function", "function": {...}}]}
    instead of populating message.tool_calls.
    """
    text = (content or "").strip()
    if not text or "tool_calls" not in text:
        return None

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _TOOL_CALLS_JSON_RE.search(text)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    if not isinstance(data, dict):
        return None
    raw = data.get("tool_calls")
    if not isinstance(raw, list) or not raw:
        return None
    return _serialize_tool_calls(raw)


def complete(model=None, messages=None, api_base=None, tools=None, **kwargs):
    """Call litellm and return a plain dict with content, tool_calls, usage, timing."""
    model = model or DEFAULT_MODEL
    api_base = api_base or DEFAULT_API_BASE
    messages = messages or []

    call_kwargs = dict(kwargs)
    if tools:
        call_kwargs["tools"] = tools

    started = time.perf_counter()
    response = completion(
        model=model,
        messages=messages,
        api_base=api_base,
        **call_kwargs,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    choice = response.choices[0].message
    content = choice.content or ""
    reasoning = getattr(choice, "reasoning_content", None) or getattr(
        choice, "reasoning", None
    )
    tool_calls = _serialize_tool_calls(getattr(choice, "tool_calls", None))

    # Fallback: tool call serialized into content (common with small local models)
    if not tool_calls:
        recovered = _tool_calls_from_content(content)
        if recovered:
            tool_calls = recovered
            content = ""

    usage_obj = getattr(response, "usage", None)
    usage = {
        "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
    }

    return {
        "content": content,
        "reasoning": reasoning,
        "tool_calls": tool_calls,
        "usage": usage,
        "elapsed_ms": elapsed_ms,
        "raw": response,
    }
