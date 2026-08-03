def empty_usage():
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "elapsed_ms": 0.0,
    }


def add_usage(a, b):
    return {
        "prompt_tokens": a.get("prompt_tokens", 0) + b.get("prompt_tokens", 0),
        "completion_tokens": a.get("completion_tokens", 0)
        + b.get("completion_tokens", 0),
        "total_tokens": a.get("total_tokens", 0) + b.get("total_tokens", 0),
        "elapsed_ms": a.get("elapsed_ms", 0.0) + b.get("elapsed_ms", 0.0),
    }


def from_llm_result(result):
    usage = result.get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "elapsed_ms": result.get("elapsed_ms", 0.0),
    }


def summarize(metrics, context_messages=0, prompt_messages=None):
    """Flatten metrics for display.

    context_messages: len of canonical Agent.messages transcript.
    prompt_messages: len of the view sent to the LLM (after context optimizer).
    When prompt_messages is omitted, it defaults to context_messages.
    """
    if prompt_messages is None:
        prompt_messages = context_messages
    return {
        "prompt_tokens": metrics.get("prompt_tokens", 0),
        "completion_tokens": metrics.get("completion_tokens", 0),
        "total_tokens": metrics.get("total_tokens", 0),
        "elapsed_ms": round(metrics.get("elapsed_ms", 0.0), 1),
        "context_messages": context_messages,
        "prompt_messages": prompt_messages,
    }
