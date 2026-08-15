"""ReAct forwards ModelProfile.think (no Ollama required)."""

from unittest.mock import patch

from sallm import Agent
from sallm.models import ModelProfile, resolve_model_profile


def _fake_llm(content="ok"):
    return {
        "content": content,
        "reasoning": None,
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "elapsed_ms": 1.0,
    }


def test_react_passes_think_false():
    """When profile.think is set, legacy ReAct must forward it to complete()."""
    profile = resolve_model_profile("ollama/qwen3.5:0.8b", think=False)
    assert profile.think is False

    with patch("sallm.legacy_ask.complete", return_value=_fake_llm()) as mocked:
        agent = Agent(tools={}, profile=profile, model=profile.model)
        agent.ask("hi")
        assert mocked.called
        assert mocked.call_args.kwargs.get("think") is False
        assert mocked.call_args.kwargs.get("max_tokens") == profile.max_output_tokens


def test_react_forwards_temperature():
    profile = resolve_model_profile("ollama/qwen3.5:0.8b", think=False, temperature=0.2)
    with patch("sallm.legacy_ask.complete", return_value=_fake_llm()) as mocked:
        agent = Agent(tools={}, profile=profile, model=profile.model)
        agent.ask("hi")
        assert mocked.call_args.kwargs.get("temperature") == 0.2


def test_llm_complete_upgrades_ollama_to_chat_when_think_set():
    """think= must use ollama_chat/ so Ollama /api/chat gets the flag."""
    from sallm.llm import _ollama_chat_model

    assert _ollama_chat_model("ollama/qwen3.5:0.8b") == "ollama_chat/qwen3.5:0.8b"
    assert _ollama_chat_model("ollama_chat/qwen3.5:0.8b") == "ollama_chat/qwen3.5:0.8b"
    # Non-ollama ids unchanged
    assert _ollama_chat_model("openai/gpt-4o") == "openai/gpt-4o"

    with patch("sallm.llm.completion", return_value=_fake_litellm_response()) as mocked:
        from sallm.llm import complete

        complete(
            model="ollama/qwen3.5:0.8b",
            messages=[{"role": "user", "content": "hi"}],
            think=False,
        )
        assert mocked.call_args.kwargs["model"] == "ollama_chat/qwen3.5:0.8b"
        assert mocked.call_args.kwargs["think"] is False


def _fake_litellm_response(content="ok", reasoning=None, finish_reason="stop"):
    """Minimal litellm-shaped object for complete() without a network call."""

    class _Msg:
        def __init__(self):
            self.content = content
            self.reasoning_content = reasoning
            self.reasoning = None
            self.thinking = None

    class _Choice:
        def __init__(self):
            self.message = _Msg()
            self.finish_reason = finish_reason

    class _Usage:
        prompt_tokens = 1
        completion_tokens = 1
        total_tokens = 2

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()

    return _Resp()


def test_react_forwards_think_level():
    profile = resolve_model_profile("ollama/qwen3.5:0.8b", think="low")
    with patch("sallm.legacy_ask.complete", return_value=_fake_llm()) as mocked:
        agent = Agent(tools={}, profile=profile, model=profile.model)
        agent.ask("hi")
        assert mocked.call_args.kwargs.get("think") == "low"


def test_complete_refuses_thinking_without_reply():
    from sallm.llm import ThinkingTruncated, complete

    fake = _fake_litellm_response(
        content="", reasoning="Okay, let's see", finish_reason="length"
    )
    with patch("sallm.llm.completion", return_value=fake):
        try:
            complete(
                model="ollama/lfm2.5-thinking",
                messages=[{"role": "user", "content": "hi"}],
                think=True,
                max_tokens=512,
            )
        except ThinkingTruncated as exc:
            assert "max_tokens=512" in str(exc)
        else:
            raise AssertionError("expected ThinkingTruncated")


def test_complete_keeps_reply_when_thinking_and_content():
    from sallm.llm import complete

    fake = _fake_litellm_response(
        content="ORANGE-19", reasoning="remember the code", finish_reason="stop"
    )
    with patch("sallm.llm.completion", return_value=fake):
        out = complete(
            model="ollama/lfm2.5-thinking",
            messages=[{"role": "user", "content": "hi"}],
            think="low",
        )
    assert out["content"] == "ORANGE-19"
    assert out["reasoning"] == "remember the code"
    assert out["usage"]["reasoning_tokens"] == len("remember the code") // 4


def test_metrics_count_thinking_expense():
    from sallm.metrics import from_llm_result, summarize

    usage = from_llm_result(
        {
            "content": "hi",
            "reasoning": "abcd" * 4,
            "usage": {"prompt_tokens": 1, "completion_tokens": 10, "total_tokens": 11},
            "elapsed_ms": 2.0,
        }
    )
    assert usage["reasoning_chars"] == 16
    assert usage["reasoning_tokens"] == 4
    summary = summarize(usage, context_messages=2)
    assert summary["reasoning_tokens"] == 4
    assert summary["reasoning_chars"] == 16


def test_react_omits_think_by_default():
    """Gemma default profile leaves think unset so complete() gets no think kwarg."""
    assert ModelProfile().think is None

    with patch("sallm.legacy_ask.complete", return_value=_fake_llm()) as mocked:
        agent = Agent(tools={})
        agent.ask("hi")
        assert "think" not in mocked.call_args.kwargs
        assert mocked.call_args.kwargs.get("max_tokens") == ModelProfile().max_output_tokens
