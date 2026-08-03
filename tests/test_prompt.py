"""Unit tests for Prompt templates and Agent.last_prompt (no Ollama)."""

from unittest.mock import patch

from sallm import Agent, Prompt
from sallm.prompt import Prompt as PromptDirect


def test_prompt_system_includes_tools_and_policy():
    p = Prompt(tools_text="calc: math", multi_step=True)
    text = p.system()
    assert "calc: math" in text
    assert "Multi-step mode is ON" in text
    assert "```run" in text
    assert "tool-advice" not in text.lower()


def test_prompt_multi_step_off_and_extra():
    p = Prompt(tools_text="(none)", multi_step=False, extra="Be terse.")
    text = p.system()
    assert text.startswith("Be terse.")
    assert "Multi-step mode is OFF" in text


def test_prompt_preview_and_as_dict():
    p = Prompt(tools_text="echo: say", multi_step=True, extra="Hi")
    preview = p.preview()
    assert "=== Prompt preview" in preview
    assert "echo: say" in preview
    assert "CONTINUE_NUDGE" in preview
    d = p.as_dict()
    assert d["multi_step"] is True
    assert d["extra"] == "Hi"
    assert d["chars"]["system"] == len(d["system"])
    assert d["results_prefix"] == Prompt.RESULTS_PREFIX


def test_agent_uses_prompt_and_sets_last_prompt():
    fake = {
        "content": "hello",
        "reasoning": None,
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "elapsed_ms": 1.0,
    }
    with patch("sallm.legacy_ask.complete", return_value=fake) as mocked:
        agent = Agent(tools={}, system="Extra.")
        assert isinstance(agent.prompt, PromptDirect)
        assert "Extra." in agent.prompt.system()
        assert agent.last_prompt is None
        result = agent.ask("hi")
        assert result["answer"] == "hello"
        assert agent.last_prompt is not None
        assert agent.last_prompt[0]["role"] == "system"
        assert any(m.get("role") == "user" and m.get("content") == "hi" for m in agent.last_prompt)
        assert mocked.called
        agent.clear()
        assert agent.last_prompt is None


def test_package_exports_prompt_not_tool_advisor():
    import sallm

    assert hasattr(sallm, "Prompt")
    assert not hasattr(sallm, "ToolAdvisor")
