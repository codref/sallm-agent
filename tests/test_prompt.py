"""Unit tests for Prompt templates and Agent.last_prompt (no Ollama)."""

from unittest.mock import patch

from sallm import Agent, Prompt
from sallm.prompt import Prompt as PromptDirect


def test_prompt_system_includes_tools_and_policy():
    p = Prompt(tools_text="calc: math", multi_step=True)
    text = p.system()
    assert "calc: math" in text
    assert "Multi-step mode is ON" in text
    assert "finish the rest of that user request" in text
    assert "```run" in text
    assert "tool-advice" not in text.lower()


def test_think_hint_only_when_thinking_on():
    from sallm.prompt import CompiledProfile

    compiled = CompiledProfile(
        target_model="x",
        instructions={"converse": "CONVERSE_TAIL"},
        demonstrations={},
        budgets={},
        metadata={},
    )
    hint = "Reason briefly; put the user-visible answer after thinking."
    on = Prompt(tools_text="calc: math", compiled=compiled, think="low", think_hint=hint)
    off = Prompt(tools_text="calc: math", compiled=compiled, think=False, think_hint=hint)
    assert hint in on.system()
    assert hint not in off.system()
    assert on.system().index("CONVERSE_TAIL") < on.system().index(hint)


def test_prompt_converse_appended_after_base():
    from sallm.prompt import CompiledProfile

    compiled = CompiledProfile(
        target_model="x",
        instructions={"converse": "CONVERSE_TAIL"},
        demonstrations={},
        budgets={},
        metadata={},
    )
    p = Prompt(tools_text="calc: math", compiled=compiled)
    text = p.system()
    assert "CONVERSE_TAIL" in text
    assert text.index("Available tools:") < text.index("CONVERSE_TAIL")


def test_prompt_converse_demos_appended():
    from sallm.prompt import CompiledProfile

    compiled = CompiledProfile(
        target_model="x",
        instructions={"converse": "CONVERSE_TAIL"},
        demonstrations={"converse": "User: calc then recall\nAssistant: The result is 21. ORANGE-19"},
        budgets={},
        metadata={},
    )
    text = Prompt(tools_text="calc: math", compiled=compiled).system()
    assert "CONVERSE_TAIL" in text
    assert "Examples:" in text
    assert "ORANGE-19" in text
    assert text.index("CONVERSE_TAIL") < text.index("Examples:")


def test_remaining_nudge_mentions_retrieved_memory():
    assert "[Retrieved memory]" in PromptDirect.REMAINING_NUDGE


def test_apply_budgets_overlays_known_keys():
    from sallm.models import ModelProfile
    from sallm.prompt import CompiledProfile

    compiled = CompiledProfile(
        target_model="ollama_chat/qwen3.5:0.8b",
        instructions={},
        demonstrations={},
        budgets={
            "think": "medium",
            "think_hint": "Keep the trace off the answer.",
            "temperature": 0.2,
            "prompt_budget": 2048,
            "unknown_knob": 99,
        },
        metadata={},
    )
    out = compiled.apply_budgets(ModelProfile())
    assert out.think == "medium"
    assert out.think_hint == "Keep the trace off the answer."
    assert out.temperature == 0.2
    assert out.prompt_budget == 2048
    assert not hasattr(out, "unknown_knob")


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


def test_agent_overlays_compiled_budgets():
    from sallm.prompt import CompiledProfile

    compiled = CompiledProfile(
        target_model="x",
        instructions={},
        demonstrations={},
        budgets={"think": False, "temperature": 0.3, "max_output_tokens": 64},
        metadata={},
    )
    agent = Agent(tools={}, compiled_profile=compiled)
    assert agent.profile.think is False
    assert agent.profile.temperature == 0.3
    assert agent.profile.max_output_tokens == 64


def test_package_exports_prompt_not_tool_advisor():
    import sallm

    assert hasattr(sallm, "Prompt")
    assert hasattr(sallm, "ThinkingTruncated")
    assert not hasattr(sallm, "ToolAdvisor")


def test_multi_step_tool_followup_asks_to_finish_the_rest():
    from sallm.tools import builtin_tools

    run = {
        "content": "```run\ncalc -e '3*7'\n```",
        "reasoning": None,
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "elapsed_ms": 1.0,
    }
    final = {**run, "content": "The result is 21. Lab code ORANGE-19."}
    with patch("sallm.legacy_ask.complete", side_effect=[run, final]):
        agent = Agent(tools=builtin_tools(("calc",)))
        agent.ask("What is 3 times 7? Use calc. Then remind me of my lab code.")
    blob = "\n".join(m.get("content") or "" for m in agent.messages)
    assert "If the user's last request has anything left" in blob
