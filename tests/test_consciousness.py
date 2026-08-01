"""Unit tests for consciousness loops (no Ollama)."""

from __future__ import annotations

from sallm import Agent
from sallm.consciousness import (
    ADVICE_MARKER,
    ToolAdvisor,
    join_addenda,
    normalize_consciousness,
)
from sallm.tools import CliTool


class _StubLayer:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def advise(self, messages, tools):
        self.calls += 1
        return self.text


def test_normalize_and_join():
    assert normalize_consciousness(None) == []
    one = _StubLayer("a")
    assert normalize_consciousness(one) == [one]
    assert normalize_consciousness([one, one]) == [one, one]

    assert join_addenda([]) == ""
    assert join_addenda(["  use memory  "]) == f"{ADVICE_MARKER}\nuse memory"
    multi = join_addenda(["first", "second"])
    assert f"{ADVICE_MARKER} (1)\nfirst" in multi
    assert f"{ADVICE_MARKER} (2)\nsecond" in multi


def test_tool_advisor_none_and_directive():
    tools = {"memory": CliTool(name="memory", argv=["memory"], summary="store")}

    none_adv = ToolAdvisor(complete_fn=lambda p: "none")
    assert none_adv.advise([{"role": "user", "content": "hi"}], tools) == ""

    dir_adv = ToolAdvisor(
        complete_fn=lambda p: "Use memory search for document facts. Do not use dig."
    )
    out = dir_adv.advise(
        [{"role": "user", "content": "Where is the outbox?"}],
        tools,
    )
    assert "memory" in out.lower()
    assert "dig" in out.lower()


def test_agent_ephemeral_addendum_in_prompt(monkeypatch):
    """Consciousness text appears in the LLM prompt system, not permanently on messages[0]."""
    captured = []

    def fake_complete(*, model, messages, api_base=None):
        captured.append(messages)
        return {"content": "ok", "usage": {}}

    monkeypatch.setattr("sallm.agent.complete", fake_complete)

    layer = _StubLayer("Prefer memory search; avoid dig.")
    agent = Agent(
        tools={"memory": CliTool(name="memory", argv=["true"], summary="m")},
        consciousness=layer,
        max_steps=1,
    )
    base_before = agent.messages[0]["content"]
    assert ADVICE_MARKER not in base_before

    result = agent.ask("What did Dale say about IAS?")
    assert layer.calls == 1
    assert result.get("consciousness")
    assert ADVICE_MARKER in result["consciousness"]

    # Transcript system stays base-only.
    assert ADVICE_MARKER not in agent.messages[0]["content"]

    # Prompt seen by complete includes addendum.
    assert captured
    sys_prompt = captured[0][0]["content"]
    assert ADVICE_MARKER in sys_prompt
    assert "Prefer memory search" in sys_prompt


def test_agent_two_layers_concatenate(monkeypatch):
    monkeypatch.setattr(
        "sallm.agent.complete",
        lambda **kw: {"content": "done", "usage": {}},
    )
    a = _StubLayer("advice-a")
    b = _StubLayer("advice-b")
    agent = Agent(tools={}, consciousness=[a, b], max_steps=1)
    result = agent.ask("hi")
    blob = result.get("consciousness") or ""
    assert "advice-a" in blob
    assert "advice-b" in blob
    assert a.calls == 1 and b.calls == 1


def test_agent_empty_advice_skips_marker(monkeypatch):
    monkeypatch.setattr(
        "sallm.agent.complete",
        lambda **kw: {"content": "hi", "usage": {}},
    )
    agent = Agent(
        tools={},
        consciousness=_StubLayer(""),
        max_steps=1,
    )
    result = agent.ask("hello")
    assert not result.get("consciousness")
