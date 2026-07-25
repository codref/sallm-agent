"""End-to-end agent tests against local Ollama (default model)."""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from sallm import Agent
from sallm.cli.tools import CHAT_TOOLS
from sallm.messages import DEFAULT_API_BASE, DEFAULT_MODEL


def _ollama_up() -> bool:
    try:
        with urllib.request.urlopen(DEFAULT_API_BASE, timeout=2) as resp:
            return 200 <= resp.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_up(),
    reason=f"Ollama not reachable at {DEFAULT_API_BASE}",
)


def _action_observations(result: dict) -> str:
    parts = []
    for step in result.get("steps") or []:
        if step.get("kind") != "action":
            continue
        for tc in step.get("tool_calls") or []:
            parts.append(str(tc.get("observation") or ""))
        if step.get("observation"):
            parts.append(str(step["observation"]))
    return "\n".join(parts)


def test_e2e_calc_power():
    agent = Agent(model=DEFAULT_MODEL, tools={"calc": CHAT_TOOLS["calc"]}, max_steps=4)
    result = agent.ask(
        "Use the calc tool exactly once. Put this in a ```run block:\n"
        "calc --expression \"2**10\"\n"
        "Then tell me the numeric result in plain text."
    )
    blob = _action_observations(result) + "\n" + (result.get("answer") or "")
    assert "1024" in blob


def test_e2e_multi_tool():
    agent = Agent(
        model=DEFAULT_MODEL,
        tools={
            "calc": CHAT_TOOLS["calc"],
            "echo": CHAT_TOOLS["echo"],
        },
        max_steps=4,
    )
    result = agent.ask(
        "In one reply, emit a single ```run block with exactly these two lines "
        "(nothing else in the block):\n"
        "calc --expression \"3+4\"\n"
        "echo --text ping\n"
        "After you see the tool results, summarize them briefly."
    )
    actions = [s for s in (result.get("steps") or []) if s.get("kind") == "action"]
    assert actions, f"expected a tool action, got steps={result.get('steps')}"
    all_names = {
        tc.get("action")
        for s in actions
        for tc in (s.get("tool_calls") or [])
    }
    assert "calc" in all_names
    assert "echo" in all_names
    obs = _action_observations(result)
    assert "7" in obs
    assert "ping" in obs


def test_e2e_help_then_calc():
    agent = Agent(model=DEFAULT_MODEL, tools={"calc": CHAT_TOOLS["calc"]}, max_steps=5)
    result = agent.ask(
        "First run `calc --help` in a ```run block. "
        "Then run calc with --expression \"1+1\". "
        "Finally answer with the number only."
    )
    obs = _action_observations(result) + "\n" + (result.get("answer") or "")
    assert "--expression" in obs or "2" in obs
