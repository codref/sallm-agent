"""E2E: ingest a multi-turn prompt script against local Ollama."""

from __future__ import annotations

from pathlib import Path
import urllib.error
import urllib.request

import pytest

from sallm import Agent
from sallm.cli.chat import iter_prompts
from sallm.messages import DEFAULT_API_BASE, DEFAULT_MODEL
from sallm.tools import builtin_tools

FIXTURE = Path(__file__).parent / "fixtures" / "sample_conversation.txt"


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


def test_e2e_script_conversation():
    prompts = list(iter_prompts(FIXTURE))
    assert len(prompts) >= 2

    agent = Agent(
        model=DEFAULT_MODEL,
        tools=builtin_tools(("echo", "calc")),
        max_steps=4,
    )

    blobs = []
    for line in prompts:
        result = agent.ask(line)
        blobs.append(
            _action_observations(result) + "\n" + (result.get("answer") or "")
        )

    joined = "\n".join(blobs)
    assert "1024" in joined
    assert "done" in joined.lower()
