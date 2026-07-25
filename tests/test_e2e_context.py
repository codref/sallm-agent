"""E2E: context optimizers against local Ollama."""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from sallm import Agent
from sallm.context import SUMMARY_PREFIX, MaxMessages, SummarizeOverflow
from sallm.llm import complete
from sallm.messages import DEFAULT_API_BASE, DEFAULT_MODEL, assistant, user


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


def test_e2e_max_messages():
    agent = Agent(
        model=DEFAULT_MODEL,
        tools={},
        context=MaxMessages(4),
        max_steps=2,
    )
    for i in range(5):
        agent.ask(f"Reply with exactly one word: ok{i}")

    result = agent.ask("Reply with exactly one word: done")
    metrics = result.get("metrics") or {}
    assert metrics["context_messages"] > metrics["prompt_messages"], metrics
    answer = (result.get("answer") or "").lower()
    assert answer.strip(), "expected a non-empty late-turn answer"
    assert "done" in answer or len(answer) > 0


def test_e2e_summarize_overflow():
    def summarize_fn(text: str) -> str:
        result = complete(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Summarize the conversation briefly. "
                        "Keep all key facts and codes.\n\n" + text
                    ),
                }
            ],
            api_base=DEFAULT_API_BASE,
        )
        return result.get("content") or ""

    opt = SummarizeOverflow(threshold=40, keep_last=2, summarize_fn=summarize_fn)
    agent = Agent(
        model=DEFAULT_MODEL,
        tools={},
        context=opt,
        max_steps=2,
    )

    agent.messages.append(user("Remember: the secret code is PURPLE-42."))
    agent.messages.append(assistant("Understood. The secret code is PURPLE-42."))
    for i in range(4):
        agent.messages.append(user("filler " + ("x" * 120) + f" turn-{i}"))
        agent.messages.append(assistant("ack " + ("y" * 120)))

    full_len = len(agent.messages)
    view = opt.prepare(agent.messages)
    assert len(agent.messages) == full_len
    assert any(SUMMARY_PREFIX in (m.get("content") or "") for m in view), view
    assert len(view) < full_len

    result = agent.ask("What is the secret code? Reply with the code only.")
    metrics = result.get("metrics") or {}
    assert metrics["context_messages"] >= full_len + 1
    assert metrics["prompt_messages"] < metrics["context_messages"], metrics
    blob = (result.get("answer") or "").upper()
    assert "PURPLE" in blob or "42" in blob, result.get("answer")
