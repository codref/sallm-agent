"""E2E: durable stack + Lance retrieval against local Ollama.

Requires gemma4:e4b-it-qat and qwen3-embedding:0.6b. Skips otherwise.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from sallm import Agent
from sallm.messages import DEFAULT_API_BASE, DEFAULT_MODEL
from sallm.models import resolve_embedding_profile
from sallm.receipt import compile_prompt_messages
from sallm.prompt import Prompt


def _ollama_up() -> bool:
    try:
        with urllib.request.urlopen(DEFAULT_API_BASE, timeout=2) as resp:
            return 200 <= resp.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _ollama_has(model: str) -> bool:
    try:
        with urllib.request.urlopen(
            DEFAULT_API_BASE.rstrip("/") + "/api/tags", timeout=5
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return False
    names = []
    for m in data.get("models") or []:
        names.append(m.get("name") or "")
        names.append((m.get("model") or ""))
    needle = model.split("/", 1)[-1]
    return any(needle in n or n.startswith(needle.split(":")[0]) for n in names)


CHAT = DEFAULT_MODEL
EMBED = "ollama/qwen3-embedding:0.6b"

pytestmark = pytest.mark.skipif(
    not _ollama_up() or not _ollama_has("gemma4:e4b-it-qat") or not _ollama_has("qwen3-embedding:0.6b"),
    reason="Need Ollama with gemma4:e4b-it-qat and qwen3-embedding:0.6b",
)


def test_e2e_stack_memory_restart_recall(tmp_path):
    state = tmp_path / "state.db"
    vectors = tmp_path / "vectors"
    sid = "e2e-stack"
    emb = resolve_embedding_profile(EMBED, api_base=DEFAULT_API_BASE, top_k=3)

    agent = Agent(
        model=CHAT,
        api_base=DEFAULT_API_BASE,
        tools={},
        state_path=state,
        vector_path=vectors,
        session_id=sid,
        embedding_profile=emb,
        retrieval_mode="rewrite",
        max_steps=2,
    )
    fact = "The unique lab code is ZEBRA-7711."
    agent.ask(f"Please remember this fact for later: {fact}")
    # Oversized filler turns to push the fact out of recent-history budget.
    for i in range(3):
        agent.ask(
            "Ignore prior secrets. Discuss clouds. " + ("padding " * 200) + f" turn-{i}"
        )

    # Early fact should be absent from a history-only compile under a tight budget.
    from sallm.models import ModelProfile

    tight = ModelProfile(
        model=CHAT,
        api_base=DEFAULT_API_BASE,
        prompt_budget=1200,
        recent_history_tokens=200,
        retrieval_tokens=400,
    )
    view, receipt = compile_prompt_messages(
        profile=tight,
        prompt=Prompt(tools_text="(none)"),
        recent_messages=agent.messages,
        hits=[],
    )
    hist_blob = "\n".join(m.get("content") or "" for m in view)
    assert "ZEBRA-7711" not in hist_blob or receipt.omitted_messages >= 1

    agent.close = getattr(agent, "close", lambda: None)

    agent2 = Agent(
        model=CHAT,
        api_base=DEFAULT_API_BASE,
        tools={},
        state_path=state,
        vector_path=vectors,
        session_id=sid,
        embedding_profile=emb,
        retrieval_mode="rewrite",
        max_steps=2,
    )
    assert agent2.goal or agent2.stack
    assert any(
        "ZEBRA" in (m.get("content") or "") for m in agent2.messages
    ) or agent2.repo.list_chunks(sid)

    result = agent2.ask("What is the unique lab code? Reply with the code only.")
    blob = (result.get("answer") or "").upper()
    assert "ZEBRA" in blob or "7711" in blob, result
    assert result.get("receipt") is not None
    assert result["receipt"]["total_tokens"] <= result["receipt"]["budget"] + 50
