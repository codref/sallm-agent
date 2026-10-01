"""E2E: drop a diagram, refer to it, then recall it after the history window.

Requires local Ollama at http://localhost:11434 with gemma4:e4b-it-qat and
qwen3-embedding:0.6b. Skips otherwise.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

import pytest

from sallm import Agent
from sallm.attachments import messages_have_images
from sallm.messages import DEFAULT_API_BASE, DEFAULT_MODEL
from sallm.models import ModelProfile, resolve_embedding_profile, resolve_model_profile


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "small_chat"
    / "fixtures"
    / "diagram.png"
)
CHAT = DEFAULT_MODEL
EMBED = "ollama/qwen3-embedding:0.6b"


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
    for item in data.get("models") or []:
        names.append(item.get("name") or "")
        names.append(item.get("model") or "")
    needle = model.split("/", 1)[-1]
    return any(needle in name or name.startswith(needle.split(":")[0]) for name in names)


pytestmark = pytest.mark.skipif(
    not FIXTURE.is_file()
    or not _ollama_up()
    or not _ollama_has("gemma4:e4b-it-qat")
    or not _ollama_has("qwen3-embedding:0.6b"),
    reason="Need Ollama on port 11434 with gemma4:e4b-it-qat and qwen3-embedding:0.6b",
)


def _colors(text: str) -> set[str]:
    low = (text or "").lower()
    return {name for name in ("red", "blue") if name in low}


def test_e2e_diagram_across_conversation(tmp_path):
    state = tmp_path / "state.db"
    vectors = tmp_path / "vectors"
    profile = replace(
        resolve_model_profile(CHAT, api_base=DEFAULT_API_BASE),
        recent_history_tokens=900,
        retrieval_tokens=800,
        image_tokens=280,
        think=False,
        max_output_tokens=256,
    )
    assert isinstance(profile, ModelProfile)
    emb = resolve_embedding_profile(EMBED, api_base=DEFAULT_API_BASE, top_k=8)
    agent = Agent(
        model=CHAT,
        api_base=DEFAULT_API_BASE,
        profile=profile,
        embedding_profile=emb,
        tools={},
        state_path=state,
        vector_path=vectors,
        session_id="e2e-vision",
        max_steps=2,
    )

    first = agent.ask(
        f"@{FIXTURE} What colors are in this diagram? Name both colors."
    )
    answer = first.get("answer") or ""
    assert _colors(answer) == {"red", "blue"}, answer
    assert messages_have_images(agent.last_prompt)
    assert any(
        (chunk.kind == "image") for chunk in agent.repo.list_chunks(agent.session_id)
    )

    second = agent.ask("Which color is on the left? Answer with one color name.")
    assert "red" in (second.get("answer") or "").lower(), second.get("answer")
    assert messages_have_images(agent.last_prompt)

    for index in range(4):
        agent.ask(
            "Ignore the diagram. Discuss clouds only. "
            + ("padding " * 80)
            + f"turn-{index}"
        )

    recall = agent.ask(
        "What did the earlier diagram look like? Name the colors you can see."
    )
    receipt = recall.get("receipt") or {}
    retrieval = next(
        section
        for section in receipt.get("sections") or []
        if section.get("name") == "retrieval"
    )
    assert messages_have_images(agent.last_prompt), receipt
    assert "image" in (retrieval.get("note") or ""), receipt
    assert _colors(recall.get("answer") or ""), recall.get("answer")
