"""E2E retrieve facade against local Ollama + LanceDB (both optional)."""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from sallm import Agent
from sallm.context import MaxMessages
from sallm.messages import DEFAULT_API_BASE, DEFAULT_MODEL, assistant, user
from sallm.retrieve import RETRIEVED_PREFIX, CompactAndRetrieve
from sallm.store import InMemoryStore


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


def test_e2e_retrieve_with_inmemory_store():
    """Economy + retrieve without Lance — proves facade against a real model."""
    store = InMemoryStore()
    facade = CompactAndRetrieve(
        MaxMessages(4),
        store,
        session_id="e2e",
        k=3,
        chunk_tokens=512,
    )
    agent = Agent(model=DEFAULT_MODEL, tools={}, context=facade, max_steps=2)

    agent.messages.append(user("Remember: the vault code is ORANGE-99."))
    agent.messages.append(assistant("Saved. The vault code is ORANGE-99."))
    for i in range(4):
        agent.messages.append(user(f"noise filler turn {i} " + ("x" * 40)))
        agent.messages.append(assistant("ok"))

    view = facade.prepare(agent.messages + [user("What is the vault code?")])
    assert any(RETRIEVED_PREFIX in (m.get("content") or "") for m in view) or store.query(
        "vault code", k=3, session_id="e2e"
    )

    result = agent.ask("What is the vault code? Reply with the code only.")
    blob = (result.get("answer") or "").upper()
    assert "ORANGE" in blob or "99" in blob, result.get("answer")


def test_e2e_lance_retrieve(tmp_path):
    lancedb = pytest.importorskip("lancedb")
    from litellm import embedding

    from sallm.lance_store import LanceStore

    dims = 1024
    model = "ollama/qwen3-embedding:0.6b"

    def embed_fn(text: str):
        response = embedding(
            model=model,
            input=[text or ""],
            api_base=DEFAULT_API_BASE,
        )
        data = response.data[0]
        vec = data.get("embedding") if isinstance(data, dict) else data["embedding"]
        vec = list(vec)
        if len(vec) != dims:
            pytest.skip(f"embedding dims {len(vec)} != {dims}")
        return [float(x) for x in vec]

    try:
        embed_fn("ping")
    except Exception as exc:
        pytest.skip(f"embedding model unavailable: {exc}")

    store = LanceStore(tmp_path / "lance", embed_fn=embed_fn, dimensions=dims)
    facade = CompactAndRetrieve(
        MaxMessages(4),
        store,
        session_id="e2e-lance",
        k=3,
        chunk_tokens=256,
    )
    agent = Agent(model=DEFAULT_MODEL, tools={}, context=facade, max_steps=2)

    agent.messages.append(user("Remember: the project codename is NEBULA-7."))
    agent.messages.append(assistant("Understood. Codename NEBULA-7."))
    for i in range(5):
        agent.messages.append(user(f"aside {i} " + ("y" * 50)))
        agent.messages.append(assistant("noted"))

    view = facade.prepare(
        list(agent.messages) + [user("What is the project codename?")]
    )
    assert any(RETRIEVED_PREFIX in (m.get("content") or "") for m in view)

    result = agent.ask("What is the project codename? Reply with the name only.")
    blob = (result.get("answer") or "").upper()
    assert "NEBULA" in blob or "7" in blob, result.get("answer")
