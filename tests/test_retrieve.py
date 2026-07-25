"""Unit tests for CompactAndRetrieve facade."""

from __future__ import annotations

from sallm.context import MaxMessages, estimate_tokens
from sallm.messages import assistant, system, user
from sallm.retrieve import (
    OFFLOAD_STUB,
    RETRIEVED_PREFIX,
    CompactAndRetrieve,
    SCOPE_ALL,
    SCOPE_SESSION,
)
from sallm.store import InMemoryStore


def test_prepare_pushes_overflow_and_injects_hits():
    store = InMemoryStore()
    facade = CompactAndRetrieve(
        MaxMessages(2),
        store,
        session_id="s1",
        k=3,
        chunk_tokens=512,
    )
    msgs = [
        system("sys"),
        user("remember unique-zebra fact one"),
        assistant("ok"),
        user("filler"),
        assistant("ok"),
        user("what about unique-zebra?"),
    ]
    view = facade.prepare(msgs)
    assert any(RETRIEVED_PREFIX in (m.get("content") or "") for m in view)
    assert view[0]["role"] == "system"
    # Overflow was stored under session
    assert store.query("unique-zebra", k=3, session_id="s1")


def test_offload_oversized_in_view():
    store = InMemoryStore()
    facade = CompactAndRetrieve(
        MaxMessages(40),
        store,
        session_id="s1",
        k=2,
        chunk_tokens=20,  # small so short*N exceeds
        chunk_overlap_tokens=4,
    )
    big = "word " * 200
    assert estimate_tokens(big) > 20
    msgs = [system("sys"), user(big)]
    view = facade.prepare(msgs)
    assert view[0]["role"] == "system"
    blob = "\n".join(m.get("content") or "" for m in view)
    assert OFFLOAD_STUB.split(";")[0] in blob
    assert store.query("word", k=2, session_id="s1")


def test_memory_scope_session_vs_all():
    store = InMemoryStore()
    store.add("old-session unique-mango", session_id="other")
    facade = CompactAndRetrieve(
        MaxMessages(40),
        store,
        session_id="mine",
        k=3,
        chunk_tokens=512,
        memory_scope=SCOPE_SESSION,
    )
    # Push something for current session via overflow
    msgs = [
        system("sys"),
        user("mine unique-kiwi"),
        assistant("ok"),
        user("ask unique-kiwi"),
    ]
    facade.compactor = MaxMessages(2)
    view = facade.prepare(msgs)
    blob = "\n".join(m.get("content") or "" for m in view)
    assert "unique-kiwi" in blob or RETRIEVED_PREFIX in blob
    assert "unique-mango" not in blob

    facade.memory_scope = SCOPE_ALL
    view2 = facade.prepare(
        [system("sys"), user("tell me about unique-mango")]
    )
    blob2 = "\n".join(m.get("content") or "" for m in view2)
    assert "unique-mango" in blob2


def test_on_clear_drops_session_rows():
    store = InMemoryStore()
    facade = CompactAndRetrieve(
        MaxMessages(2),
        store,
        session_id="s1",
        k=2,
    )
    store.add("secret", session_id="s1")
    store.add("other", session_id="s2")
    facade.on_clear()
    assert store.query("secret", k=2, session_id="s1") == []
    assert store.query("other", k=2, session_id="s2")
