"""Unit tests for InMemoryStore session scoping."""

from __future__ import annotations

from sallm.store import InMemoryStore


def test_add_query_basic():
    store = InMemoryStore()
    store.add("the secret code is PURPLE-42", session_id="s1")
    store.add("unrelated weather notes", session_id="s1")
    hits = store.query("what is the secret code?", k=2, session_id="s1")
    assert hits
    assert any("PURPLE" in h for h in hits)


def test_session_filter():
    store = InMemoryStore()
    store.add("doc A unique-apple", session_id="a")
    store.add("doc B unique-banana", session_id="b")
    hits = store.query("unique-apple", k=5, session_id="a")
    assert any("apple" in h for h in hits)
    assert not any("banana" in h for h in hits)
    all_hits = store.query("unique", k=5, session_id=None)
    assert len(all_hits) >= 2


def test_clear_session_only():
    store = InMemoryStore()
    store.add("keep me", session_id="a")
    store.add("drop me", session_id="b")
    store.clear(session_id="b")
    assert store.query("keep", k=5, session_id="a")
    assert store.query("drop", k=5, session_id="b") == []
