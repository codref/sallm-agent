"""LanceStore tests — skipped if lancedb extra not installed."""

from __future__ import annotations

import hashlib

import pytest

lancedb = pytest.importorskip("lancedb")

from sallm.lance_store import LanceStore


def _fake_embed(dims: int):
    def embed(text: str):
        h = hashlib.sha256((text or "").encode("utf-8")).digest()
        vals = [((h[i % len(h)] / 255.0) * 2 - 1) for i in range(dims)]
        return vals

    return embed


def test_lance_roundtrip(tmp_path):
    dims = 8
    store = LanceStore(tmp_path / "db", embed_fn=_fake_embed(dims), dimensions=dims)
    store.add("purple secret code", session_id="s1")
    store.add("other session banana", session_id="s2")
    hits = store.query("purple secret", k=2, session_id="s1")
    assert hits
    assert any("purple" in h for h in hits)
    assert not any("banana" in h for h in hits)

    store.clear(session_id="s1")
    assert store.query("purple", k=2, session_id="s1") == []
    assert store.query("banana", k=2, session_id="s2")
