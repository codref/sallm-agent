"""Unit tests for chunk_text."""

from __future__ import annotations

from sallm.chunk import chunk_text


def test_chunk_short_passthrough():
    assert chunk_text("hello world") == ["hello world"]


def test_chunk_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_long_splits_with_overlap():
    # ~200 estimated tokens at 4 chars/token
    text = ("alpha line\n" * 40) + ("beta line\n" * 40)
    chunks = chunk_text(text, max_tokens=50, overlap_tokens=10)
    assert len(chunks) >= 2
    # Overlap: some content from an early chunk should reappear
    joined = "\n".join(chunks)
    assert "alpha" in joined and "beta" in joined
