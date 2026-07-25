"""Split big text into embed-sized pieces.

Owns: chunk_text only.
Does not own: stores, compactors, embeddings, Agent.

Uses estimate_tokens (~4 chars/token) — not a real model tokenizer.
"""

from __future__ import annotations

from .context import estimate_tokens


def chunk_text(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[str]:
    """Slice text into overlapping windows by estimated token budget."""
    if not text or not text.strip():
        return []
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be < max_tokens")

    if estimate_tokens(text) <= max_tokens:
        return [text]

    window = max_tokens * 4
    overlap = overlap_tokens * 4
    step = max(1, window - overlap)
    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + window, n)
        if end < n:
            # Prefer a newline break in the last 20% of the window.
            region_start = start + int(window * 0.8)
            nl = text.rfind("\n", region_start, end)
            if nl > start:
                end = nl + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)

    return chunks
