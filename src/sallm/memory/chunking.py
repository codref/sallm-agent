"""Token-budget text chunking for agent memory indexing."""

from __future__ import annotations

from sallm.context import estimate_tokens


class TokenChunker:
    """Slice text into overlapping windows by estimated token budget."""

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64):
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be >= 0")
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be < max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        if estimate_tokens(text) <= self.max_tokens:
            return [text]

        window = self.max_tokens * 4
        overlap = self.overlap_tokens * 4
        chunks: list[str] = []
        start = 0
        n = len(text)

        while start < n:
            end = min(start + window, n)
            if end < n:
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


def chunk_text(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[str]:
    """Module-level helper (TokenChunker wrapper) for callers that prefer a function."""
    return TokenChunker(max_tokens, overlap_tokens).chunk(text)
