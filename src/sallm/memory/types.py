"""Backend-neutral memory contracts — no Lance/Peewee types here."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class VectorRecord:
    id: str
    text: str
    vector: list[float]
    session_id: str = ""
    source_id: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VectorQuery:
    vector: list[float]
    k: int = 4
    session_id: str | None = None
    source_ids: list[str] | None = None
    text: str | None = None  # BM25/FTS query for hybrid search
    mode: str = "dense"  # dense | hybrid


@dataclass(frozen=True)
class VectorHit:
    id: str
    text: str
    score: float
    session_id: str = ""
    source_id: str | None = None
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    """Synchronous vector index. Implementations: LanceDB, future pgvector, fakes."""

    def upsert(self, records: list[VectorRecord]) -> None: ...

    def search(self, query: VectorQuery) -> list[VectorHit]: ...

    def delete_session(self, session_id: str) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, text: str) -> list[str]: ...


@runtime_checkable
class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...

    @property
    def dimensions(self) -> int: ...


@runtime_checkable
class QueryComposer(Protocol):
    """Prepare the string that gets embedded for search (not the document)."""

    def compose(self, query: str, *, goal: str = "", mode: str = "instruct") -> str: ...
