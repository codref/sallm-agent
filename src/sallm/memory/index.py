"""Index SQLite chunks into a VectorStore (idempotent, rebuildable)."""

from __future__ import annotations

import hashlib

from sallm.state import SessionRepository, StoredChunk

from .gates import MemoryGate, PassThroughGate
from .types import Embedder, VectorRecord, VectorStore


def chunk_id(session_id: str, text: str) -> str:
    return hashlib.sha256(f"{session_id}\n{text}".encode("utf-8")).hexdigest()


class MemoryIndexer:
    """Write canonical chunks to SQLite, then upsert embeddings into VectorStore."""

    def __init__(
        self,
        repo: SessionRepository,
        store: VectorStore,
        embedder: Embedder,
        *,
        gate: MemoryGate | None = None,
    ):
        self.repo = repo
        self.store = store
        self.embedder = embedder
        self.gate: MemoryGate = gate if gate is not None else PassThroughGate()
        self.last_gated = 0

    def add_text(
        self,
        session_id: str,
        text: str,
        *,
        chunks: list[str],
        source_message_id: int | None = None,
        kind: str = "raw",
    ) -> list[StoredChunk]:
        stored: list[StoredChunk] = []
        pending: list[VectorRecord] = []
        gated = 0
        for piece in chunks:
            if not self.gate.accept(piece, kind=kind):
                gated += 1
                continue
            cid = chunk_id(session_id, piece)
            row = self.repo.add_chunk(
                session_id,
                chunk_id=cid,
                text=piece,
                source_message_id=source_message_id,
                kind=kind,
            )
            if row is None:
                continue
            stored.append(row)
            if not row.indexed:
                vec = self.embedder.embed(piece)
                pending.append(
                    VectorRecord(
                        id=row.id,
                        text=row.text,
                        vector=vec,
                        session_id=session_id,
                        source_id=str(source_message_id)
                        if source_message_id is not None
                        else None,
                        metadata={"kind": kind},
                    )
                )
        if pending:
            self.store.upsert(pending)
            for rec in pending:
                self.repo.mark_indexed(rec.id)
        self.last_gated = gated
        return stored

    def flush_unindexed(self, session_id: str) -> int:
        """Retry chunks that were committed but never indexed (crash recovery)."""
        pending_rows = self.repo.unindexed_chunks(session_id)
        if not pending_rows:
            return 0
        records = []
        for row in pending_rows:
            vec = self.embedder.embed(row.text)
            records.append(
                VectorRecord(
                    id=row.id,
                    text=row.text,
                    vector=vec,
                    session_id=session_id,
                    source_id=str(row.source_message_id)
                    if row.source_message_id is not None
                    else None,
                    metadata={"kind": row.kind},
                )
            )
        self.store.upsert(records)
        for rec in records:
            self.repo.mark_indexed(rec.id)
        return len(records)

    def rebuild(self, session_id: str) -> int:
        """Drop session vectors and re-index every SQLite chunk."""
        self.store.delete_session(session_id)
        chunks = self.repo.list_chunks(session_id)
        if not chunks:
            return 0
        records = []
        for row in chunks:
            vec = self.embedder.embed(row.text)
            records.append(
                VectorRecord(
                    id=row.id,
                    text=row.text,
                    vector=vec,
                    session_id=session_id,
                    source_id=str(row.source_message_id)
                    if row.source_message_id is not None
                    else None,
                    metadata={"kind": row.kind},
                )
            )
        self.store.upsert(records)
        for rec in records:
            self.repo.mark_indexed(rec.id)
        return len(records)
