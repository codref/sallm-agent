"""LanceDB VectorStore — all similarity ranking stays inside Lance."""

from __future__ import annotations

from pathlib import Path

from .types import VectorHit, VectorQuery, VectorRecord

TABLE_NAME = "chunks"


class LanceVectorStore:
    """Default vector backend. Rebuildable from SQLite MemoryChunk rows."""

    def __init__(self, path: str | Path, *, dimensions: int = 1024):
        if dimensions < 1:
            raise ValueError("dimensions must be >= 1")
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        self._db = None
        self._fts_ready = False

    def _connect(self):
        import lancedb

        if self._db is None:
            self._db = lancedb.connect(str(self.path))
        return self._db

    def _table_names(self) -> list[str]:
        db = self._connect()
        if hasattr(db, "list_tables"):
            result = db.list_tables()
            if isinstance(result, list):
                return list(result)
            return list(getattr(result, "tables", result) or [])
        return list(db.table_names())

    def _table(self):
        db = self._connect()
        if TABLE_NAME not in self._table_names():
            return None
        return db.open_table(TABLE_NAME)

    def _validate(self, vector: list[float]):
        if len(vector) != self.dimensions:
            raise ValueError(
                f"vector length {len(vector)} != dimensions {self.dimensions}"
            )

    def _ensure_fts(self, table) -> None:
        """Create BM25 FTS index on ``text`` once (lazy; dense users never pay)."""
        if self._fts_ready:
            return
        try:
            from lancedb.index import FTS

            table.create_index("text", config=FTS())
        except Exception:
            try:
                table.create_fts_index("text", replace=False)
            except Exception:
                try:
                    table.create_fts_index("text")
                except Exception:
                    pass
        self._fts_ready = True

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        rows = []
        for r in records:
            self._validate(r.vector)
            rows.append(
                {
                    "id": r.id,
                    "text": r.text,
                    "vector": list(r.vector),
                    "session_id": r.session_id or "",
                    "source_id": r.source_id or "",
                    "metadata": dict(r.metadata or {}),
                }
            )
        db = self._connect()
        table = self._table()
        if table is None:
            db.create_table(TABLE_NAME, data=rows)
            self._fts_ready = False
            return
        # Idempotent: delete existing ids then add (Lance has no native upsert).
        ids = [r["id"] for r in rows]
        # Escape single quotes for Lance SQL filter.
        quoted = ", ".join("'" + i.replace("'", "''") + "'" for i in ids)
        try:
            table.delete(f"id IN ({quoted})")
        except Exception:
            pass
        table.add(rows)
        # New rows invalidate a prior FTS index; rebuild lazily on next hybrid search.
        self._fts_ready = False

    def _where_clauses(self, query: VectorQuery) -> list[str]:
        clauses = []
        if query.session_id is not None:
            safe = query.session_id.replace("'", "''")
            clauses.append(f"session_id = '{safe}'")
        if query.source_ids:
            quoted = ", ".join(
                "'" + s.replace("'", "''") + "'" for s in query.source_ids
            )
            clauses.append(f"source_id IN ({quoted})")
        return clauses

    @staticmethod
    def _hit_score(row: dict) -> float:
        # Hybrid/RRF often exposes _relevance_score; dense uses _distance.
        if row.get("_relevance_score") is not None:
            return float(row["_relevance_score"])
        if row.get("_score") is not None:
            return float(row["_score"])
        dist = row.get("_distance")
        return -float(dist) if dist is not None else 0.0

    def search(self, query: VectorQuery) -> list[VectorHit]:
        if query.k < 1:
            return []
        self._validate(query.vector)
        table = self._table()
        if table is None or table.count_rows() == 0:
            return []

        mode = (query.mode or "dense").strip().lower()
        limit = max(query.k * 4, query.k)
        clauses = self._where_clauses(query)

        if mode == "hybrid" and (query.text or "").strip():
            self._ensure_fts(table)
            text_q = (query.text or "").strip()
            try:
                search = (
                    table.search(query_type="hybrid")
                    .text(text_q)
                    .vector(query.vector)
                    .limit(limit)
                )
                try:
                    from lancedb.rerankers import RRFReranker

                    search = search.rerank(RRFReranker())
                except Exception:
                    pass
            except Exception:
                # Fall back to dense if hybrid/FTS is unavailable.
                search = table.search(query.vector).limit(limit)
        else:
            search = table.search(query.vector).limit(limit)

        if clauses:
            search = search.where(" AND ".join(clauses), prefilter=True)
        rows = search.to_list()
        hits: list[VectorHit] = []
        seen = set()
        for row in rows:
            rid = row.get("id") or ""
            text = row.get("text") or ""
            if not text or rid in seen:
                continue
            seen.add(rid)
            hits.append(
                VectorHit(
                    id=rid,
                    text=text,
                    score=self._hit_score(row),
                    session_id=row.get("session_id") or "",
                    source_id=row.get("source_id") or None,
                    metadata=dict(row.get("metadata") or {}),
                )
            )
            if len(hits) >= query.k:
                break
        return hits

    def delete_session(self, session_id: str) -> None:
        table = self._table()
        if table is None:
            return
        safe = session_id.replace("'", "''")
        try:
            table.delete(f"session_id = '{safe}'")
        except Exception:
            # Rebuild without that session if delete filter fails.
            rows = table.to_pandas().to_dict(orient="records")
            keep = [r for r in rows if r.get("session_id") != session_id]
            db = self._connect()
            db.drop_table(TABLE_NAME)
            if keep:
                db.create_table(TABLE_NAME, data=keep)
            self._fts_ready = False

    def close(self) -> None:
        self._db = None
        self._fts_ready = False
