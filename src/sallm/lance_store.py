"""LanceDB-backed vector store.

Owns: LanceStore only (local path + embed_fn + dimensions).
Does not own: compactors, chunking, Agent, CLI.

Requires optional extra: uv sync --extra memory
"""

from __future__ import annotations

import hashlib
from pathlib import Path

TABLE_NAME = "chunks"


class LanceStore:
    """Durable add/query/clear with session_id tagging."""

    def __init__(self, path, embed_fn, dimensions: int):
        if embed_fn is None:
            raise ValueError("embed_fn is required")
        if dimensions < 1:
            raise ValueError("dimensions must be >= 1")
        try:
            import lancedb  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "lancedb is required for LanceStore. "
                "Install with: uv sync --extra memory"
            ) from exc

        self.path = Path(path)
        self.embed_fn = embed_fn
        self.dimensions = dimensions
        self.path.mkdir(parents=True, exist_ok=True)
        self._db = None
        self._known_ids: set[str] = set()
        self._load_known_ids()

    def _connect(self):
        import lancedb

        if self._db is None:
            self._db = lancedb.connect(str(self.path))
        return self._db

    def _table_names(self):
        db = self._connect()
        if hasattr(db, "list_tables"):
            result = db.list_tables()
            # list_tables may return a list or an object with .tables
            if isinstance(result, list):
                return result
            return list(getattr(result, "tables", result) or [])
        return list(db.table_names())

    def _load_known_ids(self):
        db = self._connect()
        if TABLE_NAME not in self._table_names():
            return
        table = db.open_table(TABLE_NAME)
        try:
            for row in table.to_pandas().itertuples(index=False):
                self._known_ids.add(getattr(row, "id"))
        except Exception:
            pass

    def _table(self):
        db = self._connect()
        if TABLE_NAME not in self._table_names():
            return None
        return db.open_table(TABLE_NAME)

    def _embed(self, text: str) -> list[float]:
        vec = self.embed_fn(text)
        if vec is None:
            raise ValueError("embed_fn returned None")
        vec = list(vec)
        if len(vec) != self.dimensions:
            raise ValueError(
                f"embedding length {len(vec)} != dimensions {self.dimensions}"
            )
        return [float(x) for x in vec]

    def add(self, text: str, *, id: str | None = None, session_id: str | None = None):
        text = text or ""
        if not text.strip():
            return
        sid = session_id or ""
        row_id = id or hashlib.sha256(f"{sid}\n{text}".encode("utf-8")).hexdigest()
        if row_id in self._known_ids:
            return

        vector = self._embed(text)
        row = {
            "id": row_id,
            "text": text,
            "session_id": sid,
            "vector": vector,
        }
        db = self._connect()
        table = self._table()
        if table is None:
            db.create_table(TABLE_NAME, data=[row])
        else:
            table.add([row])
        self._known_ids.add(row_id)

    def query(
        self, text: str, k: int = 5, *, session_id: str | None = None
    ) -> list[str]:
        if k < 1:
            return []
        table = self._table()
        if table is None or table.count_rows() == 0:
            return []
        vector = self._embed(text or "")
        search = table.search(vector).limit(max(k * 4, k))
        if session_id is not None:
            safe = session_id.replace("'", "''")
            search = search.where(f"session_id = '{safe}'", prefilter=True)
        rows = search.to_list()
        out = []
        seen = set()
        for row in rows:
            t = row.get("text") or ""
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t)
            if len(out) >= k:
                break
        return out

    def clear(self, *, session_id: str | None = None):
        db = self._connect()
        table = self._table()
        if table is None:
            self._known_ids.clear()
            return
        if session_id is None:
            db.drop_table(TABLE_NAME)
            self._known_ids.clear()
            return
        safe = session_id.replace("'", "''")
        try:
            table.delete(f"session_id = '{safe}'")
        except Exception:
            rows = table.to_pandas().to_dict(orient="records")
            keep = [r for r in rows if r.get("session_id") != session_id]
            db.drop_table(TABLE_NAME)
            if keep:
                db.create_table(TABLE_NAME, data=keep)
        self._known_ids.clear()
        self._load_known_ids()
