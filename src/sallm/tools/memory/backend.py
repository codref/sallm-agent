"""File- and Lance-backed stores for the memory CliTool (private)."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from sallm.context import estimate_tokens

TABLE_NAME = "chunks"
DEFAULT_CHUNK_TOKENS = 512
DEFAULT_CHUNK_OVERLAP = 64


def chunk_text(
    text: str,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP,
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


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t}


def _chunk_id(session_id: str, chunk: str) -> str:
    return hashlib.sha256(f"{session_id}\n{chunk}".encode("utf-8")).hexdigest()


class FileStore:
    """Subprocess-safe JSON store with token-overlap ranking."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._file = self.path / "chunks.json"

    def _load(self) -> dict[str, dict]:
        if not self._file.exists():
            return {}
        try:
            return json.loads(self._file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self, rows: dict[str, dict]) -> None:
        self._file.write_text(json.dumps(rows), encoding="utf-8")

    def add(self, text: str, *, id: str | None = None, session_id: str | None = None):
        text = text or ""
        if not text.strip():
            return
        sid = session_id or ""
        rows = self._load()
        row_id = id or _chunk_id(sid, text)
        if row_id in rows:
            return
        rows[row_id] = {
            "text": text,
            "session_id": sid,
            "tokens": sorted(_tokens(text)),
        }
        self._save(rows)

    def query(
        self, text: str, k: int = 5, *, session_id: str | None = None
    ) -> list[str]:
        if k < 1:
            return []
        q = _tokens(text)
        scored = []
        for row in self._load().values():
            if session_id is not None and row.get("session_id") != session_id:
                continue
            row_tokens = set(row.get("tokens") or [])
            overlap = len(q & row_tokens) if q else 0
            scored.append((overlap, row.get("text") or ""))
        scored.sort(key=lambda x: (-x[0], x[1]))
        out = [t for score, t in scored if score > 0 and t][:k]
        if out:
            return out
        return [t for _, t in scored[:k] if t]

    def clear(self, *, session_id: str | None = None):
        if session_id is None:
            if self._file.exists():
                self._file.unlink()
            return
        rows = self._load()
        keep = {
            rid: row
            for rid, row in rows.items()
            if row.get("session_id") != session_id
        }
        if keep:
            self._save(keep)
        elif self._file.exists():
            self._file.unlink()


class LanceStore:
    """LanceDB-backed vector store (requires `memory` extra)."""

    def __init__(self, path, embed_fn, dimensions: int):
        try:
            import lancedb  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "lancedb is required for Lance backend. "
                "Install with: uv sync --extra memory"
            ) from exc
        if embed_fn is None:
            raise ValueError("embed_fn is required")
        if dimensions < 1:
            raise ValueError("dimensions must be >= 1")
        self.embed_fn = embed_fn
        self.dimensions = dimensions
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._db = None
        self._known_ids: set[str] = set()
        self._load_known_ids()

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

    def _connect(self):
        import lancedb

        if self._db is None:
            self._db = lancedb.connect(str(self.path))
        return self._db

    def _table_names(self):
        db = self._connect()
        if hasattr(db, "list_tables"):
            result = db.list_tables()
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

    def add(self, text: str, *, id: str | None = None, session_id: str | None = None):
        text = text or ""
        if not text.strip():
            return
        sid = session_id or ""
        row_id = id or _chunk_id(sid, text)
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


def default_memory_path() -> Path:
    env = os.environ.get("SALLM_MEMORY_PATH")
    if env:
        return Path(env)
    return Path(os.environ.get("TMPDIR", "/tmp")) / "sallm-memory"


def default_session_id() -> str:
    return os.environ.get("SALLM_MEMORY_SESSION") or "default"


def make_embed_fn(model: str, api_base: str, dimensions: int):
    def embed(text: str) -> list[float]:
        from litellm import embedding

        response = embedding(
            model=model,
            input=[text or ""],
            api_base=api_base,
        )
        data = response.data[0]
        vec = data.get("embedding") if isinstance(data, dict) else data["embedding"]
        vec = list(vec)
        if len(vec) != dimensions:
            raise ValueError(
                f"embedding length {len(vec)} != dimensions {dimensions}"
            )
        return [float(x) for x in vec]

    return embed


def open_store(
    *,
    path: str | Path | None = None,
    backend: str = "file",
    embed_model: str | None = None,
    api_base: str | None = None,
    embed_dimensions: int = 1024,
):
    """Open a store. backend: file (default) | lance."""
    root = Path(path) if path else default_memory_path()
    kind = (backend or "file").strip().lower()
    if kind == "file":
        return FileStore(root)
    if kind == "lance":
        model = embed_model or os.environ.get(
            "SALLM_EMBEDDING_MODEL", "ollama/qwen3-embedding:0.6b"
        )
        base = api_base or os.environ.get(
            "SALLM_API_BASE", "http://127.0.0.1:11434"
        )
        dims = int(
            os.environ.get("SALLM_EMBEDDING_DIMENSIONS", str(embed_dimensions))
        )
        return LanceStore(
            root,
            embed_fn=make_embed_fn(model, base, dims),
            dimensions=dims,
        )
    raise ValueError(f"unknown memory backend {backend!r}; use file or lance")
