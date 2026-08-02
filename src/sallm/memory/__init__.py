"""Vector memory: chunking, embedding, LanceDB store, retrieval helpers."""

from .chunking import TokenChunker, chunk_text
from .config import (
    RetrievalConfig,
    parse_retrieval_query,
    parse_search_mode,
    resolve_retrieval_config,
)
from .embedding import LiteLLMEmbedder, make_embed_fn
from .gates import HeuristicMemoryGate, MemoryGate, PassThroughGate
from .index import MemoryIndexer, chunk_id
from .lance import LanceVectorStore
from .query import HyDE, RetrieveResult, retrieve_hits
from .retrieval import DefaultQueryComposer
from .types import (
    Chunker,
    Embedder,
    QueryComposer,
    VectorHit,
    VectorQuery,
    VectorRecord,
    VectorStore,
)

__all__ = [
    "Chunker",
    "DefaultQueryComposer",
    "Embedder",
    "HeuristicMemoryGate",
    "HyDE",
    "LanceVectorStore",
    "LiteLLMEmbedder",
    "MemoryGate",
    "MemoryIndexer",
    "PassThroughGate",
    "QueryComposer",
    "RetrieveResult",
    "RetrievalConfig",
    "TokenChunker",
    "VectorHit",
    "VectorQuery",
    "VectorRecord",
    "VectorStore",
    "chunk_id",
    "chunk_text",
    "make_embed_fn",
    "parse_retrieval_query",
    "parse_search_mode",
    "resolve_retrieval_config",
    "retrieve_hits",
]
