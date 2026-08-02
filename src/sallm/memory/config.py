"""Retrieval feature flags — stackable, CLI-wired, runtime-replaceable later."""

from __future__ import annotations

from dataclasses import dataclass, replace


_QUERY_MODES = frozenset(
    {"raw", "instruct", "rewrite", "hyde", "rewrite+hyde"}
)
_SEARCH_MODES = frozenset({"dense", "hybrid"})


@dataclass(frozen=True)
class RetrievalConfig:
    """Independent RAG knobs. Replace on the Agent to toggle at runtime later."""

    memory_gate: bool = True
    search_mode: str = "dense"  # dense | hybrid
    use_instruct: bool = True
    use_rewrite: bool = False
    use_hyde: bool = False

    @property
    def label(self) -> str:
        """Compact label for banners / legacy ``retrieval_mode`` display."""
        if not self.use_instruct and not self.use_rewrite and not self.use_hyde:
            return "raw"
        if self.use_rewrite and self.use_hyde:
            return "rewrite+hyde"
        if self.use_hyde:
            return "hyde"
        if self.use_rewrite:
            return "rewrite"
        return "instruct"


def parse_retrieval_query(mode: str) -> tuple[bool, bool, bool]:
    """Map ``raw|instruct|rewrite|hyde|rewrite+hyde`` → instruct/rewrite/hyde."""
    kind = (mode or "instruct").strip().lower()
    if kind not in _QUERY_MODES:
        raise ValueError(
            f"unknown retrieval-query {mode!r}; "
            f"choose from {', '.join(sorted(_QUERY_MODES))}"
        )
    if kind == "raw":
        return False, False, False
    if kind == "rewrite":
        return True, True, False
    if kind == "hyde":
        return True, False, True
    if kind == "rewrite+hyde":
        return True, True, True
    return True, False, False  # instruct


def parse_search_mode(mode: str) -> str:
    kind = (mode or "dense").strip().lower()
    if kind not in _SEARCH_MODES:
        raise ValueError(
            f"unknown search mode {mode!r}; "
            f"choose from {', '.join(sorted(_SEARCH_MODES))}"
        )
    return kind


def resolve_retrieval_config(
    *,
    retrieval_query: str = "instruct",
    search_mode: str = "dense",
    memory_gate: bool = True,
    base: RetrievalConfig | None = None,
) -> RetrievalConfig:
    """Build a RetrievalConfig from CLI-style flags."""
    use_instruct, use_rewrite, use_hyde = parse_retrieval_query(retrieval_query)
    search = parse_search_mode(search_mode)
    cfg = base or RetrievalConfig()
    return replace(
        cfg,
        memory_gate=bool(memory_gate),
        search_mode=search,
        use_instruct=use_instruct,
        use_rewrite=use_rewrite,
        use_hyde=use_hyde,
    )
