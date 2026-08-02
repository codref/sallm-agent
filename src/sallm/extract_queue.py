"""Deferred memory extract: run now, enqueue, or drain on miss/lazy."""

from __future__ import annotations

from typing import Callable

from . import metrics as metrics_mod
from .control import MemoryExtractor
from .memory import MemoryIndexer
from .prompt import CompiledProfile
from .state import SessionRepository

EXTRACT_MODES = frozenset({"waterfall", "queue"})


def normalize_extract_mode(mode: str | None) -> str:
    m = (mode or "waterfall").strip().lower() or "waterfall"
    if m not in EXTRACT_MODES:
        raise ValueError(
            f"unknown extract mode {mode!r}; choose from {', '.join(sorted(EXTRACT_MODES))}"
        )
    return m


def should_miss_flush(
    retrieval_query: str,
    hits: list | None,
    pending_count: int,
) -> bool:
    """True when a recall-shaped retrieve returned nothing but extracts are pending."""
    if pending_count <= 0:
        return False
    if not (retrieval_query or "").strip():
        return False
    return not (hits or [])


class ExtractScheduler:
    """Owns extract LLM + derived index; optional SQLite-backed deferral."""

    def __init__(
        self,
        *,
        repo: SessionRepository,
        extractor: MemoryExtractor,
        indexer: MemoryIndexer,
        session_id: str,
        mode: str = "waterfall",
        model: str = "",
        compiled_profile: CompiledProfile | None = None,
        trace=None,
        metrics=None,
        demos_fn: Callable[[], str] | None = None,
    ):
        self.repo = repo
        self.extractor = extractor
        self.indexer = indexer
        self.session_id = session_id
        self.mode = normalize_extract_mode(mode)
        self.model = model
        self.compiled_profile = compiled_profile
        self.trace = trace
        self.metrics = metrics
        self._demos_fn = demos_fn
        self.last_miss_flush = False
        if self.metrics is not None:
            self.metrics.set_extract_mode(self.mode)
            self.metrics.observe_extract_queue(
                self.repo.count_pending_extracts(self.session_id)
            )

    def _demos(self) -> str:
        if self._demos_fn is not None:
            return self._demos_fn() or ""
        if self.compiled_profile:
            return str(self.compiled_profile.demonstrations.get("extractor") or "")
        return ""

    def _sync_depth(self):
        if self.metrics is not None:
            self.metrics.observe_extract_queue(
                self.repo.count_pending_extracts(self.session_id)
            )

    def run(self, *, deferred: bool = False) -> tuple[dict, int]:
        """Run extract over the latest transcript window. Returns (usage, fact_count)."""
        recent = self.repo.list_messages(self.session_id)[-8:]
        valid_ids = {m.id for m in recent}
        snippet = "\n".join(f"[{m.id}] {m.role}: {m.content[:500]}" for m in recent)
        facts, ext_result = self.extractor.extract(
            transcript_snippet=snippet,
            valid_message_ids=valid_ids,
            demos=self._demos(),
        )
        usage = metrics_mod.from_llm_result(ext_result)
        if self.trace is not None:
            self.trace.llm(
                model=self.model,
                metrics=usage,
                content=ext_result.get("content") or "",
                name="extract",
                operation="extract",
            )
        for fact in facts:
            self.repo.add_derived(
                self.session_id, fact.text, fact.source_message_ids
            )
            self.indexer.add_text(
                self.session_id,
                fact.text,
                chunks=[fact.text],
                source_message_id=fact.source_message_ids[0],
                kind="derived",
            )
        n = len(facts)
        if self.metrics is not None:
            self.metrics.observe_extract(usage.get("elapsed_ms") or 0.0, facts=n)
        return usage, n

    def enqueue(self, anchor_message_id: int) -> None:
        self.repo.enqueue_extract(self.session_id, anchor_message_id)
        depth = self.repo.count_pending_extracts(self.session_id)
        if self.metrics is not None:
            self.metrics.observe_extract_queue(depth, enqueued=1)

    def drain(self, *, reason: str = "lazy") -> dict:
        """FIFO drain of pending extract jobs. Returns combined usage."""
        combined = metrics_mod.empty_usage()
        jobs = self.repo.list_pending_extracts(self.session_id)
        drained = 0
        for job in jobs:
            try:
                usage, _ = self.run(deferred=True)
                combined = metrics_mod.add_usage(combined, usage)
                self.repo.mark_extract_done(job.id)
                drained += 1
            except Exception:
                self.repo.mark_extract_failed(job.id)
                drained += 1
        depth = self.repo.count_pending_extracts(self.session_id)
        if self.metrics is not None and drained:
            self.metrics.observe_extract_queue(
                depth, drained=drained, reason=reason or "lazy"
            )
        elif self.metrics is not None:
            self._sync_depth()
        return combined

    def pending_count(self) -> int:
        return self.repo.count_pending_extracts(self.session_id)

    def after_raw(self, turn_metrics: dict, *, anchor_message_id: int) -> dict:
        """Post-raw-index hook: enqueue (queue) or run now (waterfall)."""
        if self.mode == "queue":
            self.enqueue(anchor_message_id)
            return turn_metrics
        usage, _ = self.run(deferred=False)
        return metrics_mod.add_usage(turn_metrics, usage)

    def note_miss_flush(self) -> None:
        self.last_miss_flush = True
        if self.metrics is not None:
            self.metrics.observe_extract_queue(
                self.pending_count(), miss_flush=True
            )
