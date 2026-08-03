"""Agent facade — durable session wiring + ask() entrypoint.

Heavy turn logic lives in turn.py. Without state_path, behaves like the
classic in-memory ReAct agent (existing tests keep working).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from . import metrics as metrics_mod
from .control import Controller, MemoryExtractor
from .extract_queue import ExtractScheduler, normalize_extract_mode, should_miss_flush
from .legacy_ask import ask_legacy
from .memory import (
    DefaultQueryComposer,
    HeuristicMemoryGate,
    HyDE,
    LanceVectorStore,
    LiteLLMEmbedder,
    MemoryIndexer,
    PassThroughGate,
    RetrievalConfig,
    TokenChunker,
    resolve_retrieval_config,
    retrieve_hits,
)
from .messages import DEFAULT_API_BASE, DEFAULT_MODEL, assistant, system, user
from .models import (
    EmbeddingProfile,
    ModelProfile,
    resolve_embedding_profile,
    resolve_model_profile,
)
from .prompt import CompiledProfile, Prompt
from .receipt import ContextReceipt
from .skills import SkillRegistry
from .state import SessionRepository
from .tools import normalize_registry, tool_descriptions
from .turn import TurnRunner, apply_stack_decision


class Agent:
    def __init__(
        self,
        model=None,
        api_base=None,
        tools=None,
        system=None,
        max_steps=5,
        multi_step=True,
        trace=None,
        context=None,
        *,
        session_id: str | None = None,
        state_path: str | Path | None = None,
        vector_path: str | Path | None = None,
        vector_store=None,
        profile: ModelProfile | None = None,
        embedding_profile: EmbeddingProfile | None = None,
        skills: SkillRegistry | None = None,
        compiled_profile: CompiledProfile | None = None,
        retrieval: RetrievalConfig | None = None,
        retrieval_mode: str | None = None,
        search_mode: str | None = None,
        memory_gate: bool | None = None,
        extract_mode: str | None = None,
        embedder=None,
    ):
        self.model = model or DEFAULT_MODEL
        self.api_base = api_base or DEFAULT_API_BASE
        self.tools = normalize_registry(tools)
        self.max_steps = max_steps
        self.multi_step = multi_step
        self.system = system
        self.trace = trace
        self.context = context
        self.profile = profile or resolve_model_profile(
            self.model, api_base=self.api_base
        )
        if api_base:
            self.profile = replace(
                self.profile, api_base=api_base, model=self.model
            )
        self.embedding_profile = embedding_profile or resolve_embedding_profile()
        self.skills = skills or SkillRegistry()
        self.compiled_profile = compiled_profile
        self.extract_mode = normalize_extract_mode(extract_mode)

        if retrieval is not None:
            self.retrieval = retrieval
        else:
            self.retrieval = resolve_retrieval_config(
                retrieval_query=retrieval_mode or "instruct",
                search_mode=search_mode or "dense",
                memory_gate=True if memory_gate is None else memory_gate,
            )
        # Legacy alias for banners / older callers.
        self.retrieval_mode = self.retrieval.label

        self.session_id = session_id or "default"
        self.last_prompt = None
        self.last_receipt: ContextReceipt | None = None
        self._last_hits = []
        self._last_gated = 0
        self._last_miss_flush = False
        self.messages = []

        self.repo: SessionRepository | None = None
        self.vector_store = vector_store
        self.embedder = embedder
        self.indexer: MemoryIndexer | None = None
        self.hyde: HyDE | None = None
        self.extracts: ExtractScheduler | None = None
        self.chunker = TokenChunker(
            self.embedding_profile.chunk_tokens,
            self.embedding_profile.chunk_overlap,
        )
        self.composer = DefaultQueryComposer(self.embedding_profile)

        if state_path is not None:
            self.repo = SessionRepository(state_path)
            self.repo.ensure_session(self.session_id, default_skill="converse")
            if self.vector_store is None:
                vpath = vector_path or (Path(state_path).parent / "vectors")
                self.vector_store = LanceVectorStore(
                    vpath, dimensions=self.embedding_profile.dimensions
                )
            if self.embedder is None:
                self.embedder = LiteLLMEmbedder(self.embedding_profile)
            gate = (
                HeuristicMemoryGate()
                if self.retrieval.memory_gate
                else PassThroughGate()
            )
            self.indexer = MemoryIndexer(
                self.repo, self.vector_store, self.embedder, gate=gate
            )
            self.indexer.flush_unindexed(self.session_id)
            self.hyde = HyDE(self.profile)
            self._load_transcript()

        ctrl_i = None
        ext_i = None
        if compiled_profile:
            ctrl_i = compiled_profile.instructions.get("controller")
            ext_i = compiled_profile.instructions.get("extractor")
        self.controller = Controller(self.profile, instruction=ctrl_i)
        self.extractor = MemoryExtractor(self.profile, instruction=ext_i)
        if self.repo is not None and self.indexer is not None:
            session_metrics = getattr(self.trace, "metrics", None) if self.trace else None
            self.extracts = ExtractScheduler(
                repo=self.repo,
                extractor=self.extractor,
                indexer=self.indexer,
                session_id=self.session_id,
                mode=self.extract_mode,
                model=self.model,
                compiled_profile=self.compiled_profile,
                trace=self.trace,
                metrics=session_metrics,
            )
        self.prompt = self._build_prompt()
        self._ensure_system()
        self._turns = TurnRunner(self)

    def _build_prompt(self) -> Prompt:
        active = "converse"
        goal = ""
        skill_prompt = ""
        tools = self.tools
        if self.repo is not None:
            active = self.repo.active_skill(self.session_id)
            goal = self.repo.get_goal(self.session_id)
            skill_prompt = self.skills.get(active).prompt
            tools = self.skills.resolve_tools(active, self.tools)
        return Prompt(
            tools_text=tool_descriptions(tools),
            multi_step=self.multi_step,
            extra=self.system,
            skill_prompt=skill_prompt,
            goal=goal,
            compiled=self.compiled_profile,
        )

    def _ensure_system(self):
        self.prompt = self._build_prompt()
        content = self.prompt.system()
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = system(content)
        else:
            self.messages.insert(0, system(content))

    def _load_transcript(self):
        if self.repo is None:
            return
        loaded = []
        for m in self.repo.list_messages(self.session_id):
            if m.role == "assistant":
                loaded.append(assistant(m.content))
            elif m.role == "system":
                loaded.append(system(m.content))
            else:
                loaded.append(user(m.content))
        self.messages = loaded

    def clear(self):
        self.messages = []
        self.last_prompt = None
        self.last_receipt = None
        self._last_hits = []
        self._last_gated = 0
        self._last_miss_flush = False
        if self.repo is not None:
            self.repo.clear_session(self.session_id, default_skill="converse")
            if self.vector_store is not None:
                self.vector_store.delete_session(self.session_id)
            if self.extracts is not None and self.extracts.metrics is not None:
                self.extracts.metrics.observe_extract_queue(0)
        self._ensure_system()
        ctx = self.context
        if ctx is not None:
            on_clear = getattr(ctx, "on_clear", None)
            if on_clear is not None:
                on_clear()

    @property
    def goal(self) -> str:
        return self.repo.get_goal(self.session_id) if self.repo else ""

    @property
    def stack(self):
        return self.repo.stack(self.session_id) if self.repo else []

    def _prompt_messages(self, messages):
        view = list(messages)
        if self.context is None:
            return view
        return self.context.prepare(view)

    def _retrieve(self, user_text: str, retrieval_query: str):
        """Vector retrieve; returns (hits, hyde_result_or_None)."""
        if self.vector_store is None or self.embedder is None:
            return [], None
        result = retrieve_hits(
            store=self.vector_store,
            embedder=self.embedder,
            composer=self.composer,
            session_id=self.session_id,
            user_text=user_text,
            goal=self.repo.get_goal(self.session_id) if self.repo else "",
            config=self.retrieval,
            top_k=self.embedding_profile.top_k,
            retrieval_query=retrieval_query or "",
            hyde=self.hyde if self.retrieval.use_hyde else None,
        )
        return result.hits, result.hyde_result

    def _finish(self, answer, steps, turn_metrics, stopped=None, *, control_decision=None):
        self.messages.append(assistant(answer))
        if self.repo is not None:
            self.repo.append_message(
                self.session_id, role="assistant", content=answer, kind="chat"
            )
        prompt_view = self.last_prompt or self._prompt_messages(self.messages)
        summary = metrics_mod.summarize(
            turn_metrics,
            context_messages=len(self.messages),
            prompt_messages=len(prompt_view),
        )
        stack_frames = [
            {"skill": f.skill, "depth": f.depth, "note": f.note}
            for f in self.stack
        ]
        if self.trace is not None:
            self.trace.turn_end(
                answer=answer,
                metrics=summary,
                messages=self.messages,
                stopped=stopped,
                stack=stack_frames,
                goal=self.goal,
                receipt=self.last_receipt,
                control_decision=control_decision,
                gated_chunks=self._last_gated,
                extract_miss_flush=self._last_miss_flush,
            )
        out = {
            "answer": answer,
            "steps": steps,
            "metrics": summary,
            "receipt": self.last_receipt.as_dict() if self.last_receipt else None,
            "goal": self.goal,
            "stack": stack_frames,
        }
        if stopped:
            out["stopped"] = stopped
        return out

    def ask(self, user_text: str):
        if self.repo is None:
            return ask_legacy(self, user_text)

        turn_metrics = metrics_mod.empty_usage()
        steps = []
        self._last_miss_flush = False
        if self.trace is not None:
            self.trace.turn_start(user_text, self.messages, model=self.model)

        stored = self.repo.append_message(
            self.session_id, role="user", content=user_text, kind="chat"
        )
        self.messages.append(user(user_text))

        demos = ""
        if self.compiled_profile:
            demos = str(
                self.compiled_profile.demonstrations.get("controller") or ""
            )
        decision, ctrl_result = self.controller.decide(
            user_text=user_text,
            goal=self.repo.get_goal(self.session_id),
            active_skill=self.repo.active_skill(self.session_id),
            skill_descriptions=self.skills.descriptions(),
            demos=demos,
        )
        ctrl_metrics = metrics_mod.from_llm_result(ctrl_result)
        turn_metrics = metrics_mod.add_usage(turn_metrics, ctrl_metrics)
        if self.trace is not None:
            self.trace.llm(
                model=self.model,
                metrics=ctrl_metrics,
                content=ctrl_result.get("content") or "",
                name="control",
                operation="control",
            )
        if decision.skill not in set(self.skills.names()):
            decision = replace(
                decision,
                skill=self.repo.active_skill(self.session_id),
                fallback=True,
            )
        active = apply_stack_decision(
            self.repo, self.session_id, decision, self.skills
        )
        if self.trace is not None:
            self.trace.control(decision)
        self._ensure_system()

        self._last_hits = []
        rq = decision.retrieval_query or ""
        try:
            hits, hyde_result = self._retrieve(user_text, rq)
            self._last_hits = hits
            if hyde_result is not None:
                hyde_metrics = metrics_mod.from_llm_result(hyde_result)
                turn_metrics = metrics_mod.add_usage(turn_metrics, hyde_metrics)
                if self.trace is not None:
                    self.trace.llm(
                        model=self.model,
                        metrics=hyde_metrics,
                        content=hyde_result.get("content") or "",
                        name="hyde",
                        operation="hyde",
                    )
        except Exception:
            self._last_hits = []

        # Queue mode: drain pending extracts after retrieve.
        # Miss (query + zero hits + pending) → drain + one re-retrieve.
        # Otherwise drain lazily so backlog still clears without a miss.
        if self.extracts is not None and self.extract_mode == "queue":
            pending = self.extracts.pending_count()
            if should_miss_flush(rq, self._last_hits, pending):
                self.extracts.note_miss_flush()
                self._last_miss_flush = True
                drained = self.extracts.drain(reason="miss")
                turn_metrics = metrics_mod.add_usage(turn_metrics, drained)
                try:
                    hits, _ = self._retrieve(user_text, rq)
                    self._last_hits = hits
                except Exception:
                    pass
            elif pending > 0:
                drained = self.extracts.drain(reason="lazy")
                turn_metrics = metrics_mod.add_usage(turn_metrics, drained)

        tools = self.skills.resolve_tools(active, self.tools)
        answer, react_steps, turn_metrics, stopped = self._turns.run_react(
            None, tools, turn_metrics
        )
        steps.extend(react_steps)

        self._last_gated = 0
        if self.indexer is not None:
            self.indexer.add_text(
                self.session_id,
                user_text,
                chunks=self.chunker.chunk(user_text),
                source_message_id=stored.id,
                kind="raw",
            )
            self._last_gated = int(self.indexer.last_gated or 0)
            if self.extracts is not None:
                turn_metrics = self.extracts.after_raw(
                    turn_metrics, anchor_message_id=stored.id
                )

        return self._finish(
            answer, steps, turn_metrics, stopped=stopped, control_decision=decision
        )
