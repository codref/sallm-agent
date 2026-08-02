"""Goal/skill control, query rewrite, and source-grounded memory extraction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from sallm.llm import complete
from sallm.messages import user
from sallm.models import ModelProfile

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> dict | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


@dataclass(frozen=True)
class ControlDecision:
    goal: str
    action: str  # keep | push | pop | replace
    skill: str
    retrieval_query: str
    fallback: bool = False


@dataclass(frozen=True)
class ExtractedFact:
    text: str
    source_message_ids: list[int]


CONTROL_INSTRUCTION = """You route a long-running local agent.
Reply with ONE JSON object only (no markdown):
{"goal":"...","action":"keep|push|pop|replace","skill":"...","retrieval_query":"..."}
Rules:
- goal: one short sentence for the user's current intent (or keep prior goal).
- action: keep current skill unless the user clearly changes task; then push/replace.
- skill: must be one of the registered skills.
- retrieval_query: a short standalone sentence for vector search, or "" if none needed.
"""

EXTRACT_INSTRUCTION = """Extract durable facts from the latest turn.
Reply with ONE JSON object only:
{"facts":[{"text":"...","source_message_ids":[1,2]}]}
Only include facts supported by the listed source message ids.
If nothing durable, return {"facts":[]}.
"""


class Controller:
    def __init__(self, profile: ModelProfile, *, instruction: str | None = None):
        self.profile = profile
        self.instruction = instruction or CONTROL_INSTRUCTION

    def decide(
        self,
        *,
        user_text: str,
        goal: str,
        active_skill: str,
        skill_descriptions: str,
        demos: str = "",
    ) -> tuple[ControlDecision, dict]:
        prompt = (
            f"{self.instruction}\n"
            f"Registered skills:\n{skill_descriptions}\n"
            f"Current goal: {goal or '(none)'}\n"
            f"Active skill: {active_skill}\n"
        )
        if demos:
            prompt += f"\nExamples:\n{demos}\n"
        prompt += f"\nUser message:\n{user_text}\n"
        result = complete(
            model=self.profile.model,
            messages=[user(prompt)],
            api_base=self.profile.api_base,
            max_tokens=self.profile.control_max_tokens,
        )
        data = _parse_json(result.get("content") or "")
        allowed = {"keep", "push", "pop", "replace"}
        if not data:
            return (
                ControlDecision(
                    goal=goal or user_text.strip()[:200],
                    action="keep",
                    skill=active_skill,
                    retrieval_query=user_text.strip()[:200],
                    fallback=True,
                ),
                result,
            )
        action = str(data.get("action") or "keep").strip().lower()
        if action not in allowed:
            action = "keep"
        skill = str(data.get("skill") or active_skill).strip() or active_skill
        new_goal = str(data.get("goal") or goal or "").strip()
        rq = str(data.get("retrieval_query") or "").strip()
        return (
            ControlDecision(
                goal=new_goal or goal or user_text.strip()[:200],
                action=action,
                skill=skill,
                retrieval_query=rq,
                fallback=False,
            ),
            result,
        )


class MemoryExtractor:
    def __init__(self, profile: ModelProfile, *, instruction: str | None = None):
        self.profile = profile
        self.instruction = instruction or EXTRACT_INSTRUCTION

    def extract(
        self,
        *,
        transcript_snippet: str,
        valid_message_ids: set[int],
        demos: str = "",
    ) -> tuple[list[ExtractedFact], dict]:
        prompt = f"{self.instruction}\n"
        if demos:
            prompt += f"Examples:\n{demos}\n"
        prompt += f"\nTranscript:\n{transcript_snippet}\n"
        result = complete(
            model=self.profile.model,
            messages=[user(prompt)],
            api_base=self.profile.api_base,
            max_tokens=self.profile.extract_max_tokens,
        )
        data = _parse_json(result.get("content") or "")
        facts: list[ExtractedFact] = []
        if not data:
            return facts, result
        for item in data.get("facts") or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            raw_ids = item.get("source_message_ids") or []
            ids = []
            for x in raw_ids:
                try:
                    i = int(x)
                except (TypeError, ValueError):
                    continue
                if i in valid_message_ids:
                    ids.append(i)
            if not ids:
                continue  # reject ungrounded facts
            facts.append(ExtractedFact(text=text, source_message_ids=ids))
        return facts, result
