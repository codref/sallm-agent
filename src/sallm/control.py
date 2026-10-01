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
    # Strip common fences before parse.
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, count=1, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```\s*$", "", raw)
        raw = raw.strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(raw)
    if not m:
        return None
    blob = m.group(0)
    try:
        obj = json.loads(blob)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        # Truncated completion: close open strings/arrays/objects best-effort.
        repaired = _repair_truncated_json(blob)
        if repaired is None:
            return None
        try:
            obj = json.loads(repaired)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


def _repair_truncated_json(blob: str) -> str | None:
    """Best-effort close of truncated JSON objects (common under token caps)."""
    if not blob or not blob.lstrip().startswith("{"):
        return None
    s = blob.rstrip()
    # Drop a trailing incomplete key/value fragment after the last safe comma/bracket.
    # If we end mid-string, close the string.
    in_str = False
    escape = False
    stack: list[str] = []
    last_complete = 0
    for i, ch in enumerate(s):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
                last_complete = i + 1
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
            last_complete = i + 1
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
            last_complete = i + 1
        elif ch in ",:":
            last_complete = i + 1
    if in_str:
        s = s + '"'
        last_complete = len(s)
    # Trim to last complete structural point when truncated mid-token.
    if last_complete and last_complete < len(s) and not s.endswith(("}", "]")):
        trim = s[:last_complete].rstrip()
        if trim.endswith(","):
            trim = trim[:-1]
        s = trim
    # Close remaining open containers.
    while stack:
        s += stack.pop()
    return s


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

INGEST_INSTRUCTION = """Interpret ingested content for durable agent memory.
The content may be shell history, logs, notes, or other raw blocks — not a chat turn.
Reply with ONE JSON object only:
{"facts":[{"text":"...","source_message_ids":[ID]}]}
Rules:
- Write 3–12 short English facts a future search can retrieve (hosts, IPs, users,
  repos, images, paths, timestamps, tools used, brief usage patterns).
- Prefer ordinary English over raw command lines
  (e.g. "User ran docker login then ssh to 203.0.113.10").
- Do NOT emit one fact per history line; densify related commands.
- Every fact MUST set source_message_ids to the ingest message id shown in
  brackets at the start of the transcript (e.g. [42] → use [42]).
  Never use shell-history line numbers as message ids.
- If nothing durable, return {"facts":[]}. Do not invent hosts or IPs.
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
        attachment_note: str = "",
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
        note = (attachment_note or "").strip()
        if note:
            prompt += f"\n{note}\n"
        result = complete(
            model=self.profile.model,
            messages=[user(prompt)],
            api_base=self.profile.api_base,
            **self.profile.complete_kwargs(
                max_tokens=self.profile.control_max_tokens, json_mode=True
            ),
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
        max_tokens: int | None = None,
        auto_ground: bool = False,
    ) -> tuple[list[ExtractedFact], dict]:
        prompt = f"{self.instruction}\n"
        if demos:
            prompt += f"Examples:\n{demos}\n"
        prompt += f"\nTranscript:\n{transcript_snippet}\n"
        result = complete(
            model=self.profile.model,
            messages=[user(prompt)],
            api_base=self.profile.api_base,
            **self.profile.complete_kwargs(
                max_tokens=int(max_tokens or self.profile.extract_max_tokens),
                json_mode=True,
            ),
        )
        data = _parse_json(result.get("content") or "")
        facts: list[ExtractedFact] = []
        if not data:
            return facts, result
        # remember(): sole ingest id — remap wrong history-line citations.
        sole_id = (
            next(iter(valid_message_ids))
            if auto_ground and len(valid_message_ids) == 1
            else None
        )
        for item in data.get("facts") or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            raw_ids = item.get("source_message_ids") or []
            if isinstance(raw_ids, int):
                raw_ids = [raw_ids]
            elif isinstance(raw_ids, str):
                raw_ids = [raw_ids]
            elif not isinstance(raw_ids, (list, tuple)):
                raw_ids = [raw_ids]
            ids = []
            for x in raw_ids:
                try:
                    i = int(x)
                except (TypeError, ValueError):
                    continue
                if i in valid_message_ids:
                    ids.append(i)
            if not ids and sole_id is not None:
                ids = [sole_id]
            if not ids:
                continue  # reject ungrounded facts
            facts.append(ExtractedFact(text=text, source_message_ids=ids))
        return facts, result
