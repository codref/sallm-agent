"""Scoring helpers for offline prompt optimization."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class Score:
    quality: float
    tokens: int
    latency_ms: float
    invalid: bool = False
    mandatory_fail: bool = False

    @property
    def total(self) -> float:
        """Higher is better. Penalize tokens, latency, and invalid output."""
        if self.mandatory_fail:
            return -1e9
        pen = self.tokens / 10000.0 + self.latency_ms / 100000.0
        if self.invalid:
            pen += 0.5
        return self.quality - pen


def exact_field_match(got: dict, expected: dict) -> float:
    if not expected:
        return 1.0
    hits = 0
    for k, v in expected.items():
        if got.get(k) == v:
            hits += 1
    return hits / max(len(expected), 1)


def contains_all(text: str, needles: list[str]) -> float:
    if not needles:
        return 1.0
    blob = (text or "").lower()
    hits = sum(1 for n in needles if n.lower() in blob)
    return hits / len(needles)


def _stringify(got) -> str:
    if isinstance(got, str):
        return got
    try:
        return json.dumps(got, ensure_ascii=False)
    except TypeError:
        return str(got)


def score_json_output(got: dict, expected: dict) -> float:
    """Score controller/extractor JSON: exact fields plus soft contains/query checks."""
    expected = expected or {}
    skip = {
        "contains",
        "retrieval_query_contains",
        "retrieval_query_nonempty",
        "absent",
        "no_tool",
        "tool",
        "argv_contains",
        "observation_contains",
    }
    parts: list[float] = []
    field_exp = {k: v for k, v in expected.items() if k not in skip}
    if field_exp:
        parts.append(exact_field_match(got, field_exp))
    blob = _stringify(got)
    needles = list(expected.get("contains") or [])
    if needles:
        parts.append(contains_all(blob, needles))
    rq_needles = list(expected.get("retrieval_query_contains") or [])
    if rq_needles:
        parts.append(contains_all(str(got.get("retrieval_query") or ""), rq_needles))
    if expected.get("retrieval_query_nonempty"):
        parts.append(1.0 if str(got.get("retrieval_query") or "").strip() else 0.0)
    absent = list(expected.get("absent") or [])
    if absent:
        low = blob.lower()
        parts.append(1.0 if all(n.lower() not in low for n in absent) else 0.0)
    if not parts:
        return 1.0
    return sum(parts) / len(parts)


def score_program_output(got, expected: dict) -> float:
    """Score a converse/rewriter generation: needles, parsed ```run, observations."""
    expected = expected or {}
    if isinstance(got, dict) and "content" in got:
        content = str(got.get("content") or "")
        commands = got.get("commands")
        observation = str(got.get("observation") or "")
        if not isinstance(commands, list):
            commands = None
    else:
        content = _stringify(got)
        commands = None
        observation = ""
    if commands is None:
        from sallm.tools.runner import parse_run_blocks

        commands = parse_run_blocks(content)

    parts: list[float] = []
    needles = list(expected.get("contains") or [])
    if needles:
        parts.append(contains_all(content, needles))
    tool_name = expected.get("tool")
    matched = [c for c in commands if c and str(c[0]) == str(tool_name)] if tool_name else []
    if tool_name:
        parts.append(1.0 if matched else 0.0)
    # Reject multi-line ```run that invents memory codes as tools (ORANGE-19, …).
    if expected.get("single_tool"):
        ok = (
            len(commands) == 1
            and bool(commands[0])
            and (not tool_name or str(commands[0][0]) == str(tool_name))
        )
        parts.append(1.0 if ok else 0.0)
    argv_needles = list(expected.get("argv_contains") or [])
    if argv_needles:
        blob = " ".join(" ".join(str(x) for x in c) for c in (matched or commands))
        parts.append(contains_all(blob, argv_needles))
    obs_needles = list(expected.get("observation_contains") or [])
    if obs_needles:
        parts.append(contains_all(observation, obs_needles))
    if expected.get("no_tool"):
        parts.append(1.0 if not commands else 0.0)
    absent = list(expected.get("absent") or [])
    if absent:
        blob = content.lower()
        parts.append(1.0 if all(n.lower() not in blob for n in absent) else 0.0)
    if not parts:
        return 1.0
    return sum(parts) / len(parts)
