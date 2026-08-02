"""JSONL dataset loading for offline prompt optimization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    id: str
    task: str  # controller | extractor | converse | rewriter
    input: dict
    expected: dict
    mandatory: bool = False


def load_jsonl(path: str | Path) -> list[Case]:
    cases = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        cases.append(
            Case(
                id=str(data.get("id") or f"case-{i}"),
                task=str(data.get("task") or "converse"),
                input=dict(data.get("input") or {}),
                expected=dict(data.get("expected") or {}),
                mandatory=bool(data.get("mandatory")),
            )
        )
    return cases


def fingerprint(cases: list[Case]) -> str:
    blob = json.dumps(
        [c.__dict__ for c in cases], sort_keys=True, ensure_ascii=True
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
