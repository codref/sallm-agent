"""Scoring helpers for offline prompt optimization."""

from __future__ import annotations

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
