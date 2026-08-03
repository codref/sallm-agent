"""Minimal skill registry and stack frames."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Skill:
    """A skill is a prompt fragment + optional tool subset and budget overrides."""

    name: str
    description: str
    prompt: str = ""
    tools: tuple[str, ...] | None = None  # None = all registered tools
    max_steps: int | None = None
    budget_overrides: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SkillFrame:
    skill: str
    depth: int
    note: str = ""


CONVERSE = Skill(
    name="converse",
    description="Default conversational skill: answer the user, use tools when needed.",
    prompt=(
        "Active skill: converse.\n"
        "Stay helpful and concise. Prefer plain-text answers unless a tool is required."
    ),
)


class SkillRegistry:
    def __init__(self, skills: list[Skill] | None = None):
        self._skills: dict[str, Skill] = {}
        for s in skills or [CONVERSE]:
            self.register(s)
        if "converse" not in self._skills:
            self.register(CONVERSE)

    def register(self, skill: Skill):
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"unknown skill {name!r}")
        return self._skills[name]

    def names(self) -> list[str]:
        return sorted(self._skills)

    def descriptions(self) -> str:
        lines = []
        for name in self.names():
            s = self._skills[name]
            lines.append(f"- {s.name}: {s.description}")
        return "\n".join(lines)

    def resolve_tools(
        self, skill_name: str, available: dict
    ) -> dict:
        """Subset the tool registry for the active skill (or all if unrestricted)."""
        skill = self.get(skill_name)
        if skill.tools is None:
            return dict(available)
        return {k: available[k] for k in skill.tools if k in available}
