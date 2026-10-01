"""Compile candidate instructions into the same prompt the agent runs."""

from __future__ import annotations

from sallm.attachments import content_parts, mime_for
from sallm.prompt import CompiledProfile, Prompt
from sallm.tools import format_observations, prepare_commands, run_many, tool_descriptions

from .dataset import Case


def overlay_instruction(
    compiled: CompiledProfile, task: str, instruction: str
) -> CompiledProfile:
    instructions = dict(compiled.instructions)
    instructions[task] = instruction
    return CompiledProfile(
        target_model=compiled.target_model,
        instructions=instructions,
        demonstrations=compiled.demonstrations,
        budgets=compiled.budgets,
        metadata=compiled.metadata,
    )


def converse_messages(
    case: Case,
    instruction: str,
    compiled: CompiledProfile,
    *,
    tools_text: str = "",
    demos: str = "",
) -> list[dict]:
    """System + user messages matching chat's ReAct first step."""
    text = instruction
    if demos:
        text = f"{instruction}\nExamples:\n{demos}"
    overlay = overlay_instruction(compiled, "converse", text)
    budgets = overlay.budgets or {}
    system = Prompt(
        tools_text=tools_text or "(none)",
        compiled=overlay,
        think=budgets.get("think"),
        think_hint=budgets.get("think_hint"),
    ).system()
    user_text = str(case.input.get("user") or case.input)
    messages = [{"role": "system", "content": system}]
    retrieved = str(case.input.get("retrieved") or "").strip()
    if retrieved:
        if not retrieved.startswith("[Retrieved memory]"):
            retrieved = "[Retrieved memory]\n" + retrieved
        messages.append({"role": "user", "content": retrieved})
    for turn in case.input.get("history") or []:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "user")
        if role not in ("user", "assistant"):
            role = "user"
        content = str(turn.get("content") or "")
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_text})
    images = [item for item in (case.input.get("images") or []) if isinstance(item, dict)]
    question = [item for item in images if (item.get("role") or "question") != "context"]
    context = [item for item in images if (item.get("role") or "question") == "context"]
    if question and messages[-1].get("role") == "user":
        messages[-1] = {
            "role": "user",
            "content": content_parts(str(messages[-1].get("content") or ""), question),
        }
    if context:
        host = None
        for index, message in enumerate(messages):
            content = message.get("content")
            if (
                message.get("role") == "user"
                and isinstance(content, str)
                and content.startswith("[Retrieved memory]")
            ):
                host = index
                break
        if host is None:
            messages.insert(
                1,
                {
                    "role": "user",
                    "content": content_parts("[Retrieved memory]", context),
                },
            )
        else:
            messages[host] = {
                "role": "user",
                "content": content_parts(str(messages[host].get("content") or ""), context),
            }
    return messages


def control_user_prompt(case: Case, instruction: str, demos: str = "") -> str:
    """Match Controller.decide() prompt shape."""
    user_text = str(case.input.get("user") or case.input)
    goal = str(case.input.get("goal") or "")
    active = str(case.input.get("active_skill") or "converse")
    skills = str(case.input.get("skills") or "- converse: default chat")
    prompt = (
        f"{instruction}\n"
        f"Registered skills:\n{skills}\n"
        f"Current goal: {goal or '(none)'}\n"
        f"Active skill: {active}\n"
    )
    if demos:
        prompt += f"\nExamples:\n{demos}\n"
    prompt += f"\nUser message:\n{user_text}\n"
    attached = case.input.get("attached") or ""
    if isinstance(attached, str) and attached.strip():
        prompt += f"\n{attached.strip()}\n"
    return prompt


def extract_user_prompt(case: Case, instruction: str, demos: str = "") -> str:
    """Match MemoryExtractor.extract() prompt shape."""
    transcript = str(case.input.get("transcript") or case.input)
    prompt = f"{instruction}\n"
    if demos:
        prompt += f"Examples:\n{demos}\n"
    prompt += f"\nTranscript:\n{transcript}\n"
    return prompt


def caption_messages(case: Case, instruction: str) -> list[dict]:
    """User message: caption instruction plus the case image."""
    path = case.input.get("image") or case.input.get("path") or ""
    image = {"path": str(path), "mime": mime_for(path) if path else "image/png"}
    return [
        {
            "role": "user",
            "content": content_parts(instruction, [image] if path else []),
        }
    ]


def tools_text_for(registry: dict) -> str:
    if not registry:
        return "(none)"
    return tool_descriptions(registry)


def materialize_got(content: str, registry: dict, expected: dict | None) -> dict:
    """Parse ```run / ```file blocks and optionally execute them for the metric."""
    content = content or ""
    prepared = prepare_commands(content)
    try:
        commands = list(prepared.commands)
        observation = ""
        expected = expected or {}
        should_run = bool(commands) and (
            expected.get("tool") or expected.get("observation_contains")
        )
        if prepared.error:
            observation = prepared.error
            commands = list(prepared.commands)
        elif should_run and registry:
            observation = format_observations(run_many(registry, commands))
        return {
            "content": content,
            "commands": commands,
            "observation": observation,
        }
    finally:
        prepared.cleanup()

