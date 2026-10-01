"""ContextReceipt and budgeted prompt assembly."""

from __future__ import annotations

from dataclasses import dataclass, field

from pathlib import Path

from sallm.attachments import content_parts
from sallm.context import estimate_tokens
from sallm.memory.types import VectorHit
from sallm.messages import assistant, system, user
from sallm.models import ModelProfile


@dataclass
class SectionSpend:
    name: str
    tokens: int
    included: bool = True
    note: str = ""


@dataclass
class ContextReceipt:
    """Public explanation of what entered the prompt and why."""

    profile: str
    profile_version: str
    budget: int
    sections: list[SectionSpend] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    omitted_messages: int = 0
    fallbacks: list[str] = field(default_factory=list)
    total_tokens: int = 0

    def as_dict(self) -> dict:
        return {
            "profile": self.profile,
            "profile_version": self.profile_version,
            "budget": self.budget,
            "total_tokens": self.total_tokens,
            "omitted_messages": self.omitted_messages,
            "fallbacks": list(self.fallbacks),
            "sections": [
                {
                    "name": s.name,
                    "tokens": s.tokens,
                    "included": s.included,
                    "note": s.note,
                }
                for s in self.sections
            ],
            "retrieved": list(self.retrieved),
        }


def _fit(text: str, budget: int) -> tuple[str, int, bool]:
    if budget <= 0 or not text:
        return "", 0, bool(text)
    tokens = estimate_tokens(text)
    if tokens <= budget:
        return text, tokens, False
    cut = max(1, budget * 4)
    trimmed = text[:cut].rstrip() + "…"
    return trimmed, estimate_tokens(trimmed), True


def compile_prompt_messages(
    *,
    profile: ModelProfile,
    prompt,  # Prompt-like: .system()
    recent_messages: list[dict],
    hits: list[VectorHit],
    retrieval_budget: int | None = None,
    history_budget: int | None = None,
) -> tuple[list[dict], ContextReceipt]:
    """Assemble system + memory + recent history under the profile budget."""
    receipt = ContextReceipt(
        profile=profile.model,
        profile_version=profile.version,
        budget=profile.prompt_budget,
    )
    messages: list[dict] = []

    sys_text = prompt.system()
    sys_text, sys_tok, sys_trim = _fit(sys_text, profile.prompt_budget)
    messages.append(system(sys_text))
    receipt.sections.append(
        SectionSpend("system", sys_tok, True, "trimmed" if sys_trim else "")
    )
    used = sys_tok

    r_budget = (
        retrieval_budget
        if retrieval_budget is not None
        else profile.retrieval_tokens
    )
    r_budget = min(r_budget, max(0, profile.prompt_budget - used))
    mem_parts = []
    for h in hits:
        mem_parts.append(
            f"[mem id={h.id} source={h.source_id or '-'} score={h.score:.3f}]\n{h.text}"
        )
        receipt.retrieved.append(
            {
                "id": h.id,
                "source_id": h.source_id,
                "score": h.score,
                "chars": len(h.text),
            }
        )
    if mem_parts:
        mem_text = "[Retrieved memory]\n" + "\n---\n".join(mem_parts)
        mem_text, mem_tok, mem_trim = _fit(mem_text, r_budget)
        if mem_text:
            messages.append(user(mem_text))
            used += mem_tok
            receipt.sections.append(
                SectionSpend(
                    "retrieval", mem_tok, True, "trimmed" if mem_trim else ""
                )
            )
        else:
            receipt.sections.append(SectionSpend("retrieval", 0, False, "no budget"))
    else:
        receipt.sections.append(SectionSpend("retrieval", 0, False, "no hits"))

    h_budget = (
        history_budget
        if history_budget is not None
        else profile.recent_history_tokens
    )
    h_budget = min(h_budget, max(0, profile.prompt_budget - used))
    rest = list(recent_messages)
    if rest and rest[0].get("role") == "system":
        rest = rest[1:]
    kept: list[dict] = []
    hist_tokens = 0
    charged_question = False
    for msg in reversed(rest):
        content = msg.get("content") or ""
        if not isinstance(content, str):
            content = ""
        extra = 0
        questions = _question_attachments(msg)
        if questions and not charged_question:
            extra = int(profile.image_tokens or 0)
            charged_question = True
        t = estimate_tokens(content) + 2 + extra
        if hist_tokens + t > h_budget and kept:
            break
        if hist_tokens + t > h_budget and not kept:
            content, t, _ = _fit(content, max(0, h_budget - extra))
            t += extra
            kept.append(_copy_message(msg, content))
            hist_tokens += t
            break
        kept.append(_copy_message(msg, content))
        hist_tokens += t
    kept.reverse()
    omitted = max(0, len(rest) - len(kept))
    receipt.omitted_messages = omitted
    messages.extend(kept)
    used += hist_tokens
    question, context, dropped = choose_images(messages, hits)
    if question is not None or context is not None:
        sys_text, sys_tok, sys_trim = _fit(
            prompt.system(vision=True), profile.prompt_budget
        )
        messages[0] = system(sys_text)
        used += sys_tok - receipt.sections[0].tokens
        receipt.sections[0] = SectionSpend(
            "system", sys_tok, True, "trimmed" if sys_trim else ""
        )
    messages, context_tokens = _apply_images(
        messages, question, context, profile.image_tokens
    )
    if context_tokens:
        used += context_tokens
        for index, section in enumerate(receipt.sections):
            if section.name == "retrieval":
                note = section.note
                if note in ("", "no hits", "no budget"):
                    note = "image"
                else:
                    note = f"{note} image"
                receipt.sections[index] = SectionSpend(
                    "retrieval",
                    section.tokens + context_tokens,
                    True,
                    note,
                )
                break
    hist_note = f"omitted={omitted}" if omitted else ""
    if dropped:
        names = ",".join(dropped)
        hist_note = f"{hist_note} dropped_images={names}".strip()
    receipt.sections.append(
        SectionSpend(
            "history",
            hist_tokens,
            True,
            hist_note,
        )
    )
    receipt.total_tokens = used
    if used > profile.prompt_budget:
        receipt.fallbacks.append("over_budget_estimate")
    return messages, receipt


def _question_attachments(msg: dict) -> list[dict]:
    out = []
    for item in msg.get("attachments") or []:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        if (item.get("role") or "question") == "context":
            continue
        out.append(item)
    return out


def _context_attachments(msg: dict) -> list[dict]:
    out = []
    for item in msg.get("attachments") or []:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        if (item.get("role") or "question") == "context":
            out.append(item)
    return out


def _copy_message(msg: dict, content: str) -> dict:
    role = msg.get("role") or "user"
    if role == "assistant":
        out = assistant(content)
    elif role == "system":
        out = system(content)
    else:
        out = user(content)
    attachments = msg.get("attachments")
    if attachments:
        out["attachments"] = list(attachments)
    return out


def _hit_image(hit: VectorHit) -> dict | None:
    meta = hit.metadata or {}
    if meta.get("modality") != "image":
        return None
    path = meta.get("path") or ""
    if not path or not Path(path).is_file():
        return None
    return {
        "id": meta.get("attachment_id") or hit.id,
        "role": "context",
        "path": path,
        "mime": meta.get("mime") or "image/png",
        "filename": Path(path).name,
        "caption": hit.text,
    }


def choose_images(messages: list[dict], hits: list[VectorHit]):
    """Newest question image, one context image, and filenames left out."""
    question = None
    context_from_msg = None
    dropped: list[str] = []
    for index, msg in enumerate(messages):
        for att in _question_attachments(msg):
            if question is not None:
                dropped.append(question[1].get("filename") or "image")
            question = (index, att)
        for att in _context_attachments(msg):
            if context_from_msg is not None:
                dropped.append(context_from_msg[1].get("filename") or "image")
            context_from_msg = (index, att)
    qid = question[1].get("id") if question else None
    retrieved = None
    for hit in hits or []:
        att = _hit_image(hit)
        if att is None or att.get("id") == qid:
            continue
        if retrieved is None:
            retrieved = att
        else:
            dropped.append(att.get("filename") or "image")
    if retrieved is not None:
        context = retrieved
        if context_from_msg is not None and context_from_msg[1].get("id") != retrieved.get(
            "id"
        ):
            dropped.append(context_from_msg[1].get("filename") or "image")
    else:
        context = context_from_msg[1] if context_from_msg else None
    return question, context, [name for name in dropped if name]


def _plain(msg: dict) -> dict:
    content = msg.get("content") or ""
    if not isinstance(content, str):
        content = ""
    role = msg.get("role") or "user"
    if role == "assistant":
        return assistant(content)
    if role == "system":
        return system(content)
    return user(content)


def _apply_images(messages, question, context, image_tokens: int):
    """Return LLM messages (no attachment side channel) and context image tokens."""
    q_index = question[0] if question else None
    q_att = question[1] if question else None
    out = []
    retrieval_at = None
    for index, msg in enumerate(messages):
        content = msg.get("content") or ""
        if not isinstance(content, str):
            content = ""
        if (
            retrieval_at is None
            and msg.get("role") == "user"
            and content.startswith("[Retrieved memory]")
        ):
            retrieval_at = len(out)
        if q_index is not None and index == q_index and q_att is not None:
            out.append(
                {
                    "role": msg.get("role") or "user",
                    "content": content_parts(content, [q_att]),
                }
            )
        else:
            out.append(_plain(msg))
    context_tokens = 0
    if context is not None:
        context_tokens = int(image_tokens or 0)
        caption = (context.get("caption") or "").strip()
        label = context.get("filename") or "image"
        if retrieval_at is None:
            text = "[Retrieved memory]\n"
            if caption:
                text += caption
            else:
                text += f"[image {label}]"
            out.insert(1, {"role": "user", "content": content_parts(text, [context])})
        else:
            host = out[retrieval_at]
            text = host.get("content") or ""
            if not isinstance(text, str):
                text = ""
            out[retrieval_at] = {
                "role": "user",
                "content": content_parts(text, [context]),
            }
    return out, context_tokens


def hydrate_llm_messages(messages, *, prompt, profile):
    """Attach question and context images for the legacy in-memory loop."""
    question, context, _dropped = choose_images(messages, [])
    view = list(messages)
    if (question is not None or context is not None) and view and view[0].get("role") == "system":
        view[0] = system(prompt.system(vision=True))
    rendered, _tokens = _apply_images(
        view, question, context, getattr(profile, "image_tokens", 0)
    )
    return rendered
