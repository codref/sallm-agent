"""Image mentions, prompt placement, and the Ollama chat rewrite. No live model."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from sallm.attachments import AttachmentError, parse_image_mentions
from sallm.llm import complete
from sallm.memory.gates import HeuristicMemoryGate
from sallm.models import ModelProfile
from sallm.optimization.program import caption_messages, control_user_prompt, converse_messages
from sallm.optimization.dataset import Case
from sallm.prompt import VISION_INSTRUCTION, CompiledProfile, Prompt
from sallm.receipt import compile_prompt_messages


def _png(path: Path, *, left=(220, 30, 30), right=(30, 60, 220)) -> Path:
    width, height = 8, 4
    raw = bytearray()
    for _y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(left if x < width // 2 else right)

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    blob = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(blob)
    return path


def _has_image(message) -> bool:
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(part, dict) and part.get("type") == "image_url" for part in content
    )


def test_parse_question_and_context(tmp_path):
    diagram = _png(tmp_path / "diagram.png")
    note = _png(tmp_path / "note.png", left=(1, 1, 1), right=(2, 2, 2))
    text, mentions = parse_image_mentions(
        f"see @{diagram.name} and @context:{note.name} please",
        base=tmp_path,
    )
    assert text == "see and please"
    assert [m.role for m in mentions] == ["question", "context"]
    assert mentions[0].filename == "diagram.png"


def test_parse_missing_and_not_image(tmp_path):
    with pytest.raises(AttachmentError, match="not found"):
        parse_image_mentions("@missing.png", base=tmp_path)
    plain = tmp_path / "notes.txt"
    plain.write_text("hello", encoding="utf-8")
    with pytest.raises(AttachmentError, match="not an image"):
        parse_image_mentions(f"@{plain.name}", base=tmp_path)


def test_compile_places_question_and_context_images(tmp_path):
    diagram = _png(tmp_path / "diagram.png")
    ref = _png(tmp_path / "ref.png")
    profile = ModelProfile(
        prompt_budget=4000,
        recent_history_tokens=2000,
        retrieval_tokens=800,
        image_tokens=280,
    )
    prompt = Prompt(tools_text="(none)")
    history = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": "what colors?",
            "attachments": [
                {
                    "id": "q1",
                    "role": "question",
                    "path": str(diagram),
                    "mime": "image/png",
                    "filename": "diagram.png",
                },
                {
                    "id": "c1",
                    "role": "context",
                    "path": str(ref),
                    "mime": "image/png",
                    "filename": "ref.png",
                    "caption": "a reference sketch",
                },
            ],
        },
    ]
    messages, receipt = compile_prompt_messages(
        profile=profile,
        prompt=prompt,
        recent_messages=history,
        hits=[],
    )
    assert VISION_INSTRUCTION.splitlines()[0] in messages[0]["content"]
    question = messages[-1]
    assert _has_image(question)
    assert question["content"][0]["text"] == "what colors?"
    context = next(
        m
        for m in messages
        if isinstance(m.get("content"), list)
        and str(m["content"][0].get("text") or "").startswith("[Retrieved memory]")
    )
    assert _has_image(context)
    assert "ref.png" not in question["content"][0]["text"]
    history_section = next(s for s in receipt.sections if s.name == "history")
    assert history_section.tokens >= 280
    retrieval = next(s for s in receipt.sections if s.name == "retrieval")
    assert retrieval.tokens >= 280
    assert "image" in retrieval.note


def test_compile_keeps_newest_question_image(tmp_path):
    older = _png(tmp_path / "older.png")
    newer = _png(tmp_path / "newer.png")
    profile = ModelProfile(
        prompt_budget=4000, recent_history_tokens=2000, image_tokens=10
    )
    prompt = Prompt(tools_text="(none)")
    history = [
        {
            "role": "user",
            "content": "first",
            "attachments": [
                {
                    "id": "old",
                    "role": "question",
                    "path": str(older),
                    "mime": "image/png",
                    "filename": "older.png",
                }
            ],
        },
        {"role": "assistant", "content": "ok"},
        {
            "role": "user",
            "content": "second",
            "attachments": [
                {
                    "id": "new",
                    "role": "question",
                    "path": str(newer),
                    "mime": "image/png",
                    "filename": "newer.png",
                }
            ],
        },
    ]
    messages, receipt = compile_prompt_messages(
        profile=profile, prompt=prompt, recent_messages=history, hits=[]
    )
    imaged = [m for m in messages if _has_image(m)]
    assert len(imaged) == 1
    assert imaged[0]["content"][0]["text"] == "second"
    history_section = next(s for s in receipt.sections if s.name == "history")
    assert "dropped_images=older.png" in history_section.note


def test_retrieved_image_is_context(tmp_path):
    from sallm.memory.types import VectorHit

    diagram = _png(tmp_path / "old.png")
    profile = ModelProfile(prompt_budget=4000, retrieval_tokens=800, image_tokens=40)
    prompt = Prompt(tools_text="(none)")
    hit = VectorHit(
        id="h1",
        text="red block beside a blue block",
        score=0.8,
        source_id="3",
        metadata={
            "kind": "image",
            "modality": "image",
            "attachment_id": "att-old",
            "path": str(diagram),
            "mime": "image/png",
        },
    )
    messages, _receipt = compile_prompt_messages(
        profile=profile,
        prompt=prompt,
        recent_messages=[{"role": "user", "content": "what was the diagram?"}],
        hits=[hit],
    )
    context = next(
        m
        for m in messages
        if isinstance(m.get("content"), list)
        and "[Retrieved memory]" in m["content"][0]["text"]
    )
    assert _has_image(context)
    assert "red block" in context["content"][0]["text"]
    question = messages[-1]
    assert question["content"] == "what was the diagram?"


def test_vision_instruction_absent_without_images():
    profile = ModelProfile()
    prompt = Prompt(tools_text="(none)")
    messages, _receipt = compile_prompt_messages(
        profile=profile,
        prompt=prompt,
        recent_messages=[{"role": "user", "content": "hello"}],
        hits=[],
    )
    assert "Images may appear" not in messages[0]["content"]


def test_vision_instruction_override():
    compiled = CompiledProfile(
        target_model="openai/gpt-4o",
        instructions={"vision": "CUSTOM_VISION_RULE"},
        demonstrations={},
        budgets={},
        metadata={},
    )
    prompt = Prompt(tools_text="(none)", compiled=compiled)
    assert "CUSTOM_VISION_RULE" in prompt.system(vision=True)
    assert "Images may appear" not in prompt.system(vision=True)
    assert "Images may appear" not in prompt.system()


def test_schema_migrate_2_to_3(tmp_path):
    from sallm.state import SessionRepository
    from sallm.state.models import SCHEMA_VERSION, Attachment, SchemaMeta, db

    path = tmp_path / "old.db"
    repo = SessionRepository(path)
    SchemaMeta.update(value="2").where(SchemaMeta.key == "version").execute()
    repo.close()
    db.close()

    repo2 = SessionRepository(path)
    assert SchemaMeta.get(SchemaMeta.key == "version").value == str(SCHEMA_VERSION)
    repo2.ensure_session("s")
    stored = repo2.append_message("s", role="user", content="see", kind="chat")
    att = repo2.add_attachment(
        "s",
        attachment_id="a1",
        message_id=stored.id,
        role="question",
        mime="image/png",
        sha256="abc",
        filename="diagram.png",
        path="/tmp/diagram.png",
    )
    assert att.filename == "diagram.png"
    assert Attachment.select().count() == 1
    repo2.set_caption("a1", "red beside blue")
    found = repo2.find_attachment_by_caption("s", "red beside blue", message_id=stored.id)
    assert found is not None
    assert found.path == "/tmp/diagram.png"


def test_image_kind_always_indexed():
    gate = HeuristicMemoryGate()
    assert gate.accept("red?", kind="image") is True


def test_image_chunk_metadata(tmp_path):
    from sallm.memory.index import MemoryIndexer
    from sallm.state import SessionRepository
    from tests.test_memory_vector import FakeEmbedder, InMemoryVectorStore

    repo = SessionRepository(tmp_path / "state.db")
    repo.ensure_session("s")
    store = InMemoryVectorStore(8)
    indexer = MemoryIndexer(repo, store, FakeEmbedder(), gate=HeuristicMemoryGate())
    stored = repo.append_message("s", role="user", content="see", kind="chat")
    diagram = _png(tmp_path / "diagram.png")
    att = repo.add_attachment(
        "s",
        attachment_id="att1",
        message_id=stored.id,
        role="question",
        mime="image/png",
        sha256="zzz",
        filename="diagram.png",
        path=str(diagram),
    )
    repo.set_caption(att.id, "red block beside blue")
    indexer.add_text(
        "s",
        "red block beside blue",
        chunks=["red block beside blue"],
        source_message_id=stored.id,
        kind="image",
        metadata={
            "modality": "image",
            "attachment_id": att.id,
            "path": str(diagram),
            "mime": "image/png",
        },
    )
    record = next(iter(store._rows.values()))
    assert record.metadata["modality"] == "image"
    assert record.metadata["attachment_id"] == "att1"
    assert record.metadata["kind"] == "image"


def test_optimize_messages_carry_images(tmp_path):
    diagram = _png(tmp_path / "diagram.png")
    case = Case(
        id="v",
        task="vision",
        input={
            "user": "colors?",
            "images": [{"path": str(diagram), "role": "question", "mime": "image/png"}],
        },
        expected={"contains": ["red"]},
    )
    compiled = CompiledProfile("m", {}, {}, {}, {})
    messages = converse_messages(case, "Look.", compiled)
    assert _has_image(messages[-1])
    caption = caption_messages(
        Case(
            id="c",
            task="caption",
            input={"image": str(diagram)},
            expected={},
        ),
        "Describe.",
    )
    assert _has_image(caption[0])
    control = control_user_prompt(
        Case(
            id="k",
            task="controller",
            input={"user": "colors?", "attached": "Attached images (not shown to you):\n- diagram.png (question)."},
            expected={},
        ),
        "Route.",
    )
    assert "diagram.png" in control


class _Msg:
    content = "ok"
    reasoning_content = None


class _Choice:
    message = _Msg()
    finish_reason = "stop"


class _Resp:
    choices = [_Choice()]
    usage = None


def test_complete_rewrites_ollama_only_for_images(monkeypatch):
    seen = {}

    def fake_completion(**kwargs):
        seen["model"] = kwargs["model"]
        return _Resp()

    monkeypatch.setattr("sallm.llm.completion", fake_completion)
    image = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aGk="},
                },
            ],
        }
    ]
    complete(model="ollama/gemma4:e4b-it-qat", messages=image)
    assert seen["model"] == "ollama_chat/gemma4:e4b-it-qat"
    complete(model="openai/gpt-4o", messages=image)
    assert seen["model"] == "openai/gpt-4o"
    complete(
        model="ollama/gemma4:e4b-it-qat",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert seen["model"] == "ollama/gemma4:e4b-it-qat"
