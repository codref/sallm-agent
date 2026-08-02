"""Narrow transactional session API — Peewee stays behind this facade."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .models import (
    ALL_TABLES,
    SCHEMA_VERSION,
    DerivedMemory,
    MemoryChunk,
    Message,
    PendingExtract,
    SchemaMeta,
    Session,
    SkillFrame,
    db,
)


@dataclass(frozen=True)
class StoredMessage:
    id: int
    role: str
    content: str
    kind: str
    seq: int


@dataclass(frozen=True)
class StoredFrame:
    skill: str
    depth: int
    note: str


@dataclass(frozen=True)
class StoredChunk:
    id: str
    text: str
    source_message_id: int | None
    kind: str
    indexed: bool


@dataclass(frozen=True)
class PendingExtractJob:
    id: int
    anchor_message_id: int
    status: str
    created_at: float


class SessionRepository:
    """SQLite-backed session store. One repository owns one database file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db.init(str(self.path), pragmas={"foreign_keys": 1, "journal_mode": "wal"})
        db.connect(reuse_if_open=True)
        db.create_tables(ALL_TABLES)
        self._ensure_schema()

    def close(self):
        if not db.is_closed():
            db.close()

    def _ensure_schema(self):
        row = SchemaMeta.get_or_none(SchemaMeta.key == "version")
        if row is None:
            SchemaMeta.create(key="version", value=str(SCHEMA_VERSION))
            return
        ver = int(row.value)
        if ver == SCHEMA_VERSION:
            return
        if ver == 1 and SCHEMA_VERSION == 2:
            # PendingExtract created via create_tables; just bump the marker.
            SchemaMeta.update(value="2").where(SchemaMeta.key == "version").execute()
            return
        raise RuntimeError(
            f"unsupported state schema {row.value}; expected {SCHEMA_VERSION}"
        )

    def ensure_session(self, session_id: str, *, default_skill: str = "converse"):
        now = time.time()
        with db.atomic():
            session, created = Session.get_or_create(
                id=session_id,
                defaults={
                    "goal": "",
                    "created_at": now,
                    "updated_at": now,
                },
            )
            if created:
                SkillFrame.create(
                    session=session,
                    skill=default_skill,
                    depth=0,
                    note="",
                    created_at=now,
                )
            return session

    def touch(self, session_id: str):
        Session.update(updated_at=time.time()).where(Session.id == session_id).execute()

    def get_goal(self, session_id: str) -> str:
        s = Session.get_or_none(Session.id == session_id)
        return (s.goal if s else "") or ""

    def set_goal(self, session_id: str, goal: str):
        with db.atomic():
            Session.update(goal=goal or "", updated_at=time.time()).where(
                Session.id == session_id
            ).execute()

    def list_messages(self, session_id: str) -> list[StoredMessage]:
        q = (
            Message.select()
            .where(Message.session == session_id)
            .order_by(Message.seq)
        )
        return [
            StoredMessage(
                id=m.id, role=m.role, content=m.content, kind=m.kind, seq=m.seq
            )
            for m in q
        ]

    def append_message(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        kind: str = "chat",
    ) -> StoredMessage:
        with db.atomic():
            last = (
                Message.select(Message.seq)
                .where(Message.session == session_id)
                .order_by(Message.seq.desc())
                .first()
            )
            seq = (last.seq + 1) if last else 0
            m = Message.create(
                session=session_id,
                role=role,
                content=content or "",
                kind=kind,
                created_at=time.time(),
                seq=seq,
            )
            self.touch(session_id)
            return StoredMessage(
                id=m.id, role=m.role, content=m.content, kind=m.kind, seq=m.seq
            )

    def stack(self, session_id: str) -> list[StoredFrame]:
        q = (
            SkillFrame.select()
            .where(SkillFrame.session == session_id)
            .order_by(SkillFrame.depth)
        )
        return [StoredFrame(skill=f.skill, depth=f.depth, note=f.note) for f in q]

    def active_skill(self, session_id: str) -> str:
        frames = self.stack(session_id)
        return frames[-1].skill if frames else "converse"

    def push_skill(self, session_id: str, skill: str, note: str = ""):
        with db.atomic():
            frames = self.stack(session_id)
            depth = frames[-1].depth + 1 if frames else 0
            SkillFrame.create(
                session=session_id,
                skill=skill,
                depth=depth,
                note=note or "",
                created_at=time.time(),
            )
            self.touch(session_id)

    def pop_skill(self, session_id: str) -> str | None:
        """Pop top frame if depth > 0; never remove the root converse frame."""
        with db.atomic():
            frames = list(
                SkillFrame.select()
                .where(SkillFrame.session == session_id)
                .order_by(SkillFrame.depth.desc())
            )
            if not frames or frames[0].depth == 0:
                return None
            skill = frames[0].skill
            frames[0].delete_instance()
            self.touch(session_id)
            return skill

    def replace_skill(self, session_id: str, skill: str, note: str = ""):
        with db.atomic():
            top = (
                SkillFrame.select()
                .where(SkillFrame.session == session_id)
                .order_by(SkillFrame.depth.desc())
                .first()
            )
            if top is None:
                self.push_skill(session_id, skill, note)
                return
            top.skill = skill
            top.note = note or ""
            top.save()
            self.touch(session_id)

    def add_chunk(
        self,
        session_id: str,
        *,
        chunk_id: str,
        text: str,
        source_message_id: int | None = None,
        kind: str = "raw",
    ) -> StoredChunk | None:
        if not (text or "").strip():
            return None
        with db.atomic():
            existing = MemoryChunk.get_or_none(MemoryChunk.id == chunk_id)
            if existing is not None:
                return StoredChunk(
                    id=existing.id,
                    text=existing.text,
                    source_message_id=existing.source_message_id,
                    kind=existing.kind,
                    indexed=bool(existing.indexed),
                )
            row = MemoryChunk.create(
                id=chunk_id,
                session=session_id,
                source_message_id=source_message_id,
                text=text,
                kind=kind,
                indexed=False,
                created_at=time.time(),
            )
            return StoredChunk(
                id=row.id,
                text=row.text,
                source_message_id=row.source_message_id,
                kind=row.kind,
                indexed=False,
            )

    def mark_indexed(self, chunk_id: str):
        MemoryChunk.update(indexed=True).where(MemoryChunk.id == chunk_id).execute()

    def unindexed_chunks(self, session_id: str) -> list[StoredChunk]:
        q = MemoryChunk.select().where(
            (MemoryChunk.session == session_id) & (MemoryChunk.indexed == False)  # noqa: E712
        )
        return [
            StoredChunk(
                id=c.id,
                text=c.text,
                source_message_id=c.source_message_id,
                kind=c.kind,
                indexed=False,
            )
            for c in q
        ]

    def list_chunks(self, session_id: str) -> list[StoredChunk]:
        q = MemoryChunk.select().where(MemoryChunk.session == session_id)
        return [
            StoredChunk(
                id=c.id,
                text=c.text,
                source_message_id=c.source_message_id,
                kind=c.kind,
                indexed=bool(c.indexed),
            )
            for c in q
        ]

    def add_derived(
        self,
        session_id: str,
        text: str,
        source_message_ids: list[int],
    ):
        if not (text or "").strip():
            return
        ids = ",".join(str(i) for i in source_message_ids)
        DerivedMemory.create(
            session=session_id,
            text=text.strip(),
            source_message_ids=ids,
            created_at=time.time(),
        )

    def list_derived(self, session_id: str) -> list[tuple[str, list[int]]]:
        out = []
        for row in DerivedMemory.select().where(DerivedMemory.session == session_id):
            ids = []
            for part in (row.source_message_ids or "").split(","):
                part = part.strip()
                if part.isdigit():
                    ids.append(int(part))
            out.append((row.text, ids))
        return out

    def enqueue_extract(self, session_id: str, anchor_message_id: int) -> PendingExtractJob:
        row = PendingExtract.create(
            session=session_id,
            anchor_message_id=int(anchor_message_id),
            status="pending",
            created_at=time.time(),
        )
        return PendingExtractJob(
            id=row.id,
            anchor_message_id=row.anchor_message_id,
            status=row.status,
            created_at=row.created_at,
        )

    def list_pending_extracts(self, session_id: str) -> list[PendingExtractJob]:
        q = (
            PendingExtract.select()
            .where(
                (PendingExtract.session == session_id)
                & (PendingExtract.status == "pending")
            )
            .order_by(PendingExtract.id)
        )
        return [
            PendingExtractJob(
                id=r.id,
                anchor_message_id=r.anchor_message_id,
                status=r.status,
                created_at=r.created_at,
            )
            for r in q
        ]

    def count_pending_extracts(self, session_id: str) -> int:
        return (
            PendingExtract.select()
            .where(
                (PendingExtract.session == session_id)
                & (PendingExtract.status == "pending")
            )
            .count()
        )

    def mark_extract_done(self, job_id: int):
        PendingExtract.update(status="done").where(PendingExtract.id == job_id).execute()

    def mark_extract_failed(self, job_id: int):
        PendingExtract.update(status="failed").where(
            PendingExtract.id == job_id
        ).execute()

    def clear_session(self, session_id: str, *, default_skill: str = "converse"):
        """Delete all rows for a session, then recreate an empty root frame."""
        with db.atomic():
            Session.delete().where(Session.id == session_id).execute()
        self.ensure_session(session_id, default_skill=default_skill)
