"""Narrow transactional session API — Peewee stays behind this facade."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from peewee import SqliteDatabase

from .models import SCHEMA_VERSION, build_models


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
class StoredAttachment:
    id: str
    message_id: int
    role: str
    mime: str
    sha256: str
    filename: str
    path: str
    caption: str = ""

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "role": self.role,
            "path": self.path,
            "mime": self.mime,
            "sha256": self.sha256,
            "filename": self.filename,
            "caption": self.caption,
        }


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
        # Own database and model classes: several repositories (state files) can be open in one
        # process, on any threads, without one rebinding the tables of another.
        self.db = SqliteDatabase(str(self.path), pragmas={"foreign_keys": 1, "journal_mode": "wal"})
        self.m = build_models(self.db)
        self.db.connect(reuse_if_open=True)
        self.db.create_tables(self.m.ALL_TABLES)
        self._ensure_schema()

    def close(self):
        if not self.db.is_closed():
            self.db.close()

    def _ensure_schema(self):
        row = self.m.SchemaMeta.get_or_none(self.m.SchemaMeta.key == "version")
        if row is None:
            self.m.SchemaMeta.create(key="version", value=str(SCHEMA_VERSION))
            return
        ver = int(row.value)
        if ver == SCHEMA_VERSION:
            return
        if 1 <= ver < SCHEMA_VERSION:
            # New tables are created above; bump the marker.
            self.m.SchemaMeta.update(value=str(SCHEMA_VERSION)).where(
                self.m.SchemaMeta.key == "version"
            ).execute()
            return
        raise RuntimeError(
            f"unsupported state schema {row.value}; expected {SCHEMA_VERSION}"
        )

    def ensure_session(self, session_id: str, *, default_skill: str = "converse"):
        now = time.time()
        with self.db.atomic():
            session, created = self.m.Session.get_or_create(
                id=session_id,
                defaults={
                    "goal": "",
                    "created_at": now,
                    "updated_at": now,
                },
            )
            if created:
                self.m.SkillFrame.create(
                    session=session,
                    skill=default_skill,
                    depth=0,
                    note="",
                    created_at=now,
                )
            return session

    def touch(self, session_id: str):
        self.m.Session.update(updated_at=time.time()).where(self.m.Session.id == session_id).execute()

    def get_goal(self, session_id: str) -> str:
        s = self.m.Session.get_or_none(self.m.Session.id == session_id)
        return (s.goal if s else "") or ""

    def set_goal(self, session_id: str, goal: str):
        with self.db.atomic():
            self.m.Session.update(goal=goal or "", updated_at=time.time()).where(
                self.m.Session.id == session_id
            ).execute()

    def list_messages(self, session_id: str) -> list[StoredMessage]:
        q = (
            self.m.Message.select()
            .where(self.m.Message.session == session_id)
            .order_by(self.m.Message.seq)
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
        with self.db.atomic():
            last = (
                self.m.Message.select(self.m.Message.seq)
                .where(self.m.Message.session == session_id)
                .order_by(self.m.Message.seq.desc())
                .first()
            )
            seq = (last.seq + 1) if last else 0
            m = self.m.Message.create(
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
            self.m.SkillFrame.select()
            .where(self.m.SkillFrame.session == session_id)
            .order_by(self.m.SkillFrame.depth)
        )
        return [StoredFrame(skill=f.skill, depth=f.depth, note=f.note) for f in q]

    def active_skill(self, session_id: str) -> str:
        frames = self.stack(session_id)
        return frames[-1].skill if frames else "converse"

    def push_skill(self, session_id: str, skill: str, note: str = ""):
        with self.db.atomic():
            frames = self.stack(session_id)
            depth = frames[-1].depth + 1 if frames else 0
            self.m.SkillFrame.create(
                session=session_id,
                skill=skill,
                depth=depth,
                note=note or "",
                created_at=time.time(),
            )
            self.touch(session_id)

    def pop_skill(self, session_id: str) -> str | None:
        """Pop top frame if depth > 0; never remove the root converse frame."""
        with self.db.atomic():
            frames = list(
                self.m.SkillFrame.select()
                .where(self.m.SkillFrame.session == session_id)
                .order_by(self.m.SkillFrame.depth.desc())
            )
            if not frames or frames[0].depth == 0:
                return None
            skill = frames[0].skill
            frames[0].delete_instance()
            self.touch(session_id)
            return skill

    def replace_skill(self, session_id: str, skill: str, note: str = ""):
        with self.db.atomic():
            top = (
                self.m.SkillFrame.select()
                .where(self.m.SkillFrame.session == session_id)
                .order_by(self.m.SkillFrame.depth.desc())
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
        with self.db.atomic():
            existing = self.m.MemoryChunk.get_or_none(self.m.MemoryChunk.id == chunk_id)
            if existing is not None:
                return StoredChunk(
                    id=existing.id,
                    text=existing.text,
                    source_message_id=existing.source_message_id,
                    kind=existing.kind,
                    indexed=bool(existing.indexed),
                )
            row = self.m.MemoryChunk.create(
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
        self.m.MemoryChunk.update(indexed=True).where(self.m.MemoryChunk.id == chunk_id).execute()

    def unindexed_chunks(self, session_id: str) -> list[StoredChunk]:
        q = self.m.MemoryChunk.select().where(
            (self.m.MemoryChunk.session == session_id) & (self.m.MemoryChunk.indexed == False)  # noqa: E712
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
        q = self.m.MemoryChunk.select().where(self.m.MemoryChunk.session == session_id)
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
        self.m.DerivedMemory.create(
            session=session_id,
            text=text.strip(),
            source_message_ids=ids,
            created_at=time.time(),
        )

    def list_derived(self, session_id: str) -> list[tuple[str, list[int]]]:
        out = []
        for row in self.m.DerivedMemory.select().where(self.m.DerivedMemory.session == session_id):
            ids = []
            for part in (row.source_message_ids or "").split(","):
                part = part.strip()
                if part.isdigit():
                    ids.append(int(part))
            out.append((row.text, ids))
        return out

    def _stored_attachment(self, row) -> StoredAttachment:
        return StoredAttachment(
            id=row.id,
            message_id=row.message_id,
            role=row.role,
            mime=row.mime,
            sha256=row.sha256,
            filename=row.filename,
            path=row.path,
            caption=row.caption or "",
        )

    def add_attachment(
        self,
        session_id: str,
        *,
        attachment_id: str,
        message_id: int,
        role: str,
        mime: str,
        sha256: str,
        filename: str,
        path: str,
    ) -> StoredAttachment:
        with self.db.atomic():
            existing = self.m.Attachment.get_or_none(self.m.Attachment.id == attachment_id)
            if existing is not None:
                return self._stored_attachment(existing)
            row = self.m.Attachment.create(
                id=attachment_id,
                session=session_id,
                message_id=int(message_id),
                role=role,
                mime=mime,
                sha256=sha256,
                filename=filename,
                path=path,
                caption="",
            )
            return self._stored_attachment(row)

    def set_caption(self, attachment_id: str, caption: str):
        self.m.Attachment.update(caption=caption or "").where(
            self.m.Attachment.id == attachment_id
        ).execute()

    def list_attachments(self, session_id: str) -> list[StoredAttachment]:
        q = (
            self.m.Attachment.select()
            .where(self.m.Attachment.session == session_id)
            .order_by(self.m.Attachment.message_id, self.m.Attachment.filename)
        )
        return [self._stored_attachment(row) for row in q]

    def find_caption_by_sha256(self, session_id: str, digest: str) -> str | None:
        """Latest non-empty caption for these bytes in the session, if any."""
        text = (digest or "").strip()
        if not text:
            return None
        q = (
            self.m.Attachment.select()
            .where((self.m.Attachment.session == session_id) & (self.m.Attachment.sha256 == text))
            .order_by(self.m.Attachment.message_id.desc())
        )
        for row in q:
            caption = (row.caption or "").strip()
            if caption:
                return caption
        return None

    def captioned_digests(self, session_id: str) -> set[str]:
        """SHA-256 digests in this session that already have a caption."""
        q = self.m.Attachment.select(self.m.Attachment.sha256, self.m.Attachment.caption).where(
            self.m.Attachment.session == session_id
        )
        return {
            row.sha256
            for row in q
            if (row.sha256 or "").strip() and (row.caption or "").strip()
        }

    def find_attachment_by_caption(
        self,
        session_id: str,
        caption: str,
        *,
        message_id: int | None = None,
    ) -> StoredAttachment | None:
        text = (caption or "").strip()
        if not text:
            return None
        q = self.m.Attachment.select().where(self.m.Attachment.session == session_id)
        if message_id is not None:
            q = q.where(self.m.Attachment.message_id == int(message_id))
        for row in q:
            if (row.caption or "").strip() == text:
                return self._stored_attachment(row)
        return None

    def enqueue_extract(self, session_id: str, anchor_message_id: int) -> PendingExtractJob:
        row = self.m.PendingExtract.create(
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
            self.m.PendingExtract.select()
            .where(
                (self.m.PendingExtract.session == session_id)
                & (self.m.PendingExtract.status == "pending")
            )
            .order_by(self.m.PendingExtract.id)
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
            self.m.PendingExtract.select()
            .where(
                (self.m.PendingExtract.session == session_id)
                & (self.m.PendingExtract.status == "pending")
            )
            .count()
        )

    def mark_extract_done(self, job_id: int):
        self.m.PendingExtract.update(status="done").where(self.m.PendingExtract.id == job_id).execute()

    def mark_extract_failed(self, job_id: int):
        self.m.PendingExtract.update(status="failed").where(
            self.m.PendingExtract.id == job_id
        ).execute()

    def clear_session(self, session_id: str, *, default_skill: str = "converse"):
        """Delete all rows for a session, then recreate an empty root frame."""
        with self.db.atomic():
            self.m.Session.delete().where(self.m.Session.id == session_id).execute()
        self.ensure_session(session_id, default_skill=default_skill)
