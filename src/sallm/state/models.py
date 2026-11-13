"""Peewee tables for durable session state (private to the state package)."""

from __future__ import annotations

from types import SimpleNamespace

from peewee import (
    BooleanField,
    CharField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    SqliteDatabase,
    TextField,
)

SCHEMA_VERSION = 3

TABLE_NAMES = (
    "SchemaMeta",
    "Session",
    "Message",
    "SkillFrame",
    "MemoryChunk",
    "DerivedMemory",
    "Attachment",
    "PendingExtract",
)


def build_models(database: SqliteDatabase) -> SimpleNamespace:
    """Model classes bound to *database*.

    Each SessionRepository builds its own set, so several repositories (several state files) can
    be open in one process, even on different threads, without rebinding shared classes.
    """

    bound_db = database  # a class body cannot read a same-named enclosing variable

    class BaseModel(Model):
        class Meta:
            database = bound_db

    class SchemaMeta(BaseModel):
        key = CharField(primary_key=True)
        value = CharField()

    class Session(BaseModel):
        id = CharField(primary_key=True)
        goal = TextField(default="")
        created_at = FloatField()
        updated_at = FloatField()

    class Message(BaseModel):
        id = IntegerField(primary_key=True)
        session = ForeignKeyField(Session, backref="messages", on_delete="CASCADE")
        role = CharField()
        content = TextField()
        kind = CharField(default="chat")  # chat | tool | nudge | system | ingest
        created_at = FloatField()
        seq = IntegerField()  # order within session

    class SkillFrame(BaseModel):
        id = IntegerField(primary_key=True)
        session = ForeignKeyField(Session, backref="frames", on_delete="CASCADE")
        skill = CharField()
        depth = IntegerField()  # 0 = bottom / root
        note = TextField(default="")
        created_at = FloatField()

    class MemoryChunk(BaseModel):
        """Canonical raw chunk text — vector index is rebuildable from these rows."""

        id = CharField(primary_key=True)
        session = ForeignKeyField(Session, backref="chunks", on_delete="CASCADE")
        source_message_id = IntegerField(null=True)
        text = TextField()
        kind = CharField(default="raw")  # raw | derived | fact | ingest | image
        indexed = BooleanField(default=False)
        created_at = FloatField()

    class DerivedMemory(BaseModel):
        """Model-extracted facts; always tied back to raw message ids."""

        id = IntegerField(primary_key=True)
        session = ForeignKeyField(Session, backref="derived", on_delete="CASCADE")
        text = TextField()
        source_message_ids = TextField(default="")  # comma-separated ints
        created_at = FloatField()

    class Attachment(BaseModel):
        """Image file copied beside the session database. Message text stays plain."""

        id = CharField(primary_key=True)
        session = ForeignKeyField(Session, backref="attachments", on_delete="CASCADE")
        message_id = IntegerField()
        role = CharField()  # question | context
        mime = CharField()
        sha256 = CharField()
        filename = CharField()
        path = TextField()
        caption = TextField(default="")

    class PendingExtract(BaseModel):
        """Deferred memory-extract job (queue mode)."""

        id = IntegerField(primary_key=True)
        session = ForeignKeyField(Session, backref="pending_extracts", on_delete="CASCADE")
        anchor_message_id = IntegerField()
        status = CharField(default="pending")  # pending | done | failed
        created_at = FloatField()

    ns = SimpleNamespace(**{name: locals()[name] for name in TABLE_NAMES})
    ns.ALL_TABLES = tuple(getattr(ns, name) for name in TABLE_NAMES)
    return ns


# Unbound module-level set, kept for imports such as ``from sallm.state.models import Message``.
# Repositories never use it: use ``repo.m.<Model>`` to query a repository's own database.
db = SqliteDatabase(None)
_default = build_models(db)
SchemaMeta = _default.SchemaMeta
Session = _default.Session
Message = _default.Message
SkillFrame = _default.SkillFrame
MemoryChunk = _default.MemoryChunk
DerivedMemory = _default.DerivedMemory
Attachment = _default.Attachment
PendingExtract = _default.PendingExtract
ALL_TABLES = _default.ALL_TABLES
