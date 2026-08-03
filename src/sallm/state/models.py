"""Peewee tables for durable session state (private to the state package)."""

from __future__ import annotations

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

SCHEMA_VERSION = 2

# Bound per repository open(); models share this proxy.
db = SqliteDatabase(None)


class BaseModel(Model):
    class Meta:
        database = db


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
    kind = CharField(default="chat")  # chat | tool | nudge | system
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
    kind = CharField(default="raw")  # raw | derived
    indexed = BooleanField(default=False)
    created_at = FloatField()


class DerivedMemory(BaseModel):
    """Model-extracted facts; always tied back to raw message ids."""

    id = IntegerField(primary_key=True)
    session = ForeignKeyField(Session, backref="derived", on_delete="CASCADE")
    text = TextField()
    source_message_ids = TextField(default="")  # comma-separated ints
    created_at = FloatField()


class PendingExtract(BaseModel):
    """Deferred memory-extract job (queue mode)."""

    id = IntegerField(primary_key=True)
    session = ForeignKeyField(Session, backref="pending_extracts", on_delete="CASCADE")
    anchor_message_id = IntegerField()
    status = CharField(default="pending")  # pending | done | failed
    created_at = FloatField()


ALL_TABLES = (
    SchemaMeta,
    Session,
    Message,
    SkillFrame,
    MemoryChunk,
    DerivedMemory,
    PendingExtract,
)
