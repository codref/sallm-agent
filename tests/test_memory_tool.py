"""Memory CliTool — file backend (no LLM / Lance required)."""

from __future__ import annotations

import pytest

from sallm.tools import builtin_tools, run_tool
from sallm.tools.memory.backend import FileStore, chunk_text


def test_chunk_short_passthrough():
    assert chunk_text("hello world") == ["hello world"]


def test_chunk_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_long_splits_with_overlap():
    text = ("word " * 200).strip()
    chunks = chunk_text(text, max_tokens=50, overlap_tokens=10)
    assert len(chunks) >= 2
    joined = "\n".join(chunks)
    assert "word" in joined


def test_file_store_session_scoped(tmp_path):
    store = FileStore(tmp_path / "mem")
    store.add("alpha vault code is 42", session_id="a")
    store.add("beta unrelated note", session_id="b")
    hits = store.query("vault code", k=3, session_id="a")
    assert hits
    assert "42" in hits[0]
    store.clear(session_id="a")
    assert store.query("vault", k=3, session_id="a") == []
    assert store.query("beta", k=3, session_id="b")


def test_memory_cli_add_search_clear(tmp_path):
    path = tmp_path / "cli-mem"
    tools = builtin_tools(
        ("memory",),
        memory_path=path,
        memory_session="t1",
        memory_backend="file",
    )
    mem = tools["memory"]

    add = run_tool(
        mem,
        ["add", "--text", "The vault code is 9911. Dale mentioned IAS service."],
    )
    assert add.returncode == 0
    assert "chunk" in add.observation.lower()

    # Quoted single argv (shlex-style) still works.
    search = run_tool(mem, ["search", "--query", "vault code IAS", "-k", "2"])
    assert search.returncode == 0
    assert "9911" in search.observation or "IAS" in search.observation

    clear = run_tool(mem, ["clear"])
    assert clear.returncode == 0
    empty = run_tool(mem, ["search", "--query", "vault"])
    assert empty.observation.strip() == "(no matches)"


def test_memory_search_unquoted_multiword(tmp_path):
    """Models often omit quotes: --query a b c must not argparse-fail."""
    path = tmp_path / "cli-mem2"
    tools = builtin_tools(
        ("memory",),
        memory_path=path,
        memory_session="t2",
        memory_backend="file",
    )
    mem = tools["memory"]
    add = run_tool(
        mem,
        [
            "add",
            "--text",
            "developers",
            "used",
            "an",
            "outbox",
            "before",
            "streaming",
        ],
    )
    assert add.returncode == 0, add.observation

    search = run_tool(
        mem,
        [
            "search",
            "--query",
            "developers",
            "store",
            "data",
            "method",
            "before",
            "streaming",
            "-k",
            "2",
        ],
    )
    assert search.returncode == 0, search.observation
    assert "RuntimeWarning" not in (search.stderr or "")
    assert "unrecognized arguments" not in search.observation
    assert "outbox" in search.observation.lower() or "streaming" in search.observation.lower()


def test_lance_store_optional(tmp_path):
    pytest.importorskip("lancedb")
    from sallm.tools.memory.backend import LanceStore

    def embed(text: str):
        raw = [float(ord(c) % 7) for c in (text or "x")[:8]]
        while len(raw) < 8:
            raw.append(0.0)
        return raw[:8]

    store = LanceStore(tmp_path / "lance", embed_fn=embed, dimensions=8)
    store.add("golden ticket number seven", session_id="s")
    hits = store.query("golden ticket", k=2, session_id="s")
    assert hits
    assert "golden" in hits[0].lower()
