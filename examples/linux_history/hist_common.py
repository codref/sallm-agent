"""Shared helpers for the linux_history example tools.

Parses bash and zsh history files, applies time windows and context slices,
and formats compact rows so the agent can reconstruct usage stories without
dumping the entire histfile into the prompt.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Soft clip for search rows — keep host/IP-looking tokens when possible.
DEFAULT_CMD_CHARS = 160

# zsh EXTENDED_HISTORY: `: <epoch>:<duration>;<command>`
_ZSH_EXT = re.compile(r"^:\s*(\d+):(\d+);(.*)$", re.DOTALL)

# Rough host / IPv4 tokens — used only when clipping long commands.
_HOSTISH = re.compile(
    r"(?:"
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    r"|"
    r"\b[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+\b"
    r")"
)


@dataclass(frozen=True)
class HistEntry:
    """One history command with optional timestamp."""

    idx: int  # 1-based position in the parsed file (oldest = 1)
    command: str
    epoch: int | None  # Unix seconds when known (zsh extended / bash timed)


def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE from .env into os.environ (no overwrite)."""
    env_path = path or (HERE / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_histfile(explicit: str | Path | None = None) -> Path:
    """Pick the history file: --file, HISTFILE, ~/.zsh_history, ~/.bash_history."""
    load_env()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise SystemExit(f"History file not found: {path}")
        return path

    env_file = (os.environ.get("HISTFILE") or "").strip()
    if env_file:
        path = Path(env_file).expanduser()
        if path.is_file():
            return path

    home = Path.home()
    for candidate in (home / ".zsh_history", home / ".bash_history"):
        if candidate.is_file():
            return candidate

    raise SystemExit(
        "No history file found. Set HISTFILE in the environment or "
        f"{HERE / '.env'}, or pass --file PATH."
    )


def parse_histfile(path: Path) -> list[HistEntry]:
    """Parse bash or zsh history into ordered HistEntry list (oldest first)."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SystemExit(f"Cannot read {path}: {exc}") from exc

    text = raw.decode("utf-8", errors="replace")
    # zsh may use literal escaped newlines; keep simple line split.
    lines = text.splitlines()
    entries: list[HistEntry] = []
    idx = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        # Skip empty / comment-ish
        if not line.strip():
            i += 1
            continue
        m = _ZSH_EXT.match(line)
        if m:
            epoch = int(m.group(1))
            cmd = m.group(3)
            # Continuation: rare; if next lines lack `: n:n;` append until next meta.
            while i + 1 < len(lines) and not _ZSH_EXT.match(lines[i + 1]) and lines[i + 1].startswith("\t"):
                i += 1
                cmd += "\n" + lines[i].lstrip("\t")
            idx += 1
            entries.append(HistEntry(idx=idx, command=cmd.strip("\n"), epoch=epoch))
            i += 1
            continue

        # Plain bash line (or bash HISTTIMEFORMAT comment preceding a command).
        # Bash may store: `#<epoch>` then the command on the next line.
        if line.startswith("#") and len(line) > 1 and line[1:].isdigit():
            epoch = int(line[1:])
            i += 1
            if i >= len(lines):
                break
            cmd = lines[i]
            idx += 1
            entries.append(HistEntry(idx=idx, command=cmd, epoch=epoch))
            i += 1
            continue

        idx += 1
        entries.append(HistEntry(idx=idx, command=line, epoch=None))
        i += 1

    return entries


def parse_time_bound(value: str | None) -> int | None:
    """Parse --since/--until: Unix epoch int, or ISO date/datetime (local → UTC epoch)."""
    if value is None or not str(value).strip():
        return None
    s = str(value).strip()
    if re.fullmatch(r"\d{9,12}", s):
        return int(s)
    # ISO date or datetime
    try:
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        # Allow trailing Z
        s2 = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError as exc:
        raise SystemExit(
            f"Invalid time {value!r}; use Unix epoch or ISO date (YYYY-MM-DD)."
        ) from exc


def days_to_since(days: int) -> int:
    """Epoch for start of the window: now minus N*86400 seconds."""
    now = int(datetime.now(tz=timezone.utc).timestamp())
    return now - max(0, int(days)) * 86400


def shell_tokens(command: str) -> list[str]:
    """Whitespace tokens; strip common quotes and trailing backslashes."""
    out: list[str] = []
    for raw in (command or "").split():
        t = raw.strip().strip("\"'`").rstrip("\\")
        if t:
            out.append(t)
    return out


def matches_query(command: str, query: str, mode: str = "token") -> bool:
    """Return whether ``command`` matches ``query`` under the given mode.

    - ``token`` (default): every query word must appear as a whole shell token
      (fixes ``git`` matching inside path ``.../git-codref/...``).
    - ``substr``: legacy case-insensitive substring.
    - ``semantic``: not used here (scored separately).
    """
    q = (query or "").strip()
    if not q:
        return True
    mode = (mode or "token").strip().lower()
    if mode == "substr":
        return q.lower() in (command or "").lower()
    # token (default)
    q_tokens = [t.lower() for t in q.split() if t.strip()]
    if not q_tokens:
        return True
    cmd_tokens = [t.lower() for t in shell_tokens(command)]
    return all(qt in cmd_tokens for qt in q_tokens)


def filter_entries(
    entries: list[HistEntry],
    *,
    query: str = "",
    since: int | None = None,
    until: int | None = None,
    include_undated: bool = False,
    mode: str = "token",
) -> list[HistEntry]:
    """Filter by query mode and optional time window (not used for semantic ranking)."""
    q = (query or "").strip()
    time_active = since is not None or until is not None
    out: list[HistEntry] = []
    for e in entries:
        if q and mode != "semantic" and not matches_query(e.command, q, mode=mode):
            continue
        if time_active:
            if e.epoch is None:
                if not include_undated:
                    continue
            else:
                if since is not None and e.epoch < since:
                    continue
                if until is not None and e.epoch > until:
                    continue
        out.append(e)
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na**0.5) * (nb**0.5))


def _embed_batch(texts: list[str], *, model: str, api_base: str) -> list[list[float]]:
    """Embed many strings via LiteLLM (Ollama)."""
    from litellm import embedding

    if not texts:
        return []
    # Chunk to keep request size reasonable.
    out: list[list[float]] = []
    chunk_size = 64
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        response = embedding(model=model, input=chunk, api_base=api_base)
        data = response.data
        # Preserve order by index when present.
        by_idx: dict[int, list[float]] = {}
        for j, item in enumerate(data):
            if isinstance(item, dict):
                vec = item.get("embedding")
                idx = int(item.get("index", j))
            else:
                vec = item["embedding"]
                idx = j
            by_idx[idx] = [float(x) for x in list(vec)]
        for j in range(len(chunk)):
            out.append(by_idx[j])
    return out


def semantic_rank(
    entries: list[HistEntry],
    query: str,
    *,
    limit: int = 20,
    pool: int = 3000,
    min_score: float = 0.25,
    model: str = "ollama/qwen3-embedding:0.6b",
    api_base: str = "http://localhost:11434",
) -> list[tuple[HistEntry, float]]:
    """Rank history entries by embedding similarity to ``query`` (newest pool first).

    Uses the same Ollama embedding model as sallm durable retrieval. Scores the
    newest ``pool`` entries (after any prior time filter), returns top ``limit``
    above ``min_score``.
    """
    q = (query or "").strip()
    if not q or not entries:
        return []
    # Prefer newest commands as the semantic corpus.
    corpus = list(reversed(entries))[: max(1, int(pool))]
    texts = [e.command.replace("\n", " ")[:500] for e in corpus]
    q_vec = _embed_batch([q], model=model, api_base=api_base)[0]
    doc_vecs = _embed_batch(texts, model=model, api_base=api_base)
    scored: list[tuple[HistEntry, float]] = []
    for entry, vec in zip(corpus, doc_vecs):
        score = _cosine(q_vec, vec)
        if score >= min_score:
            scored.append((entry, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[: max(1, min(int(limit), 200))]


def expand_context(
    all_entries: list[HistEntry],
    matches: list[HistEntry],
    context: int,
) -> list[HistEntry]:
    """Return unique entries covering each match ± context lines (by idx order)."""
    if context <= 0 or not matches:
        return list(matches)
    by_idx = {e.idx: e for e in all_entries}
    max_idx = max(by_idx) if by_idx else 0
    wanted: set[int] = set()
    for m in matches:
        lo = max(1, m.idx - context)
        hi = min(max_idx, m.idx + context)
        for i in range(lo, hi + 1):
            if i in by_idx:
                wanted.add(i)
    return [by_idx[i] for i in sorted(wanted)]


def format_time(epoch: int | None) -> str:
    if epoch is None:
        return "?"
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except (OSError, OverflowError, ValueError):
        return str(epoch)


def clip_command(cmd: str, max_chars: int = DEFAULT_CMD_CHARS) -> str:
    """Clip long commands; prefer keeping host/IP-looking tokens visible."""
    cmd = cmd.replace("\n", " ").strip()
    if len(cmd) <= max_chars:
        return cmd
    hosts = _HOSTISH.findall(cmd)
    head = max_chars - 40
    if head < 40:
        head = max_chars // 2
    clipped = cmd[:head].rstrip() + "…"
    if hosts:
        # Append distinct host tokens not already in the head.
        extras = []
        for h in hosts:
            if h not in clipped and h not in extras:
                extras.append(h)
            if len(extras) >= 3:
                break
        if extras:
            suffix = " [" + ", ".join(extras) + "]"
            budget = max_chars - len(suffix)
            if budget > 20:
                clipped = cmd[: budget - 1].rstrip() + "…" + suffix
    return clipped[: max_chars + 40]  # allow slight overrun for host suffix


def format_row(
    entry: HistEntry,
    *,
    match: bool = True,
    max_chars: int = DEFAULT_CMD_CHARS,
    score: float | None = None,
) -> str:
    """One compact line. Context (non-match) rows are prefixed with '|'."""
    mark = " " if match else "|"
    score_bit = f" | score={score:.3f}" if score is not None else ""
    return (
        f"{mark}{entry.idx} | {format_time(entry.epoch)}{score_bit} | "
        f"{clip_command(entry.command, max_chars)}"
    )


def search_history(
    *,
    histfile: Path | None = None,
    query: str = "",
    limit: int = 20,
    since: int | None = None,
    until: int | None = None,
    include_undated: bool = False,
    context: int = 0,
    mode: str = "token",
    semantic_pool: int = 3000,
    min_score: float = 0.25,
    embed_model: str = "ollama/qwen3-embedding:0.6b",
    api_base: str = "http://localhost:11434",
) -> tuple[Path, list[tuple[HistEntry, bool, float | None]]]:
    """Run a full search. Returns (path, list of (entry, is_match, score|None))."""
    path = resolve_histfile(histfile)
    all_entries = parse_histfile(path)
    mode = (mode or "token").strip().lower()
    if mode not in ("token", "substr", "semantic"):
        raise SystemExit(f"Unknown --mode {mode!r}; use token|substr|semantic")

    limit = max(1, min(int(limit), 200))
    scores: dict[int, float] = {}

    if mode == "semantic" and (query or "").strip():
        timed = filter_entries(
            all_entries,
            query="",
            since=since,
            until=until,
            include_undated=include_undated,
            mode="token",
        )
        ranked = semantic_rank(
            timed,
            query,
            limit=limit,
            pool=semantic_pool,
            min_score=min_score,
            model=embed_model,
            api_base=api_base,
        )
        selected = [e for e, _s in ranked]
        scores = {e.idx: s for e, s in ranked}
    else:
        matches = filter_entries(
            all_entries,
            query=query,
            since=since,
            until=until,
            include_undated=include_undated,
            mode=mode,
        )
        matches_newest = list(reversed(matches))
        selected = matches_newest[:limit]

    match_idxs = {e.idx for e in selected}

    if context <= 0:
        rows = [(e, True, scores.get(e.idx)) for e in selected]
        return path, rows

    by_idx = {e.idx: e for e in all_entries}
    max_idx = max(by_idx) if by_idx else 0
    seen: set[int] = set()
    rows: list[tuple[HistEntry, bool, float | None]] = []
    for m in selected:
        lo = max(1, m.idx - context)
        hi = min(max_idx, m.idx + context)
        window_idxs = [i for i in range(lo, hi + 1) if i in by_idx and i not in seen]
        if not window_idxs:
            continue
        for i in window_idxs:
            seen.add(i)
            rows.append((by_idx[i], i in match_idxs, scores.get(i)))
        rows.append((HistEntry(idx=-1, command="", epoch=None), False, None))
    if rows and rows[-1][0].idx == -1:
        rows.pop()
    return path, rows
