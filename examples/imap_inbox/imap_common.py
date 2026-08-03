"""Shared IMAP helpers for the inbox example tools.

Each tool is a separate CLI subprocess. They import this module so connect,
credential loading, and compact formatting stay in one place — not duplicated
across three scripts, and not mixed into the agent REPL.
"""

from __future__ import annotations

import email
import email.header
import email.utils
import imaplib
import os
import re
from email.message import Message
from pathlib import Path
from typing import Iterator

# Directory that holds .env next to these scripts.
HERE = Path(__file__).resolve().parent

# Soft cap used when a tool does not pass --max-chars.
DEFAULT_BODY_CHARS = 1500


def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (no overwrite).

    Tiny loader — no python-dotenv dependency. Lines starting with # and blank
    lines are ignored. Values may be optionally quoted with " or '.
    """
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
        # Do not clobber variables already set in the shell.
        if key and key not in os.environ:
            os.environ[key] = value


def require_credentials() -> dict[str, str]:
    """Return IMAP settings from the environment, or raise SystemExit."""
    load_env()
    host = (os.environ.get("IMAP_HOST") or "").strip()
    user = (os.environ.get("IMAP_USER") or "").strip()
    password = os.environ.get("IMAP_PASSWORD") or ""
    missing = [
        name
        for name, val in (
            ("IMAP_HOST", host),
            ("IMAP_USER", user),
            ("IMAP_PASSWORD", password),
        )
        if not val
    ]
    if missing:
        raise SystemExit(
            "Missing credentials: "
            + ", ".join(missing)
            + f". Copy {HERE / '.env.example'} to {HERE / '.env'} and fill in."
        )
    port = (os.environ.get("IMAP_PORT") or "993").strip()
    ssl_raw = (os.environ.get("IMAP_SSL") or "true").strip().lower()
    folder = (os.environ.get("IMAP_FOLDER") or "INBOX").strip() or "INBOX"
    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "ssl": ssl_raw in ("1", "true", "yes", "on"),
        "folder": folder,
    }


def connect() -> tuple[imaplib.IMAP4, dict[str, str]]:
    """Open an IMAP connection and log in. Caller must logout()."""
    cfg = require_credentials()
    port = int(cfg["port"])
    if cfg["ssl"]:
        client: imaplib.IMAP4 = imaplib.IMAP4_SSL(cfg["host"], port)
    else:
        client = imaplib.IMAP4(cfg["host"], port)
    client.login(cfg["user"], cfg["password"])
    return client, cfg


def select_folder(client: imaplib.IMAP4, folder: str) -> None:
    """Select a mailbox (read-only). Raise on failure."""
    status, _ = client.select(folder, readonly=True)
    if status != "OK":
        raise SystemExit(f"Cannot select folder {folder!r} (status={status})")


def decode_header_value(raw: str | None) -> str:
    """Decode an RFC 2047 header into a plain Unicode string."""
    if not raw:
        return ""
    parts: list[str] = []
    for fragment, charset in email.header.decode_header(raw):
        if isinstance(fragment, bytes):
            parts.append(fragment.decode(charset or "utf-8", errors="replace"))
        else:
            parts.append(fragment)
    return "".join(parts).strip()


def header_line(msg: Message, name: str) -> str:
    return decode_header_value(msg.get(name))


def format_search_row(uid: str, msg: Message) -> str:
    """One compact line for search results — easy for the model and for extract."""
    date = header_line(msg, "Date") or "?"
    # Prefer a short date if parseable.
    try:
        parsed = email.utils.parsedate_to_datetime(msg.get("Date"))
        if parsed is not None:
            date = parsed.strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, IndexError):
        pass
    frm = header_line(msg, "From") or "?"
    subject = header_line(msg, "Subject") or "(no subject)"
    # Keep each field short so a 10-row search stays under the history budget.
    frm = _clip(frm, 60)
    subject = _clip(subject, 80)
    return f"{uid} | {date} | {frm} | {subject}"


def extract_text_body(msg: Message, max_chars: int = DEFAULT_BODY_CHARS) -> str:
    """Best-effort plaintext body, truncated. Prefer text/plain over HTML."""
    plain_chunks: list[str] = []
    html_chunks: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            ctype = (part.get_content_type() or "").lower()
            disp = str(part.get("Content-Disposition") or "").lower()
            if "attachment" in disp:
                continue
            payload = part.get_payload(decode=True)
            if not isinstance(payload, bytes):
                continue
            charset = part.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            if ctype == "text/plain":
                plain_chunks.append(text)
            elif ctype == "text/html":
                html_chunks.append(text)
    else:
        payload = msg.get_payload(decode=True)
        if isinstance(payload, bytes):
            charset = msg.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            ctype = (msg.get_content_type() or "").lower()
            if ctype == "text/html":
                html_chunks.append(text)
            else:
                plain_chunks.append(text)

    if plain_chunks:
        body = "\n".join(plain_chunks)
    elif html_chunks:
        # Crude strip — good enough for a snippet, not a browser.
        body = re.sub(r"<[^>]+>", " ", html_chunks[0])
        body = re.sub(r"\s+", " ", body).strip()
    else:
        body = "(no text body)"

    body = body.strip()
    if len(body) > max_chars:
        body = body[: max_chars - 20].rstrip() + "\n… [truncated]"
    return body


def format_fetch(uid: str, msg: Message, max_chars: int = DEFAULT_BODY_CHARS) -> str:
    """Structured fetch block: headers first, then a capped body snippet."""
    lines = [
        f"UID: {uid}",
        f"Date: {header_line(msg, 'Date')}",
        f"From: {header_line(msg, 'From')}",
        f"To: {header_line(msg, 'To')}",
        f"Subject: {header_line(msg, 'Subject')}",
        f"Message-ID: {header_line(msg, 'Message-ID')}",
        "",
        "Body:",
        extract_text_body(msg, max_chars=max_chars),
    ]
    return "\n".join(lines)


def list_mailbox_names(client: imaplib.IMAP4) -> list[str]:
    """Return decoded mailbox names from LIST."""
    status, data = client.list()
    if status != "OK" or not data:
        return []
    names: list[str] = []
    for item in data:
        if not item:
            continue
        raw = item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item)
        # LIST replies look like: (\\HasNoChildren) "/" "INBOX"
        # Take the last quoted token, or the last whitespace token.
        quoted = re.findall(r'"([^"]*)"', raw)
        if quoted:
            names.append(quoted[-1])
        else:
            names.append(raw.rsplit(None, 1)[-1])
    return names


def fetch_headers(client: imaplib.IMAP4, uids: list[str]) -> Iterator[tuple[str, Message]]:
    """Yield (uid, message) for HEADER-only fetches."""
    if not uids:
        return
    # Batch fetch keeps tool latency reasonable for --limit 10–20.
    uid_set = ",".join(uids)
    status, data = client.uid("fetch", uid_set, "(BODY.PEEK[HEADER])")
    if status != "OK" or not data:
        return
    yield from _parse_fetch_pairs(data)


def fetch_full(client: imaplib.IMAP4, uid: str) -> Message | None:
    """Fetch one full RFC822 message by UID."""
    status, data = client.uid("fetch", uid, "(BODY.PEEK[])")
    if status != "OK" or not data:
        return None
    for _uid, msg in _parse_fetch_pairs(data):
        return msg
    return None


def _parse_fetch_pairs(data) -> Iterator[tuple[str, Message]]:
    """Parse imaplib fetch tuples into (uid, Message)."""
    i = 0
    while i < len(data):
        item = data[i]
        if not isinstance(item, tuple) or len(item) < 2:
            i += 1
            continue
        meta, payload = item[0], item[1]
        if not isinstance(payload, (bytes, bytearray)):
            i += 1
            continue
        meta_s = meta.decode("utf-8", errors="replace") if isinstance(meta, bytes) else str(meta)
        m = re.search(r"UID\s+(\d+)", meta_s, re.IGNORECASE)
        uid = m.group(1) if m else "?"
        yield uid, email.message_from_bytes(bytes(payload))
        i += 1


def resolve_search_criteria(query: str) -> str:
    """Map short presets to IMAP SEARCH criteria; otherwise normalize and pass through.

    Important: human phrases like "recent / latest / newest mail" must NOT use
    the IMAP ``RECENT`` flag — that flag is often empty on Gmail and similar
    hosts. Those presets map to ``ALL``; ``--limit`` then keeps the newest UIDs.

    Also normalizes quoting: models often emit ``FROM 'addr'`` (single quotes),
    which IMAP does not treat as a string — use double quotes or bare atoms.
    """
    q = (query or "").strip()
    if not q:
        return "ALL"
    key = q.upper()
    # "Most recent N messages" → ALL + --limit (not IMAP \Recent).
    latest_aliases = {
        "ALL",
        "LATEST",
        "NEWEST",
        "RECENT",
        "NEW",
    }
    if key in latest_aliases:
        return "ALL"
    presets = {
        "UNSEEN": "UNSEEN",
        "SEEN": "SEEN",
        "UNREAD": "UNSEEN",
    }
    if key in presets:
        return presets[key]
    return normalize_imap_query(q)


def normalize_imap_query(query: str) -> str:
    """Fix common model mistakes in IMAP SEARCH criteria strings.

    - Single-quoted values → double-quoted (IMAP astring uses \")
    - ``FROM user@host`` without quotes is fine (atom-safe)
    - Bare email / ``from:user@host`` → ``FROM "user@host"``
    """
    q = (query or "").strip()
    if not q:
        return "ALL"

    # from:addr / to:addr shortcuts (Gmail-ish)
    m = re.match(r"^(from|to|subject|cc):\s*(.+)$", q, re.IGNORECASE)
    if m:
        key = m.group(1).upper()
        return f'{key} {_imap_string(m.group(2).strip())}'

    # Bare email address → FROM
    if "@" in q and not re.search(r"\b(FROM|TO|SUBJECT|CC|OR|NOT|AND)\b", q, re.I):
        if re.fullmatch(r"[^\"'\s]+@[^\"'\s]+", q):
            return f"FROM {_imap_string(q)}"

    # Replace 'value' with "value" for header keys that take a string.
    def _fix_single(match: re.Match) -> str:
        key = match.group(1).upper()
        return f"{key} {_imap_string(match.group(2))}"

    q = re.sub(
        r"\b(FROM|TO|SUBJECT|CC)\s+'([^']*)'",
        _fix_single,
        q,
        flags=re.IGNORECASE,
    )
    # Also FROM "…" already fine; FROM <unquoted> fine.
    # Collapse smart/curly quotes if the model used them.
    q = q.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    q = re.sub(
        r"\b(FROM|TO|SUBJECT|CC)\s+'([^']*)'",
        _fix_single,
        q,
        flags=re.IGNORECASE,
    )
    return q


def _imap_string(value: str) -> str:
    """Quote a value for IMAP SEARCH (double quotes; escape internal quotes)."""
    value = (value or "").strip().strip('"').strip("'")
    # Simple atoms (email, word) can stay unquoted — clearer for servers.
    if value and re.fullmatch(r"[\w.+/=@-]+", value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_search_criteria(
    *,
    query: str | None = None,
    from_addr: str | None = None,
    subject: str | None = None,
) -> str:
    """Combine --query / --from / --subject into one IMAP SEARCH criteria string."""
    parts: list[str] = []
    if from_addr and from_addr.strip():
        parts.append(f"FROM {_imap_string(from_addr)}")
    if subject and subject.strip():
        parts.append(f"SUBJECT {_imap_string(subject)}")
    q = (query or "").strip()
    if q and q.upper() not in ("ALL", "LATEST", "NEWEST", "RECENT", "NEW"):
        # If only presets + from/subject, resolve_search_criteria handles ALL.
        if q.upper() in ("UNSEEN", "SEEN", "UNREAD"):
            parts.insert(0, resolve_search_criteria(q))
        else:
            parts.append(resolve_search_criteria(q))
    elif q and not parts:
        return resolve_search_criteria(q)
    if not parts:
        return "ALL"
    if len(parts) == 1:
        return parts[0]
    return "(" + " ".join(parts) + ")"


def _clip(text: str, n: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"
