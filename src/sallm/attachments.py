"""Image mentions on a chat line, and OpenAI content parts for any provider.

A user line may drop ``@path`` (part of the question) or ``@context:path``
(background). Bytes are copied next to the session database. The wire format
is the OpenAI content-part list; LiteLLM maps that to each provider.
"""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

_MENTION = re.compile(r"(?<!\S)@(context:)?(\S+)")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class AttachmentError(ValueError):
    """The line named an image that cannot be read."""


@dataclass(frozen=True)
class ImageMention:
    role: str  # question | context
    path: Path
    filename: str


def mime_for(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    return MIME_BY_SUFFIX.get(suffix, "application/octet-stream")


def parse_image_mentions(
    line: str, *, base: Path | None = None
) -> tuple[str, list[ImageMention]]:
    """Split ``@path`` / ``@context:path`` tokens out of a chat line.

    The returned text has those tokens removed. A missing file or a non-image
    suffix raises ``AttachmentError``.
    """
    root = base or Path.cwd()
    mentions: list[ImageMention] = []
    pieces: list[str] = []
    last = 0
    for match in _MENTION.finditer(line or ""):
        pieces.append(line[last : match.start()])
        last = match.end()
        role = "context" if match.group(1) else "question"
        raw = match.group(2).strip().rstrip(".,;:!?")
        if not raw:
            raise AttachmentError("empty image path")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (root / path).resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            raise AttachmentError(f"image not found: {raw}")
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise AttachmentError(f"not an image: {raw}")
        mentions.append(ImageMention(role=role, path=path, filename=path.name))
    pieces.append((line or "")[last:])
    text = " ".join("".join(pieces).split())
    return text, mentions


def copy_image(state_dir: Path, session_id: str, mention: ImageMention) -> tuple[Path, str, str]:
    """Copy image bytes under ``{state_dir}/media/{session}/{sha256}.ext``."""
    data = mention.path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    suffix = mention.path.suffix.lower() or ".png"
    dest = state_dir / "media" / session_id / f"{digest}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(data)
    return dest, digest, mime_for(dest)


def attachment_row_id(session_id: str, message_id: int, role: str, digest: str) -> str:
    blob = f"{session_id}\n{message_id}\n{role}\n{digest}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def memory_attachments(mentions: list[ImageMention]) -> list[dict]:
    """In-memory attachment dicts that point at the original files."""
    out = []
    for index, mention in enumerate(mentions):
        data = mention.path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        out.append(
            {
                "id": f"mem-{index}-{digest[:16]}",
                "role": mention.role,
                "path": str(mention.path),
                "mime": mime_for(mention.path),
                "filename": mention.filename,
                "caption": "",
            }
        )
    return out


def image_part(path: str | Path, mime: str | None = None) -> dict:
    """One OpenAI ``image_url`` part (base64 data URI)."""
    raw = Path(path).read_bytes()
    encoded = base64.standard_b64encode(raw).decode("ascii")
    media = mime or mime_for(path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media};base64,{encoded}"},
    }


def content_parts(text: str, images: list[dict]) -> list[dict]:
    """Text part plus one ``image_url`` part per image dict (``path``, ``mime``)."""
    parts: list[dict] = [{"type": "text", "text": text or ""}]
    for image in images:
        path = image.get("path")
        if not path:
            continue
        parts.append(image_part(path, image.get("mime")))
    return parts


def control_attachment_note(attachments: list[dict], *, rules: str) -> str:
    """Text-only note for the controller. The picture itself is not included."""
    lines = ["Attached images (not shown to you):"]
    for item in attachments:
        name = item.get("filename") or "image"
        role = item.get("role") or "question"
        caption = (item.get("caption") or "").strip()
        if caption:
            lines.append(f"- {name} ({role}). Caption: {caption}")
        else:
            lines.append(f"- {name} ({role}).")
    rules_text = (rules or "").strip()
    if rules_text:
        lines.append(rules_text)
    return "\n".join(lines)


def messages_have_images(messages: list[dict] | None) -> bool:
    for message in messages or []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                return True
    return False
