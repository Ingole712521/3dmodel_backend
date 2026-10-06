import re
from pathlib import Path

from app.config import MAX_IMAGE_BYTES

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str, fallback: str) -> str:
    cleaned = _SAFE_NAME.sub("-", Path(name).name).strip(".-")
    if not cleaned:
        cleaned = fallback
    return cleaned[:120]


def image_kind(payload: bytes) -> str | None:
    if payload.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    return None


def validate_image(filename: str, payload: bytes) -> str:
    if len(payload) == 0:
        raise ValueError(f"{filename} is empty.")
    if len(payload) > MAX_IMAGE_BYTES:
        raise ValueError(f"{filename} is larger than 25 MB.")
    kind = image_kind(payload)
    if kind is None:
        raise ValueError(f"{filename} must be a jpg, jpeg, or png photo.")
    return ".png" if kind == "png" else ".jpg"
